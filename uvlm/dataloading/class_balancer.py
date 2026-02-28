"""
General dataset balancing module - can be automatically called during training

Use cls_columns parameter to specify columns to balance (passed from plan)

Main features:
- Strictly limits maximum repetitions per original sample
- Outputs statistics for all categories before and after balancing (compact log printing + full txt file saving)
- All categories in logs are sorted by count descending, with 4 key-value pairs per line (to avoid flooding)
- txt file saves complete list of all categories (one per line, with alignment)

Usage example:
    balance_csv_files(
        csv_paths=["/path/to/train.csv"],
        cls_columns=["lung_nodule", ...],
        target_samples_per_class=20000,
        ...
    )
"""

import os
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
import random


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV file"""
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def save_csv(df: pd.DataFrame, csv_path: str):
    """Save DataFrame to CSV file"""
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    df.to_csv(csv_path, index=False, encoding='utf-8')


def extract_labels_from_csv(
    df: pd.DataFrame,
    cls_columns: List[str]
) -> Tuple[Dict[int, Set[str]], List[str]]:
    """
    Extract positive label sets for each sample from CSV

    Args:
        df: DataFrame
        cls_columns: List of classification column names (required)

    Returns:
        sample_labels: {sample index: {label set}}
        used_columns: List of actually used column names
    """
    sample_labels = {}

    # Verify columns exist
    used_columns = [col for col in cls_columns if col in df.columns]
    if len(used_columns) != len(cls_columns):
        missing = set(cls_columns) - set(used_columns)
        print(f"WARNING: The following columns do not exist in CSV and will be ignored: {missing}")

    if not used_columns:
        raise ValueError(f"None of the columns in cls_columns exist in CSV: {cls_columns}")

    for idx, row in df.iterrows():
        labels = set()
        for col in used_columns:
            val = row.get(col)
            if pd.notna(val) and val == 1:
                labels.add(col)
        sample_labels[idx] = labels

    return sample_labels, used_columns


def count_classes(sample_labels: Dict[int, Set[str]]) -> Dict[str, int]:
    """Count occurrences for each category"""
    counter = Counter()
    for labels in sample_labels.values():
        counter.update(labels)
    return dict(counter)


def balance_dataset_indices(
    sample_labels: Dict[int, Set[str]],
    class_counts: Dict[str, int],
    target_samples_per_class: int = 8000,
    max_times_per_sample: int = 5,
    random_seed: int = 42,
    negative_ratio: float = 0.18,
    logger=None
) -> List[int]:
    """
    Balancing strategy with strict repetition control
    Each original sample appears at most max_times_per_sample times
    """
    random.seed(random_seed)

    labeled_indices = [idx for idx, lbls in sample_labels.items() if lbls]
    negative_indices = [idx for idx, lbls in sample_labels.items() if not lbls]

    remaining_uses = {idx: max_times_per_sample for idx in sample_labels}

    class_to_samples = defaultdict(list)
    for idx in labeled_indices:
        for label in sample_labels[idx]:
            class_to_samples[label].append(idx)

    class_targets = {}
    for cls, samples in class_to_samples.items():
        unique_count = len(set(samples))
        max_possible = unique_count * max_times_per_sample
        target = min(target_samples_per_class, max_possible)
        class_targets[cls] = target

    current_counts = {cls: 0 for cls in class_targets}

    selected_indices = []

    sorted_classes = sorted(
        class_targets.keys(),
        key=lambda c: class_counts.get(c, 0)
    )

    for cls in sorted_classes:
        target = class_targets[cls]
        needed = target - current_counts.get(cls, 0)
        if needed <= 0:
            continue

        candidates = class_to_samples[cls]
        valid = [idx for idx in candidates if remaining_uses[idx] > 0]

        if not valid:
            if logger:
                logger(f"Category {cls} has exhausted available repetitions, cannot supplement further (still need {needed})")
            continue

        valid.sort(key=lambda i: len(sample_labels[i]))

        take = min(len(valid), needed)
        chosen = valid[:take]

        for idx in chosen:
            selected_indices.append(idx)
            remaining_uses[idx] -= 1
            for label in sample_labels[idx]:
                current_counts[label] = current_counts.get(label, 0) + 1

    # Supplement negative samples
    n_labeled = sum(1 for i in selected_indices if sample_labels[i])
    if n_labeled > 0:
        target_total = int(n_labeled / (1 - negative_ratio))
        needed_neg = max(0, target_total - n_labeled)
    else:
        needed_neg = 0

    if needed_neg > 0 and negative_indices:
        valid_neg = [idx for idx in negative_indices if remaining_uses[idx] > 0]

        if valid_neg:
            take = min(len(valid_neg), needed_neg)
            chosen_neg = random.sample(valid_neg, take)
            for idx in chosen_neg:
                selected_indices.append(idx)
                remaining_uses[idx] -= 1

            if logger and len(chosen_neg) < needed_neg:
                logger(f"Insufficient negative samples, only supplemented {len(chosen_neg)} / {needed_neg}")

    if logger:
        actual_max = max(max_times_per_sample - remaining_uses.get(idx, 0)
                         for idx in remaining_uses) if remaining_uses else 0
        logger(f"Maximum repetitions per sample after balancing: {actual_max} / limit {max_times_per_sample}")

    return selected_indices


def save_balancing_statistics(
    original_class_counts: Dict[str, int],
    balanced_class_counts: Dict[str, int],
    original_sample_labels: Dict[int, Set[str]],
    balanced_sample_labels: Dict[int, Set[str]],
    output_dir: str,
    logger=None
):
    """Save complete statistics before and after balancing (all categories, colon-aligned)"""
    os.makedirs(output_dir, exist_ok=True)

    stats_file = os.path.join(output_dir, "balancing_statistics.txt")

    # Find the longest name length among all categories (considering both original and balanced)
    all_classes = set(original_class_counts.keys()) | set(balanced_class_counts.keys())
    if all_classes:
        max_name_length = max(len(cls) for cls in all_classes)
    else:
        max_name_length = 10  # Minimum width to prevent empty case

    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=== Dataset Balancing Statistics ===\n\n")

        f.write(f"Original total samples: {len(original_sample_labels)}\n")
        f.write(f"Balanced total samples: {len(balanced_sample_labels)}\n\n")

        # Original - all categories, colon-aligned
        f.write("Original distribution of all categories (sorted by count descending):\n")
        sorted_orig = sorted(original_class_counts.items(), key=lambda x: x[1], reverse=True)
        for cls, cnt in sorted_orig:
            f.write(f"  {cls:<{max_name_length}} : {cnt:>8d}\n")

        f.write("\n")

        # Balanced - all categories, colon-aligned
        f.write("Balanced distribution of all categories (sorted by count descending):\n")
        sorted_bal = sorted(balanced_class_counts.items(), key=lambda x: x[1], reverse=True)
        for cls, cnt in sorted_bal:
            f.write(f"  {cls:<{max_name_length}} : {cnt:>8d}\n")

    if logger:
        logger(f"Complete statistics saved to: {stats_file} (category names aligned)")


def balance_csv_files(
    csv_paths: List[str],
    cls_columns: List[str],
    target_samples_per_class: int = 8000,
    max_times_per_sample: int = 5,
    random_seed: int = 42,
    negative_ratio: float = 0.18,
    output_path: Optional[str] = None,
    stats_output_dir: Optional[str] = None,
    logger=None
) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    """
    Main function: Read multiple CSVs → merge → balance → save results

    Args:
        csv_paths: List of CSV file paths
        cls_columns: List of classification column names for balancing (required)
        target_samples_per_class: Target number of samples per class
        max_times_per_sample: Maximum repetitions per sample
        random_seed: Random seed
        negative_ratio: Ratio of negative samples
        output_path: Output file path
        stats_output_dir: Statistics output directory
        logger: Logging function

    Returns:
        (List of output paths, original class statistics, balanced class statistics)
    """
    all_dfs = []
    for path in csv_paths:
        df = load_csv(path)
        if not df.empty:
            all_dfs.append(df)
            if logger:
                logger(f"Loaded CSV: {path}, sample count: {len(df)}")
        else:
            if logger:
                logger(f"File invalid or empty: {path}")

    if not all_dfs:
        if logger:
            logger("No valid data loaded, returning original paths")
        return csv_paths, {}, {}

    combined_df = pd.concat(all_dfs, ignore_index=True)

    if logger:
        logger(f"Total samples after merging: {len(combined_df)}")

    # Extract labels (using cls_columns)
    sample_labels, used_columns = extract_labels_from_csv(combined_df, cls_columns)

    if logger:
        logger(f"Using classification columns for balancing: {used_columns}")

    original_class_counts = count_classes(sample_labels)

    # Print original distribution of all categories (compact format)
    if logger:
        logger("=" * 80)
        logger("Original distribution of all categories (sorted by count descending):")

        sorted_orig = sorted(original_class_counts.items(), key=lambda x: x[1], reverse=True)
        line_parts = []
        for i, (cls, cnt) in enumerate(sorted_orig):
            line_parts.append(f"{cls}:{cnt}")
            if len(line_parts) == 4 or i == len(sorted_orig)-1:
                logger("  " + "  ".join(line_parts))
                line_parts = []

        logger("")

    # Perform balancing
    selected_indices = balance_dataset_indices(
        sample_labels,
        original_class_counts,
        target_samples_per_class=target_samples_per_class,
        max_times_per_sample=max_times_per_sample,
        random_seed=random_seed,
        negative_ratio=negative_ratio,
        logger=logger
    )

    balanced_df = combined_df.iloc[selected_indices].reset_index(drop=True)

    balanced_labels, _ = extract_labels_from_csv(balanced_df, cls_columns)
    balanced_class_counts = count_classes(balanced_labels)

    # Print balanced distribution of all categories (compact format)
    if logger:
        logger("Balanced distribution of all categories (sorted by count descending):")

        sorted_bal = sorted(balanced_class_counts.items(), key=lambda x: x[1], reverse=True)
        line_parts = []
        for i, (cls, cnt) in enumerate(sorted_bal):
            line_parts.append(f"{cls}:{cnt}")
            if len(line_parts) == 4 or i == len(sorted_bal)-1:
                logger("  " + "  ".join(line_parts))
                line_parts = []

        logger("")
        logger(f"Total samples after balancing: {len(balanced_df)}")
        logger("=" * 80)

    # Save balanced data
    if output_path is None:
        import tempfile
        output_path = os.path.join(tempfile.gettempdir(), f"balanced_dataset_{random_seed}.csv")

    save_csv(balanced_df, output_path)

    if logger:
        logger(f"Balanced dataset saved to: {output_path}")

    # Save complete statistics file
    if stats_output_dir:
        save_balancing_statistics(
            original_class_counts,
            balanced_class_counts,
            sample_labels,
            balanced_labels,
            stats_output_dir,
            logger
        )

    return [output_path], original_class_counts, balanced_class_counts
