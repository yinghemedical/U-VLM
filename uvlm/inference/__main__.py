"""
U-VLM Inference Entry Point

Usage:
    # Classification inference
    uvlm_inference cls \
        --csv-path /path/to/test.csv \
        --model-dir /path/to/model \
        --output-dir /path/to/output

    # Segmentation inference
    uvlm_inference seg \
        --csv-path /path/to/test.csv \
        --model-dir /path/to/model \
        --output-dir /path/to/output

    # Report generation inference
    uvlm_inference report \
        --csv-path /path/to/test.csv \
        --model-dir /path/to/model \
        --output-dir /path/to/output
"""

import argparse
import sys
from pathlib import Path


def parse_model_path(model_dir):
    """Parse model directory path to extract nnUNet components.

    Args:
        model_dir: Full path like .../nnUNet_results/DATASET/TRAINER__PLANS__CONFIG/fold_X

    Returns:
        dict with dataset_name, trainer_name, plans_name, configuration_name, base_results_dir, fold
    """
    model_path = Path(model_dir)
    parts = model_path.parts

    nnunet_idx = parts.index('nnUNet_results') if 'nnUNet_results' in parts else -1

    result = {
        'dataset_name': None,
        'trainer_name': None,
        'plans_name': None,
        'configuration_name': None,
        'base_results_dir': str(model_path.parent.parent) if nnunet_idx >= 0 else str(model_path),
        'fold': 0
    }

    if nnunet_idx >= 0 and len(parts) >= nnunet_idx + 3:
        result['dataset_name'] = parts[nnunet_idx + 1]
        trainer_plans_config = parts[nnunet_idx + 2]
        split_parts = trainer_plans_config.split('__')
        if len(split_parts) >= 3:
            result['trainer_name'] = split_parts[0]
            result['plans_name'] = split_parts[1]
            result['configuration_name'] = split_parts[2]
        else:
            result['trainer_name'] = 'nnUNetTrainer_UVLM'
            result['plans_name'] = split_parts[0] if len(split_parts) >= 1 else 'Plans'
            result['configuration_name'] = split_parts[1] if len(split_parts) >= 2 else '3d_fullres'

        for part in parts:
            if part.startswith('fold_'):
                fold_str = part.split('_')[1]
                if fold_str.isdigit():
                    result['fold'] = int(fold_str)
                break

    return result


def main():
    parser = argparse.ArgumentParser(description='U-VLM Inference')
    parser.add_argument('task', type=str, choices=['cls', 'seg', 'report'],
                        help='Inference task type')
    parser.add_argument('--csv-path', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--gpu-config', type=str, default='0:1',
                        help='GPU config like "0:2,1:2" (gpu_id:num_workers)')
    parser.add_argument('--checkpoint', type=str, default='best',
                        choices=['best', 'final'],
                        help='Checkpoint to use')
    parser.add_argument('--debug-max-cases', type=int, default=None,
                        help='Limit number of cases for debugging')

    args = parser.parse_args()

    if args.task == 'cls':
        # Build sys.argv for inference_cls
        # inference_cls expects nnUNet-style arguments: --csv-path, --base-results-dir, --dataset-name,
        # --trainer-name, --plans-name, --configuration-name, --fold, --checkpoint-name, --output-suffix
        parsed = parse_model_path(args.model_dir)
        checkpoint_name = f"checkpoint_{args.checkpoint}.pth"
        new_argv = [
            '--csv-path', args.csv_path,
            '--dataset-name', parsed['dataset_name'] or 'Dataset001_UVLM',
            '--trainer-name', parsed['trainer_name'] or 'nnUNetTrainer_UVLM',
            '--plans-name', parsed['plans_name'] or 'UVLM_ResEncUNetLPlans',
            '--configuration-name', parsed['configuration_name'] or '3d_fullres',
            '--fold', str(parsed['fold']),
            '--checkpoint-name', checkpoint_name,
            '--base-results-dir', parsed['base_results_dir'],
            '--output-suffix', 'inference_cls',
            '--gpu-config', args.gpu_config,
        ]
        if args.debug_max_cases is not None:
            new_argv.extend(['--debug-max-cases', str(args.debug_max_cases)])
        sys.argv = ['inference_cls'] + new_argv
        from uvlm.inference import inference_cls
        sys.exit(inference_cls.main())

    elif args.task == 'seg':
        # Build sys.argv for inference_seg
        # inference_seg expects: --csv-path, --model-folder, --output, --checkpoint-name, --folds
        checkpoint_name = f"checkpoint_{args.checkpoint}.pth"
        new_argv = [
            '--csv-path', args.csv_path,
            '--model-folder', args.model_dir,
            '--output', args.output_dir,
            '--checkpoint-name', checkpoint_name,
            '--folds', '0',
        ]
        if args.debug_max_cases is not None:
            new_argv.extend(['--debug-max-cases', str(args.debug_max_cases)])
        sys.argv = ['inference_seg'] + new_argv
        from uvlm.inference import inference_seg
        sys.exit(inference_seg.main())

    elif args.task == 'report':
        parsed = parse_model_path(args.model_dir)
        checkpoint_name = f"checkpoint_{args.checkpoint}.pth"

        new_argv = [
            '--csv-path', args.csv_path,
            '--dataset-name', parsed['dataset_name'] or 'Dataset001_UVLM',
            '--trainer-name', parsed['trainer_name'] or 'nnUNetTrainer_UVLM',
            '--plans-name', parsed['plans_name'] or 'UVLM_ResEncUNetLPlans',
            '--configuration-name', parsed['configuration_name'] or '3d_fullres',
            '--fold', str(parsed['fold']),
            '--checkpoint-name', checkpoint_name,
            '--base-results-dir', parsed['base_results_dir'],
            '--output-suffix', 'inference_report',
            '--gpu-config', args.gpu_config,
        ]
        if args.debug_max_cases is not None:
            new_argv.extend(['--debug-max-cases', str(args.debug_max_cases)])
        sys.argv = ['inference_reportgen'] + new_argv
        from uvlm.inference import inference_reportgen
        sys.exit(inference_reportgen.main(args=None))


if __name__ == '__main__':
    main()
