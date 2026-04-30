#!/usr/bin/env python3
"""
U-VLM Plan Generator

Generates nnUNet-style plan JSON files from templates by replacing placeholder variables.
Makes it easy to adapt U-VLM to new datasets — just provide the template and a config.

Usage:
    # Generate a single plan
    python generate_plans.py \\
        --template uvlm/configs/plans/UVLM_ResEncUNetLPlans_chest_cls.json \\
        --output /path/to/nnUNet_preprocessed/Dataset201_MyData/UVLM_ResEncUNetLPlans_chest_cls.json \\
        --var PREPROCESSED_DIR=/path/to/nnUNet_preprocessed \\
        --var DATASET_NAME=Dataset201_MyData \\
        --var CSV_FILE=train_merged.csv

    # Generate using a config file
    python generate_plans.py \\
        --template uvlm/configs/plans/UVLM_ResEncUNetLPlans_chest_cls.json \\
        --config my_dataset_config.json \\
        --output-dir /path/to/nnUNet_preprocessed/Dataset201_MyData

Config file format (JSON):
{
    "PREPROCESSED_DIR": "/path/to/nnUNet_preprocessed",
    "DATASET_NAME": "Dataset201_MyData",
    "CSV_FILE": "train_merged.csv",
    "TOKENIZER_PATH": "/path/to/Qwen3-4B"
}

Template placeholders:
    {{PREPROCESSED_DIR}}  → base nnUNet_preprocessed directory
    {{DATASET_NAME}}      → dataset name (e.g., Dataset201_MyData)
    {{CSV_FILE}}          → CSV file name (e.g., train_merged.csv)
    {{TOKENIZER_PATH}}    → path to Qwen3-4B tokenizer (reportgen only)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


def load_template(template_path: str) -> dict:
    """Load a plan template JSON file."""
    with open(template_path, 'r') as f:
        return json.load(f)


def resolve_variables(template: dict, variables: dict) -> dict:
    """Recursively replace {{VAR}} placeholders in a dict/list/string."""

    def _replace(value):
        if isinstance(value, str):
            # Replace all {{VAR}} patterns
            for var_name, var_value in variables.items():
                value = value.replace('{{%s}}' % var_name, str(var_value))
            return value
        elif isinstance(value, dict):
            return {k: _replace(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_replace(item) for item in value]
        return value

    return _replace(template)


def collect_template_variables(template: dict) -> set:
    """Find all {{VAR}} placeholders in a template."""
    variables = set()

    def _collect(value):
        if isinstance(value, str):
            for match in re.finditer(r'\{\{(\w+)\}\}', value):
                variables.add(match.group(1))
        elif isinstance(value, dict):
            for v in value.values():
                _collect(v)
        elif isinstance(value, list):
            for item in value:
                _collect(item)

    _collect(template)
    return variables


def main():
    parser = argparse.ArgumentParser(
        description='U-VLM Plan Generator - generates plan JSONs from templates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick: specify variables on command line
  python generate_plans.py --template uvlm/configs/plans/UVLM_ResEncUNetLPlans_chest_cls.json \\
      --output ./my_plan.json \\
      --var PREPROCESSED_DIR=/data/nnUNet_preprocessed \\
      --var DATASET_NAME=Dataset201_MyData \\
      --var CSV_FILE=train_merged.csv

  # Using a config file:
  python generate_plans.py --template uvlm/configs/plans/UVLM_ResEncUNetLPlans_chest_cls.json \\
      --config my_config.json --output ./my_plan.json

  # Show required variables for a template:
  python generate_plans.py --template uvlm/configs/plans/UVLM_ResEncUNetLPlans_chest_report.json --show-vars
"""
    )

    parser.add_argument('--template', required=True, help='Path to plan template JSON')
    parser.add_argument('--output', help='Output file path for the generated plan')
    parser.add_argument('--output-dir', help='Output directory (uses template filename)')
    parser.add_argument('--config', help='JSON config file with variable values')
    parser.add_argument('--var', action='append', default=[],
                        help='Variables in KEY=VALUE format (can be used multiple times)')
    parser.add_argument('--show-vars', action='store_true',
                        help='Show required variables for the template and exit')

    args = parser.parse_args()

    if not os.path.exists(args.template):
        print(f"ERROR: Template not found: {args.template}")
        sys.exit(1)

    template = load_template(args.template)

    # Show variables mode
    if args.show_vars:
        vars_needed = collect_template_variables(template)
        print(f"Template: {args.template}")
        print(f"Required variables:")
        for v in sorted(vars_needed):
            print(f"  --var {v}=<value>")
        return

    # Collect variables
    variables = {}

    # From config file
    if args.config:
        if not os.path.exists(args.config):
            print(f"ERROR: Config file not found: {args.config}")
            sys.exit(1)
        with open(args.config, 'r') as f:
            variables.update(json.load(f))

    # From command line (overrides config)
    for var_str in args.var:
        if '=' not in var_str:
            print(f"ERROR: Invalid --var format: {var_str}. Use KEY=VALUE.")
            sys.exit(1)
        key, value = var_str.split('=', 1)
        variables[key] = value

    # Resolve
    result = resolve_variables(template, variables)

    # Determine output path
    if args.output:
        output_path = args.output
    elif args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, os.path.basename(args.template))
    else:
        print("ERROR: --output or --output-dir required")
        sys.exit(1)

    # Check for unresolved variables
    remaining = collect_template_variables(result)
    if remaining:
        print(f"WARNING: Unresolved variables remain: {remaining}")
        print(f"  Provide them with --var NAME=value")

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Plan generated: {output_path}")
    if remaining:
        print(f"Unresolved: {sorted(remaining)}")


if __name__ == '__main__':
    main()
