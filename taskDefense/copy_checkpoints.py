#!/usr/bin/env python3
"""
Simple script to copy checkpoint files from yield_results to Temp folder.
This is a more concise version of the checkpoint copier.
"""

import os
import shutil
from pathlib import Path

def copy_checkpoints():
    """Copy checkpoint files preserving folder structure"""

    # Configuration
    source_dir = "yield_results"
    target_dir = "data"

    defense_methods = ['TRADES', 'PGD_AT', 'AMD']
    models = ['awn', 'amcnet', 'mcl', 'cldnn', 'res', 'vtcnn', 'ctdnn', 'mcd']
    datasets = ['RML2016.10a', 'RML2016.10b']

    copied = 0
    missing = 0

    print(f"Copying checkpoints from {source_dir} to {target_dir}...")

    for method in defense_methods:
        for dataset in datasets:
            for model in models:
                # Create source path
                source_path = Path(source_dir) / f"taskDefense.advTraining.{method}" / dataset / "fit" / model / "checkpoint" / f"{dataset}_{model}.best.pt"

                # Create target path
                target_path = Path(target_dir) / dataset / f"advTraining.{method}" / f"{dataset}_{model}.{method}.pt"

                if source_path.exists():
                    # Create target directory
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    # Copy file
                    shutil.copy2(source_path, target_path)
                    print(f"✓ Copied: {method}/{dataset}/{model}")
                    copied += 1
                else:
                    print(f"✗ Missing: {method}/{dataset}/{model}")
                    missing += 1

    print(f"\nSummary:")
    print(f"- Successfully copied: {copied} files")
    print(f"- Missing files: {missing} files")
    print(f"- Target directory: {Path(target_dir).absolute()}")

if __name__ == "__main__":
    copy_checkpoints()