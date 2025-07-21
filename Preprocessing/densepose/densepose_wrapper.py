#!/usr/bin/env python3
"""
DensePose Wrapper Script

This script ensures DensePose runs from the correct directory with proper paths.
It's designed to be called from the comprehensive preprocessing pipeline.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def main():
    """Main wrapper function."""
    parser = argparse.ArgumentParser(description="DensePose wrapper with proper directory handling")
    parser.add_argument('--input_path', type=str, required=True, help='Input image or directory path')
    parser.add_argument('--output_path', type=str, required=True, help='Output image or directory path')
    parser.add_argument('--conda_env', type=str, default="/home/mayank/miniconda3/envs/densepose",
                       help='DensePose conda environment path')
    
    args = parser.parse_args()
    
    # Get the directory of this wrapper script (should be in Preprocessing/densepose/)
    wrapper_dir = Path(__file__).parent
    densepose_script = wrapper_dir / "densepose_convert.py"
    
    if not densepose_script.exists():
        print(f"ERROR: DensePose script not found: {densepose_script}")
        sys.exit(1)
    
    # Set up Python executable
    if args.conda_env:
        python_cmd = f"{args.conda_env}/bin/python"
    else:
        python_cmd = sys.executable
    
    if not Path(python_cmd).exists():
        print(f"ERROR: Python executable not found: {python_cmd}")
        sys.exit(1)
    
    # Convert paths to absolute paths
    input_path_abs = str(Path(args.input_path).resolve())
    output_path_abs = str(Path(args.output_path).resolve())
    
    # Ensure output directory exists
    if os.path.isfile(args.input_path):
        # Single file input - ensure output directory exists
        output_dir = Path(output_path_abs).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Directory input - ensure output directory exists
        Path(output_path_abs).mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        python_cmd,
        str(densepose_script.name),  # Use just the filename since we're in the right directory
        "--input_path", input_path_abs,
        "--output_path", output_path_abs
    ]
    
    print(f"Running DensePose from directory: {wrapper_dir}")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command from the DensePose directory
    try:
        result = subprocess.run(cmd, cwd=str(wrapper_dir), check=True)
        print("DensePose processing completed successfully!")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: DensePose processing failed with return code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
