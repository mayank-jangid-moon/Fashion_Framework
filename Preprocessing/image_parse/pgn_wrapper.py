#!/usr/bin/env python3
"""
Image Parsing (PGN) Wrapper Script

This script ensures PGN image parsing runs from the correct directory with proper paths.
It's designed to be called from the comprehensive preprocessing pipeline.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def main():
    """Main wrapper function."""
    parser = argparse.ArgumentParser(description="PGN Image Parsing wrapper with proper directory handling")
    parser.add_argument('-i', '--input_image', type=str, required=True, help='Input image path')
    parser.add_argument('-o', '--output_image', type=str, required=True, help='Output parsed image path')
    parser.add_argument('-c', '--checkpoint', type=str, default='./checkpoint/CIHP_pgn',
                       help='Path to model checkpoint')
    parser.add_argument('--conda_env', type=str, default="/home/mayank/miniconda3/envs/image-parse",
                       help='Image parsing conda environment path')
    
    args = parser.parse_args()
    
    # Get the directory of this wrapper script (should be in Preprocessing/image_parse/)
    wrapper_dir = Path(__file__).parent
    pgn_script = wrapper_dir / "inf_pgn.py"
    
    if not pgn_script.exists():
        print(f"ERROR: PGN script not found: {pgn_script}")
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
    input_path_abs = str(Path(args.input_image).resolve())
    output_path_abs = str(Path(args.output_image).resolve())
    
    # Ensure output directory exists
    output_dir = Path(output_path_abs).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle checkpoint path - it should be relative to the image_parse directory
    # If the checkpoint path contains "Preprocessing/image_parse/", remove that part
    checkpoint_path = args.checkpoint
    if "Preprocessing/image_parse/" in checkpoint_path:
        # Extract the relative part after "Preprocessing/image_parse/"
        parts = checkpoint_path.split("Preprocessing/image_parse/")
        checkpoint_path = parts[-1]  # Get the part after the split
    elif args.checkpoint.startswith('./'):
        checkpoint_path = args.checkpoint[2:]  # Remove "./"
    
    print(f"Processed checkpoint path: {checkpoint_path}")
    
    # Build command
    cmd = [
        python_cmd,
        str(pgn_script.name),  # Use just the filename since we're in the right directory
        "-i", input_path_abs,
        "-o", output_path_abs,
        "-c", checkpoint_path
    ]
    
    print(f"Running PGN Image Parsing from directory: {wrapper_dir}")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command from the image_parse directory
    try:
        result = subprocess.run(cmd, 
                              cwd=str(wrapper_dir), 
                              capture_output=True, 
                              text=True,
                              timeout=300)  # 5 minute timeout
        
        # Combine stdout and stderr for analysis
        full_output = result.stdout + result.stderr
        print("PGN Processing Output:", full_output)
        
        # Check for success indicators in the output (ignore TensorFlow warnings)
        success_indicators = [
            "Model loaded successfully",
            "Saved visualization to:",
            "Processing complete"
        ]
        
        has_success_indicators = any(indicator in full_output for indicator in success_indicators)
        
        # Check if output file was actually created
        output_exists = Path(output_path_abs).exists()
        
        if has_success_indicators and output_exists:
            print("✅ PGN Image Parsing completed successfully!")
            sys.exit(0)
        else:
            print(f"ERROR: PGN Image Parsing failed")
            if not has_success_indicators:
                print("  - No success indicators found in output")
            if not output_exists:
                print(f"  - Output file not created: {output_path_abs}")
            if result.returncode != 0:
                print(f"  - Process returned code: {result.returncode}")
            sys.exit(1)
            
    except subprocess.TimeoutExpired:
        print("ERROR: PGN Image Parsing timed out")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
