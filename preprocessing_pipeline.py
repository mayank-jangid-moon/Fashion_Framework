#!/usr/bin/env python3
"""
Comprehensive Fashion Preprocessing Pipeline

This script processes fashion images through multiple preprocessing steps:
1. Copy input human image to output directory
2. Copy input cloth image to output directory (if provided)
3. Generate cloth mask using background removal in NSTclothes environment (if cloth image provided)
4. DensePose generation for human image
5. OpenPose skeleton and JSON generation for human image
6. Image parsing/segmentation (using PGN) - creates both visualization and raw outputs for human image
7. Agnostic mask generation for human image
8. Agnostic segmentation mask generation for human image

Usage:
    # Basic usage with just human image
    python preprocessing_pipeline.py --input_image person.jpg --output_dir ./output/
    
    # With cloth image for mask generation
    python preprocessing_pipeline.py --input_image person.jpg --cloth_image shirt.jpg --output_dir ./output/

Note: The image_parse_agnostic.py script requires access to the utils directory in the image_parse folder,
so it runs with image_parse as the working directory. The cloth mask generation runs in the NSTclothes 
conda environment which contains the rembg library.
"""

import os
import sys
import argparse
import subprocess
import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import shutil
from typing import Optional, Tuple, List
import logging

# Add preprocessing directories to path for imports
current_dir = Path(__file__).parent
preprocessing_dir = current_dir / "Preprocessing"
sys.path.insert(0, str(preprocessing_dir / "densepose"))
sys.path.insert(0, str(preprocessing_dir / "image_parse"))
sys.path.insert(0, str(preprocessing_dir))

# Import our utility functions
from preprocessing_utils import (
    run_densepose, run_openpose, run_image_parsing, run_image_parsing_single,
    create_agnostic_mask, create_agnostic_binary_mask, 
    create_agnostic_segmentation, create_agnostic_segmentation_with_script,
    create_cloth_mask
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FashionPreprocessor:
    """Main class for comprehensive fashion preprocessing pipeline."""
    
    def __init__(self, input_image: str, output_dir: str, 
                 cloth_image: Optional[str] = None,
                 openpose_path: Optional[str] = None,
                 pgn_checkpoint: Optional[str] = None,
                 temp_dir: Optional[str] = None,
                 densepose_env: str = "/home/mayank/miniconda3/envs/densepose",
                 image_parse_env: str = "/home/mayank/miniconda3/envs/image-parse",
                 cloth_mask_env: str = "/home/mayank/miniconda3/envs/NSTclothes"):
        """
        Initialize the fashion preprocessor.
        
        Args:
            input_image: Path to input human image
            output_dir: Output directory for all processed files
            cloth_image: Path to input cloth image (optional, for cloth mask generation)
            openpose_path: Path to OpenPose binary (auto-detected if None)
            pgn_checkpoint: Path to PGN model checkpoint
            temp_dir: Temporary directory for intermediate files
            densepose_env: Path to DensePose conda environment
            image_parse_env: Path to image parsing conda environment
            cloth_mask_env: Path to cloth mask conda environment (with rembg)
        """
        self.input_image = Path(input_image)
        self.cloth_image = Path(cloth_image) if cloth_image else None
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir) if temp_dir else self.output_dir / "temp"
        self.densepose_env = densepose_env
        self.image_parse_env = image_parse_env
        self.cloth_mask_env = cloth_mask_env
        
        # Validate input
        if not self.input_image.exists():
            raise FileNotFoundError(f"Input human image not found: {input_image}")
        
        if self.cloth_image and not self.cloth_image.exists():
            raise FileNotFoundError(f"Input cloth image not found: {cloth_image}")
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up paths
        self.base_name = self.input_image.stem
        self.setup_paths()
        
        # Auto-detect tool paths
        self.openpose_path = self._find_openpose_binary(openpose_path)
        self.pgn_checkpoint = self._find_pgn_checkpoint(pgn_checkpoint)
        
        # Validate conda environments
        self._validate_conda_environments()
        
        logger.info(f"Initialized preprocessor for human image: {self.input_image}")
        if self.cloth_image:
            logger.info(f"Cloth image for mask generation: {self.cloth_image}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"DensePose environment: {self.densepose_env}")
        logger.info(f"Image parsing environment: {self.image_parse_env}")
        logger.info(f"Cloth mask environment: {self.cloth_mask_env}")
    
    def setup_paths(self):
        """Setup all output file paths."""
        self.paths = {
            # Original image copy - renamed to end with _image
            'image': self.output_dir / f"{self.base_name}_image.jpg",
            
            # DensePose output
            'densepose': self.output_dir / f"{self.base_name}_densepose.jpg",
            
            # OpenPose outputs
            'openpose_image': self.output_dir / f"{self.base_name}_openpose.jpg",
            'openpose_json': self.output_dir / f"{self.base_name}_keypoints.json",
            
            # Image parsing output (single paletted file for all purposes)
            'image_parse': self.output_dir / f"{self.base_name}_parse.png",  # single paletted image
            
            # Agnostic outputs
            'agnostic': self.output_dir / f"{self.base_name}_agnostic.png",  # Changed to PNG to support RGBA
            'agnostic_mask': self.output_dir / f"{self.base_name}_agnostic_mask.png",
            'agnostic_segmentation': self.output_dir / f"{self.base_name}_agnostic_segmentation.png",
            
            # Temporary directories
            'temp_openpose_images': self.temp_dir / "openpose_images",
            'temp_openpose_json': self.temp_dir / "openpose_json",
        }
        
        # Add cloth-related paths if cloth image is provided
        if self.cloth_image:
            cloth_base_name = self.cloth_image.stem
            self.paths['cloth_image'] = self.output_dir / f"{cloth_base_name}_cloth.jpg"
            self.paths['cloth_mask'] = self.output_dir / f"{cloth_base_name}_cloth_mask.png"
        
        # Create temp directories
        self.paths['temp_openpose_images'].mkdir(parents=True, exist_ok=True)
        self.paths['temp_openpose_json'].mkdir(parents=True, exist_ok=True)
    
    def _find_openpose_binary(self, openpose_path: Optional[str]) -> str:
        """Find OpenPose binary path."""
        if openpose_path and Path(openpose_path).exists():
            return openpose_path
        
        # Search in common locations
        search_paths = [
            current_dir / "Preprocessing" / "openpose" / "build" / "examples" / "openpose" / "openpose.bin",
            Path("./Preprocessing/openpose/build/examples/openpose/openpose.bin"),
            Path("./openpose/build/examples/openpose/openpose.bin"),
        ]
        
        for path in search_paths:
            if path.exists():
                logger.info(f"Found OpenPose binary at: {path}")
                return str(path)
        
        raise FileNotFoundError("OpenPose binary not found. Please specify --openpose_path")
    
    def _find_pgn_checkpoint(self, pgn_checkpoint: Optional[str]) -> str:
        """Find PGN model checkpoint path."""
        if pgn_checkpoint and Path(pgn_checkpoint).exists():
            return pgn_checkpoint
        
        # Search in common locations
        search_paths = [
            current_dir / "Preprocessing" / "image_parse" / "checkpoint" / "CIHP_pgn",
            Path("./Preprocessing/image_parse/checkpoint/CIHP_pgn"),
        ]
        
        for path in search_paths:
            if path.exists():
                logger.info(f"Found PGN checkpoint at: {path}")
                # Return the relative path from the image_parse directory perspective
                return "./checkpoint/CIHP_pgn"  # This is what inf_pgn.py expects by default
        
        logger.warning("PGN checkpoint not found. Using default relative path.")
        return "./checkpoint/CIHP_pgn"  # Default path relative to image_parse directory
    
    def _validate_conda_environments(self):
        """Validate that conda environments exist and have Python."""
        # Check DensePose environment
        densepose_python = Path(self.densepose_env) / "bin" / "python"
        if not densepose_python.exists():
            logger.warning(f"DensePose Python not found: {densepose_python}")
            logger.warning("DensePose processing may fail")
        else:
            logger.info(f"✓ DensePose Python found: {densepose_python}")
        
        # Check image parsing environment
        image_parse_python = Path(self.image_parse_env) / "bin" / "python"
        if not image_parse_python.exists():
            logger.warning(f"Image parsing Python not found: {image_parse_python}")
            logger.warning("Image parsing may fail")
        else:
            logger.info(f"✓ Image parsing Python found: {image_parse_python}")
        
        # Check cloth mask environment (only if cloth image is provided)
        if self.cloth_image:
            cloth_mask_python = Path(self.cloth_mask_env) / "bin" / "python"
            if not cloth_mask_python.exists():
                logger.warning(f"Cloth mask Python not found: {cloth_mask_python}")
                logger.warning("Cloth mask processing may fail")
            else:
                logger.info(f"✓ Cloth mask Python found: {cloth_mask_python}")
    
    def copy_input_image(self):
        """Copy input image to output directory."""
        logger.info("Copying input human image...")
        shutil.copy2(self.input_image, self.paths['image'])
        return self.paths['image']
    
    def copy_cloth_image(self):
        """Copy cloth image to output directory."""
        if not self.cloth_image:
            return None
        logger.info("Copying input cloth image...")
        shutil.copy2(self.cloth_image, self.paths['cloth_image'])
        return self.paths['cloth_image']
    
    def generate_cloth_mask(self) -> bool:
        """Generate cloth mask from cloth image."""
        if not self.cloth_image:
            return False
        logger.info("Generating cloth mask...")
        return create_cloth_mask(
            str(self.cloth_image),
            str(self.paths['cloth_mask']),
            conda_env=self.cloth_mask_env
        )
    
    def generate_densepose(self) -> bool:
        """Generate DensePose visualization."""
        logger.info("Generating DensePose...")
        return run_densepose(
            str(self.input_image), 
            str(self.paths['densepose']),
            conda_env=self.densepose_env
        )
    
    def generate_openpose(self) -> bool:
        """Generate OpenPose skeleton and JSON."""
        logger.info("Generating OpenPose...")
        
        # Run OpenPose
        success = run_openpose(
            str(self.input_image),
            str(self.paths['temp_openpose_json']),
            str(self.paths['temp_openpose_images']),
            self.openpose_path
        )
        
        if not success:
            return False
        
        # Move outputs to final locations
        try:
            json_files = list(self.paths['temp_openpose_json'].glob("*.json"))
            image_files = list(self.paths['temp_openpose_images'].glob("*.png"))
            
            if json_files:
                shutil.move(str(json_files[0]), str(self.paths['openpose_json']))
                logger.info(f"OpenPose JSON saved to: {self.paths['openpose_json']}")
            
            if image_files:
                shutil.move(str(image_files[0]), str(self.paths['openpose_image']))
                logger.info(f"OpenPose image saved to: {self.paths['openpose_image']}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to move OpenPose outputs: {e}")
            return False
    
    def generate_image_parse(self) -> bool:
        """Generate image parsing/segmentation using PGN - now creates single paletted output."""
        logger.info("Generating image parsing...")
        return run_image_parsing_single(
            str(self.input_image),
            str(self.paths['image_parse']),  # single output
            checkpoint=self.pgn_checkpoint,
            conda_env=self.image_parse_env
        )
    
    def generate_agnostic_mask(self) -> bool:
        """Generate agnostic person mask."""
        logger.info("Generating agnostic mask...")
        return create_agnostic_mask(
            str(self.input_image),
            str(self.paths['image_parse']),  # use single parsing image
            str(self.paths['openpose_json']),
            str(self.paths['agnostic'])
        )
    
    def generate_agnostic_binary_mask(self) -> bool:
        """Generate binary mask from agnostic image."""
        logger.info("Generating agnostic binary mask...")
        return create_agnostic_binary_mask(
            str(self.paths['agnostic']),
            str(self.paths['agnostic_mask'])
        )
    
    def generate_agnostic_segmentation(self) -> bool:
        """Generate agnostic segmentation mask using the updated script."""
        logger.info("Generating agnostic segmentation...")
        return create_agnostic_segmentation_with_script(
            str(self.paths['image_parse']),  # use single parsing image
            str(self.paths['openpose_json']),
            str(self.paths['agnostic_segmentation']),
            conda_env=self.image_parse_env
        )
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temporary files")
    
    def process_all(self, skip_on_failure: bool = True) -> dict:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            skip_on_failure: If True, continue processing even if some steps fail
            
        Returns:
            dict: Status of each processing step
        """
        logger.info("Starting comprehensive preprocessing pipeline...")
        
        results = {}
        
        # Step 1: Copy input images
        try:
            self.copy_input_image()
            results['copy_image'] = True
        except Exception as e:
            logger.error(f"Failed to copy input human image: {e}")
            results['copy_image'] = False
            if not skip_on_failure:
                return results
        
        # Step 1.1: Copy cloth image and generate cloth mask (if provided)
        if self.cloth_image:
            try:
                self.copy_cloth_image()
                results['copy_cloth_image'] = True
            except Exception as e:
                logger.error(f"Failed to copy cloth image: {e}")
                results['copy_cloth_image'] = False
                if not skip_on_failure:
                    return results
            
            try:
                results['cloth_mask'] = self.generate_cloth_mask()
                if not results['cloth_mask'] and not skip_on_failure:
                    return results
            except Exception as e:
                logger.error(f"Failed to generate cloth mask: {e}")
                results['cloth_mask'] = False
                if not skip_on_failure:
                    return results
        else:
            results['copy_cloth_image'] = False
            results['cloth_mask'] = False
        
        # Step 2: Generate DensePose
        results['densepose'] = self.generate_densepose()
        if not results['densepose'] and not skip_on_failure:
            return results
        
        # Step 3: Generate OpenPose
        results['openpose'] = self.generate_openpose()
        if not results['openpose'] and not skip_on_failure:
            return results
        
        # Step 4: Generate image parsing
        results['image_parse'] = self.generate_image_parse()
        if not results['image_parse'] and not skip_on_failure:
            return results
        
        # Step 5: Generate agnostic mask (requires OpenPose and image parsing)
        if results['openpose'] and results['image_parse']:
            results['agnostic_mask'] = self.generate_agnostic_mask()
        else:
            results['agnostic_mask'] = False
            logger.warning("Skipping agnostic mask generation (missing dependencies)")
        
        # Step 6: Generate agnostic binary mask (requires agnostic mask)
        if results['agnostic_mask']:
            results['agnostic_binary_mask'] = self.generate_agnostic_binary_mask()
        else:
            results['agnostic_binary_mask'] = False
            logger.warning("Skipping agnostic binary mask generation (missing dependencies)")
        
        # Step 7: Generate agnostic segmentation (requires OpenPose and image parsing)
        if results['openpose'] and results['image_parse']:
            results['agnostic_segmentation'] = self.generate_agnostic_segmentation()
        else:
            results['agnostic_segmentation'] = False
            logger.warning("Skipping agnostic segmentation generation (missing dependencies)")
        
        # Cleanup
        try:
            self.cleanup_temp_files()
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")
        
        logger.info("Preprocessing pipeline completed!")
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: dict):
        """Print a summary of processing results."""
        logger.info("\n" + "="*50)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("="*50)
        
        for step, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            logger.info(f"{step.upper():<25} {status}")
        
        logger.info("="*50)
        
        # List output files with descriptions
        logger.info("OUTPUT FILES:")
        file_descriptions = {
            'image': 'Original input human image copy (renamed with _image suffix)',
            'cloth_image': 'Original input cloth image copy (renamed with _cloth suffix)',
            'densepose': 'DensePose visualization',
            'openpose_image': 'OpenPose skeleton visualization',
            'openpose_json': 'OpenPose keypoints JSON',
            'image_parse': 'Image parsing/segmentation (single paletted PNG)',
            'agnostic': 'Agnostic person mask',
            'agnostic_mask': 'Agnostic binary mask',
            'agnostic_segmentation': 'Final agnostic segmentation (colorful)',
            'cloth_mask': 'Cloth binary mask (white=cloth, black=background) with _cloth_mask suffix'
        }
        
        for name, path in self.paths.items():
            if not name.startswith('temp_') and path.exists():
                description = file_descriptions.get(name, 'Generated file')
                logger.info(f"  {name}: {path}")
                logger.info(f"    → {description}")
        
        logger.info("="*50)


def main():
    """Main function to run the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Fashion Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single person image
    python preprocessing_pipeline.py --input_image person.jpg --output_dir ./output/
    
    # Process person image with cloth mask generation
    python preprocessing_pipeline.py --input_image person.jpg --cloth_image shirt.jpg --output_dir ./output/
    
    # Process with custom paths
    python preprocessing_pipeline.py \\
        --input_image person.jpg \\
        --cloth_image shirt.jpg \\
        --output_dir ./output/ \\
        --openpose_path ./openpose/build/examples/openpose/openpose.bin \\
        --pgn_checkpoint ./image_parse/checkpoint/CIHP_pgn \\
        --temp_dir ./temp/
        """
    )
    
    parser.add_argument(
        '--input_image', '-i',
        type=str, required=True,
        help='Path to input person image'
    )
    
    parser.add_argument(
        '--cloth_image', '-c',
        type=str, default=None,
        help='Path to input cloth image (for cloth mask generation)'
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        type=str, required=True,
        help='Output directory for all processed files'
    )
    
    parser.add_argument(
        '--openpose_path',
        type=str, default=None,
        help='Path to OpenPose binary (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--pgn_checkpoint',
        type=str, default=None,
        help='Path to PGN model checkpoint (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--temp_dir',
        type=str, default=None,
        help='Temporary directory for intermediate files (default: output_dir/temp)'
    )
    
    parser.add_argument(
        '--densepose_env',
        type=str, default="/home/mayank/miniconda3/envs/densepose",
        help='Path to DensePose conda environment (default: /home/mayank/miniconda3/envs/densepose)'
    )
    
    parser.add_argument(
        '--image_parse_env',
        type=str, default="/home/mayank/miniconda3/envs/image-parse",
        help='Path to image parsing conda environment (default: /home/mayank/miniconda3/envs/image-parse)'
    )
    
    parser.add_argument(
        '--cloth_mask_env',
        type=str, default="/home/mayank/miniconda3/envs/NSTclothes",
        help='Path to cloth mask conda environment with rembg (default: /home/mayank/miniconda3/envs/NSTclothes)'
    )
    
    parser.add_argument(
        '--continue_on_failure',
        action='store_true',
        help='Continue processing even if some steps fail'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize preprocessor
        preprocessor = FashionPreprocessor(
            input_image=args.input_image,
            cloth_image=args.cloth_image,
            output_dir=args.output_dir,
            openpose_path=args.openpose_path,
            pgn_checkpoint=args.pgn_checkpoint,
            temp_dir=args.temp_dir,
            densepose_env=args.densepose_env,
            image_parse_env=args.image_parse_env,
            cloth_mask_env=args.cloth_mask_env
        )
        
        # Run preprocessing pipeline
        results = preprocessor.process_all(skip_on_failure=args.continue_on_failure)
        
        # Exit with appropriate code
        if all(results.values()):
            logger.info("All preprocessing steps completed successfully!")
            sys.exit(0)
        else:
            logger.warning("Some preprocessing steps failed. Check the logs above.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
