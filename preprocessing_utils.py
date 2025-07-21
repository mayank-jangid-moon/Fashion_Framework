#!/usr/bin/env python3
"""
Preprocessing Utilities Module

Contains utility functions for individual preprocessing steps.
"""

import os
import subprocess
import sys
import json
import stat
import tempfile
import shutil
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def run_densepose(input_image: str, output_path: str, densepose_script: str = None,
                 conda_env: str = "/home/mayank/miniconda3/envs/densepose") -> bool:
    """Run DensePose processing."""
    try:
        if not densepose_script:
            current_dir = Path(__file__).parent
            # Use the wrapper script instead of the direct script
            densepose_script = current_dir / "Preprocessing" / "densepose" / "densepose_wrapper.py"
        
        if not Path(densepose_script).exists():
            logger.error(f"DensePose wrapper script not found: {densepose_script}")
            return False
        
        # Use system Python for the wrapper (wrapper handles conda env internally)
        python_cmd = sys.executable
        
        # Convert paths to absolute paths
        input_image_abs = str(Path(input_image).resolve())
        output_path_abs = str(Path(output_path).resolve())
        
        cmd = [
            python_cmd, str(densepose_script),
            "--input_path", input_image_abs,
            "--output_path", output_path_abs,
            "--conda_env", conda_env
        ]
        
        logger.info(f"Running DensePose: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"DensePose failed: {result.stderr}")
            return False
        
        logger.info("DensePose completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"DensePose processing failed: {e}")
        return False

def run_openpose(input_image: str, output_json_dir: str, output_image_dir: str, 
                openpose_binary: str) -> bool:
    """Run OpenPose processing."""
    try:
        if not Path(openpose_binary).exists():
            logger.error(f"OpenPose binary not found: {openpose_binary}")
            return False
        
        # Create output directories
        Path(output_json_dir).mkdir(parents=True, exist_ok=True)
        Path(output_image_dir).mkdir(parents=True, exist_ok=True)
        
        # Create temporary directory for single image processing
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_image_dir = temp_dir_path / "input"
            temp_json_dir = temp_dir_path / "json_output"  
            temp_images_dir = temp_dir_path / "image_output"
            
            # Create temporary directories
            temp_image_dir.mkdir()
            temp_json_dir.mkdir()
            temp_images_dir.mkdir()
            
            # Copy the single input image to temp directory
            input_image_path = Path(input_image)
            temp_image_path = temp_image_dir / input_image_path.name
            shutil.copy2(input_image, temp_image_path)
            
            logger.info(f"Copied single image to temp directory: {temp_image_path}")
            
            # Use the OpenPose wrapper script
            current_dir = Path(__file__).parent
            wrapper_script = current_dir / "Preprocessing" / "openpose" / "openpose_wrapper.sh"
            
            if not wrapper_script.exists():
                logger.error(f"OpenPose wrapper script not found: {wrapper_script}")
                return False
            
            # Make sure wrapper is executable
            current_perms = wrapper_script.stat().st_mode
            wrapper_script.chmod(current_perms | stat.S_IEXEC)
            
            cmd = [
                str(wrapper_script),
                "--openpose_bin", str(openpose_binary),
                "--image_dir", str(temp_image_dir),  # Use temp directory with single image
                "--write_json", str(temp_json_dir),
                "--write_images", str(temp_images_dir),
                "--hand",
                "--disable_blending", 
                "--display", "0"
            ]
            
            logger.info(f"Running OpenPose: {' '.join(cmd)}")
            
            # Use shell=False to avoid shell interpretation issues
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"OpenPose failed: {result.stderr}")
                return False
            
            # Copy results back to the desired output directories
            # Copy JSON files
            json_files = list(temp_json_dir.glob("*.json"))
            for json_file in json_files:
                shutil.copy2(json_file, Path(output_json_dir) / json_file.name)
                logger.info(f"Copied JSON: {json_file.name}")
            
            # Copy image files
            image_files = list(temp_images_dir.glob("*"))
            for img_file in image_files:
                if img_file.is_file():  # Skip directories
                    shutil.copy2(img_file, Path(output_image_dir) / img_file.name)
                    logger.info(f"Copied image: {img_file.name}")
            
            logger.info("OpenPose completed successfully")
            return True
        
    except Exception as e:
        logger.error(f"OpenPose processing failed: {e}")
        return False

def run_image_parsing_single(input_image: str, output_path: str, 
                            pgn_script: str = None, checkpoint: str = None, 
                            conda_env: str = "/home/mayank/miniconda3/envs/image-parse") -> bool:
    """Run PGN image parsing - updated version that outputs single paletted image."""
    try:
        if not pgn_script:
            current_dir = Path(__file__).parent
            # Use the updated inf_pgn.py script
            pgn_script = current_dir / "Preprocessing" / "image_parse" / "inf_pgn.py"
        
        if not Path(pgn_script).exists():
            logger.error(f"PGN script not found: {pgn_script}")
            return False
        
        # Use conda environment Python directly
        conda_python = Path(conda_env) / "bin" / "python"
        if not conda_python.exists():
            logger.error(f"Conda Python not found: {conda_python}")
            return False
        
        # Convert paths to absolute paths
        input_image_abs = str(Path(input_image).resolve())
        output_abs = str(Path(output_path).resolve())
        
        # Get the image_parse directory to set as working directory
        image_parse_dir = Path(pgn_script).parent
        script_name = Path(pgn_script).name
        
        # Handle checkpoint path - make it relative to image_parse directory
        checkpoint_relative = None
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.is_absolute():
                # If absolute path, check if it's within image_parse directory
                try:
                    checkpoint_relative = str(checkpoint_path.relative_to(image_parse_dir))
                except ValueError:
                    # If not within image_parse dir, use the default relative path
                    checkpoint_relative = "./checkpoint/CIHP_pgn"
                    logger.warning(f"Checkpoint path {checkpoint} not within image_parse dir, using default: {checkpoint_relative}")
            else:
                # If already relative, check if it needs adjustment
                if "Preprocessing/image_parse/" in str(checkpoint_path):
                    # Remove the "Preprocessing/image_parse/" prefix
                    parts = str(checkpoint_path).split("Preprocessing/image_parse/")
                    checkpoint_relative = parts[-1]
                else:
                    checkpoint_relative = str(checkpoint_path)
        else:
            # Use default relative path
            checkpoint_relative = "./checkpoint/CIHP_pgn"
        
        cmd = [
            str(conda_python), script_name,  # Use just script name since we're in the right directory
            "-i", input_image_abs,
            "-o", output_abs,  # single output
        ]
        
        if checkpoint_relative:
            cmd.extend(["-c", checkpoint_relative])
        
        logger.info(f"Running PGN parsing from {image_parse_dir}: {' '.join(cmd)}")
        logger.info(f"Using checkpoint path: {checkpoint_relative}")
        
        # Run with image_parse directory as working directory
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(image_parse_dir))
        
        # Combine stdout and stderr for analysis
        full_output = result.stdout + result.stderr
        logger.info(f"PGN output: {full_output}")
        
        # Check for success indicators in the output
        success_indicators = [
            "Model loaded successfully",
            "Paletted image saved to:"
        ]
        
        has_success_indicators = any(indicator in full_output for indicator in success_indicators)
        
        # Check if output file was created
        output_exists = Path(output_abs).exists()
        
        if has_success_indicators and output_exists:
            logger.info("PGN parsing completed successfully - single paletted output generated")
            return True
        else:
            logger.error(f"PGN parsing failed. Output exists: {output_exists}")
            logger.error(f"Full output: {full_output}")
            return False
        
    except Exception as e:
        logger.error(f"Image parsing failed: {e}")
        return False


def run_image_parsing(input_image: str, output_vis_path: str, output_raw_path: str, 
                     pgn_script: str = None, checkpoint: str = None, 
                     conda_env: str = "/home/mayank/miniconda3/envs/image-parse") -> bool:
    """Run PGN image parsing - now outputs both visualization and raw parsing images."""
    try:
        if not pgn_script:
            current_dir = Path(__file__).parent
            # Use the inf_pgn.py script directly 
            pgn_script = current_dir / "Preprocessing" / "image_parse" / "inf_pgn.py"
        
        if not Path(pgn_script).exists():
            logger.error(f"PGN script not found: {pgn_script}")
            return False
        
        # Use conda environment Python directly
        conda_python = Path(conda_env) / "bin" / "python"
        if not conda_python.exists():
            logger.error(f"Conda Python not found: {conda_python}")
            return False
        
        # Convert paths to absolute paths
        input_image_abs = str(Path(input_image).resolve())
        output_vis_abs = str(Path(output_vis_path).resolve())
        output_raw_abs = str(Path(output_raw_path).resolve())
        
        # Get the image_parse directory to set as working directory
        image_parse_dir = Path(pgn_script).parent
        script_name = Path(pgn_script).name
        
        # Handle checkpoint path - make it relative to image_parse directory
        checkpoint_relative = None
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.is_absolute():
                # If absolute path, check if it's within image_parse directory
                try:
                    checkpoint_relative = str(checkpoint_path.relative_to(image_parse_dir))
                except ValueError:
                    # If not within image_parse dir, use the default relative path
                    checkpoint_relative = "./checkpoint/CIHP_pgn"
                    logger.warning(f"Checkpoint path {checkpoint} not within image_parse dir, using default: {checkpoint_relative}")
            else:
                # If already relative, check if it needs adjustment
                if "Preprocessing/image_parse/" in str(checkpoint_path):
                    # Remove the "Preprocessing/image_parse/" prefix
                    parts = str(checkpoint_path).split("Preprocessing/image_parse/")
                    checkpoint_relative = parts[-1]
                else:
                    checkpoint_relative = str(checkpoint_path)
        else:
            # Use default relative path
            checkpoint_relative = "./checkpoint/CIHP_pgn"
        
        cmd = [
            str(conda_python), script_name,  # Use just script name since we're in the right directory
            "-i", input_image_abs,
            "-ov", output_vis_abs,  # visualization output
            "-or", output_raw_abs,  # raw output for next step
        ]
        
        if checkpoint_relative:
            cmd.extend(["-c", checkpoint_relative])
        
        logger.info(f"Running PGN parsing from {image_parse_dir}: {' '.join(cmd)}")
        logger.info(f"Using checkpoint path: {checkpoint_relative}")
        
        # Run with image_parse directory as working directory
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(image_parse_dir))
        
        # Combine stdout and stderr for analysis
        full_output = result.stdout + result.stderr
        logger.info(f"PGN output: {full_output}")
        
        # Check for success indicators in the output
        success_indicators = [
            "Model loaded successfully",
            "Saved visualization to:",
            "Saved raw label map to:"
        ]
        
        has_success_indicators = any(indicator in full_output for indicator in success_indicators)
        
        # Check if both output files were created
        vis_exists = Path(output_vis_abs).exists()
        raw_exists = Path(output_raw_abs).exists()
        
        if has_success_indicators and vis_exists and raw_exists:
            logger.info("PGN parsing completed successfully - both visualization and raw outputs generated")
            return True
        else:
            logger.error(f"PGN parsing failed. Vis exists: {vis_exists}, Raw exists: {raw_exists}")
            logger.error(f"Full output: {full_output}")
            return False
        
    except Exception as e:
        logger.error(f"Image parsing failed: {e}")
        return False

def create_agnostic_mask(person_image_path: str, parse_mask_path: str, 
                        pose_json_path: str, output_path: str) -> bool:
    """Create agnostic person mask."""
    try:
        # Load inputs - use RGBA for person image like the original script
        person_image = Image.open(person_image_path).convert("RGBA")
        # Keep parse mask in its original format (don't convert to grayscale)
        parse_mask = Image.open(parse_mask_path)
        
        with open(pose_json_path, 'r') as f:
            pose_label = json.load(f)
            if not pose_label['people']:
                logger.error("No people detected in OpenPose JSON")
                return False
                
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]
        
        # Generate agnostic mask
        agnostic_result = get_agnostic(person_image, parse_mask, pose_data)
        
        # Handle file format based on extension
        output_path = Path(output_path)
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            # Convert RGBA to RGB with white background for JPEG
            if agnostic_result.mode == 'RGBA':
                background = Image.new('RGB', agnostic_result.size, (255, 255, 255))
                background.paste(agnostic_result, mask=agnostic_result.split()[-1])  # Use alpha channel as mask
                agnostic_result = background
        
        # Save result
        agnostic_result.save(output_path)
        logger.info(f"Agnostic mask saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Agnostic mask creation failed: {e}")
        return False

def create_agnostic_binary_mask(agnostic_image_path: str, output_path: str, 
                               target_color: tuple = (128, 128, 128)) -> bool:
    """Create binary mask from agnostic image."""
    try:
        input_image = Image.open(agnostic_image_path).convert('RGB')
        image_array = np.array(input_image)
        
        # Create binary mask for target color
        is_target = np.all(image_array == target_color, axis=-1)
        mask_array = np.zeros(is_target.shape, dtype=np.uint8)
        mask_array[is_target] = 255
        
        output_mask = Image.fromarray(mask_array, 'L')
        output_mask.save(output_path)
        return True
        
    except Exception as e:
        logger.error(f"Agnostic binary mask creation failed: {e}")
        return False

def create_agnostic_segmentation_with_script(parse_path: str, pose_json_path: str, 
                                          output_path: str, agnostic_script: str = None,
                                          conda_env: str = "/home/mayank/miniconda3/envs/image-parse") -> bool:
    """Create agnostic segmentation using the updated image_parse_agnostic.py script."""
    try:
        if not agnostic_script:
            current_dir = Path(__file__).parent
            agnostic_script = current_dir / "Preprocessing" / "image_parse" / "image_parse_agnostic.py"
        
        if not Path(agnostic_script).exists():
            logger.error(f"Agnostic script not found: {agnostic_script}")
            return False
        
        # Use conda environment Python directly
        conda_python = Path(conda_env) / "bin" / "python"
        if not conda_python.exists():
            logger.error(f"Conda Python not found: {conda_python}")
            return False
        
        # Convert paths to absolute paths
        parse_abs = str(Path(parse_path).resolve())
        pose_json_abs = str(Path(pose_json_path).resolve())
        output_path_abs = str(Path(output_path).resolve())
        
        # Check parse image dimensions to ensure proper processing
        try:
            parse_img = Image.open(parse_abs)
            w, h = parse_img.size
            logger.info(f"Parse image dimensions: {w}x{h}")
        except Exception as e:
            logger.warning(f"Could not read parse image dimensions: {e}")
        
        # Get the image_parse directory to set as working directory
        image_parse_dir = Path(agnostic_script).parent
        script_name = Path(agnostic_script).name
        
        cmd = [
            str(conda_python), script_name,  # Use just script name since we're in the right directory
            "--parse_image", parse_abs,
            "--pose_json", pose_json_abs,
            "--output_path", output_path_abs
        ]
        
        logger.info(f"Running agnostic segmentation from {image_parse_dir}: {' '.join(cmd)}")
        
        # Run with image_parse directory as working directory so it can find utils
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(image_parse_dir))
        
        # Combine stdout and stderr for analysis
        full_output = result.stdout + result.stderr
        logger.info(f"Agnostic segmentation output: {full_output}")
        
        # Check for success indicators
        success_indicators = [
            "Colorful agnostic map saved successfully"
        ]
        
        has_success_indicators = any(indicator in full_output for indicator in success_indicators)
        
        # Check if output file was created
        output_exists = Path(output_path_abs).exists()
        
        if has_success_indicators and output_exists:
            logger.info("Agnostic segmentation completed successfully")
            return True
        else:
            logger.error(f"Agnostic segmentation failed: {full_output}")
            return False
        
    except Exception as e:
        logger.error(f"Agnostic segmentation creation failed: {e}")
        return False

def create_agnostic_segmentation(parse_mask_path: str, pose_json_path: str, 
                                output_path: str) -> bool:
    """Create agnostic segmentation mask."""
    try:
        # Load inputs
        im_parse = Image.open(parse_mask_path)
        
        with open(pose_json_path, 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]
        
        # Generate agnostic segmentation
        agnostic_seg = get_im_parse_agnostic(im_parse, pose_data)
        
        # Save result
        agnostic_seg.save(output_path)
        return True
        
    except Exception as e:
        logger.error(f"Agnostic segmentation creation failed: {e}")
        return False

def get_agnostic(im, im_parse, pose_data):
    """Generate agnostic mask (from agnostic.py) - exact copy of original implementation."""
    # Convert parsing mask to a numpy array - keep original format handling
    parse_array = np.array(im_parse)
    
    # Extract head and lower body parts from the parsing mask
    # These parts will be kept in the final image
    parse_head = ((parse_array == 4).astype(np.float32) +
                  (parse_array == 13).astype(np.float32))
    parse_lower = ((parse_array == 9).astype(np.float32) +
                   (parse_array == 12).astype(np.float32) +
                   (parse_array == 16).astype(np.float32) +
                   (parse_array == 17).astype(np.float32) +
                   (parse_array == 18).astype(np.float32) +
                   (parse_array == 19).astype(np.float32))

    # Create a copy of the original image to draw on
    agnostic = im.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    # Calculate a radius for drawing based on shoulder distance
    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[12] - pose_data[9])
    point = (pose_data[9] + pose_data[12]) / 2
    pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
    r = int(length_a / 16) + 1

    # --- Mask Torso ---
    # Draw gray ellipses and lines to cover the torso area based on pose keypoints.
    for i in [9, 12]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

    # --- Mask Neck ---
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx-r*5, pointy-r*9, pointx+r*5, pointy), 'gray', 'gray')

    # --- Mask Arms ---
    # Draw lines and ellipses to cover the arms.
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*12)
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'gray', 'gray')
    for i in [3, 4, 6, 7]:
        # Check for valid keypoints before drawing
        if (pose_data[i-1, 0] == 0.0 and pose_data[i-1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

    # --- Refine Arm Masks using Parsing Data ---
    # This ensures that only the actual arm regions are masked, not the background.
    for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
        # Create a temporary black and white mask for an arm
        mask_arm = Image.new('L', (im.width, im.height), 'white')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        pointx, pointy = pose_data[pose_ids[0]]
        mask_arm_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'black', 'black')
        for i in pose_ids[1:]:
            if (pose_data[i-1, 0] == 0.0 and pose_data[i-1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r*10)
            pointx, pointy = pose_data[i]
            if i != pose_ids[-1]:
                mask_arm_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'black', 'black')
        mask_arm_draw.ellipse((pointx-r*4, pointy-r*4, pointx+r*4, pointy+r*4), 'black', 'black')

        # Combine the drawn arm mask with the semantic parsing mask
        parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
        # Paste the original image back onto the arm region
        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    # --- Paste back Head and Lower Body ---
    # Use the masks created at the beginning to restore the original head and lower body.
    agnostic.paste(im, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(im, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
    
    return agnostic

def get_im_parse_agnostic(im_parse, pose_data, w=None, h=None):
    """Generate agnostic segmentation (from image_parse_agnostic.py) - exact copy of original implementation."""
    # Auto-detect dimensions from input image
    if w is None or h is None:
        w, h = im_parse.size
        
    parse_array = np.array(im_parse)
    parse_upper = ((parse_array == 5).astype(np.float32) +  # Upper-clothes
                   (parse_array == 6).astype(np.float32) +  # Dress
                   (parse_array == 7).astype(np.float32))   # Coat
    parse_neck = (parse_array == 10).astype(np.float32)

    r = 10
    agnostic = im_parse.copy()

    # Mask arms
    for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
        mask_arm = Image.new('L', (w, h), 'black')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        i_prev = pose_ids[0]
        for i in pose_ids[1:]:
            if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or \
               (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
            pointx, pointy = pose_data[i]
            radius = r*4 if i == pose_ids[-1] else r*15
            mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
            i_prev = i
        
        parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    # Mask torso and neck
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

    return agnostic


def create_cloth_mask(cloth_image_path: str, output_mask_path: str,
                     conda_env: str = "/home/mayank/miniconda3/envs/NSTclothes") -> bool:
    """
    Create a binary mask from a cloth image using background removal.
    Uses the cloth_mask.py script in the NSTclothes conda environment.
    
    Args:
        cloth_image_path: Path to input cloth image
        output_mask_path: Path where the binary mask will be saved
        conda_env: Path to conda environment with rembg installed
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Creating cloth mask for: {cloth_image_path}")
        
        # Check if conda environment exists
        conda_python = Path(conda_env) / "bin" / "python"
        if not conda_python.exists():
            logger.error(f"Conda Python not found: {conda_python}")
            logger.error("Please ensure the NSTclothes environment is installed and contains rembg")
            return False
        
        # Get the cloth_mask.py script path
        current_dir = Path(__file__).parent
        cloth_mask_script = current_dir / "Preprocessing" / "cloth_mask.py"
        
        if not cloth_mask_script.exists():
            logger.error(f"Cloth mask script not found: {cloth_mask_script}")
            return False
        
        # Convert paths to absolute paths
        cloth_image_abs = str(Path(cloth_image_path).resolve())
        output_mask_abs = str(Path(output_mask_path).resolve())
        
        # Create a temporary script that accepts command line arguments
        # since the original cloth_mask.py has hardcoded paths
        temp_script_content = '''
from rembg import remove
from PIL import Image
import io
import sys

def create_cloth_mask(input_path, output_path):
    """Create binary mask from cloth image using rembg"""
    try:
        # Load input image
        with open(input_path, 'rb') as inp_file:
            input_data = inp_file.read()

        # Remove background
        output_data = remove(input_data)

        # Convert to PIL Image
        output_image = Image.open(io.BytesIO(output_data))

        # Convert to RGBA if not already
        output_image = output_image.convert("RGBA")

        # Create binary mask: white for cloth, black for background
        binary_mask = Image.new("RGB", output_image.size, (0, 0, 0))  # Start with black background

        # Iterate through pixels and make non-transparent areas white
        pixels = output_image.load()
        mask_pixels = binary_mask.load()

        for y in range(output_image.height):
            for x in range(output_image.width):
                # If pixel is not transparent (alpha > threshold), make it white
                if pixels[x, y][3] > 128:  # Alpha channel > 128
                    mask_pixels[x, y] = (255, 255, 255)  # White

        # Save the binary mask
        binary_mask.save(output_path)
        print(f"Cloth mask saved successfully to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating cloth mask: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python temp_cloth_mask.py <input_image> <output_mask>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    success = create_cloth_mask(input_path, output_path)
    sys.exit(0 if success else 1)
'''
        
        # Write temporary script
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(temp_script_content)
            temp_script_path = temp_file.name
        
        try:
            # Run the temporary script with the NSTclothes environment
            cmd = [
                str(conda_python),
                temp_script_path,
                cloth_image_abs,
                output_mask_abs
            ]
            
            logger.info(f"Running cloth mask generation: {' '.join(cmd[:2])} <input> <output>")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Clean up temporary script
            os.unlink(temp_script_path)
            
            if result.returncode == 0:
                # Check if output file was created
                if Path(output_mask_abs).exists():
                    logger.info("Cloth mask created successfully")
                    return True
                else:
                    logger.error("Cloth mask script succeeded but output file not found")
                    return False
            else:
                logger.error(f"Cloth mask generation failed: {result.stderr}")
                logger.error(f"stdout: {result.stdout}")
                return False
                
        except Exception as e:
            # Clean up temporary script in case of error
            if os.path.exists(temp_script_path):
                os.unlink(temp_script_path)
            raise e
        
    except Exception as e:
        logger.error(f"Failed to create cloth mask: {e}")
        return False
