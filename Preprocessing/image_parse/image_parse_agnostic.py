# File: get_parse_agnostic_color.py
import json
from os import path as osp
import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import argparse

# We need the same 'decode_labels' function from the previous script's utilities
# to apply the color palette to our final map.
try:
    from utils.utils import decode_labels
except ImportError:
    print("❌ Error: Could not import 'decode_labels' from 'utils.utils'.")
    print("Please ensure the 'utils' directory from the PGN model is in the same folder as this script.")
    sys.exit(1)


def get_im_parse_agnostic(im_parse, pose_data, w=None, h=None):
    """
    Creates a cloth-agnostic parsing map by masking the arms, upper torso, and neck.
    Auto-detects dimensions from the input image if not provided.
    """
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a colorful cloth-agnostic parsing map from a single image.")
    parser.add_argument('--parse_image', type=str, required=True, help="Path to the RAW, single-channel parsing image.")
    parser.add_argument('--pose_json', type=str, required=True, help="Path to the input OpenPose JSON file.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the final COLORFUL agnostic image.")
    args = parser.parse_args()

    # The number of classes defined by the pre-trained model
    N_CLASSES = 20

    # Load the RAW (single-channel) parsing image
    try:
        im_parse_raw = Image.open(args.parse_image)
    except FileNotFoundError:
        print(f"❌ Error: Parsing image not found at '{args.parse_image}'")
        sys.exit(1)

    # Load pose data
    try:
        with open(args.pose_json, 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data).reshape((-1, 3))[:, :2]
    except (FileNotFoundError, IndexError, KeyError):
        print(f"❌ Error: Could not read a valid pose from '{args.pose_json}'.")
        sys.exit(1)
        
    # --- Process the Image ---
    # 1. Get the raw agnostic map (a black & white label map)
    agnostic_map_raw = get_im_parse_agnostic(im_parse_raw, pose_data)
    
    # 2. Convert the raw map to a NumPy array
    agnostic_np_raw = np.array(agnostic_map_raw)
    
    # 3. Reshape the array to the format expected by decode_labels: (1, H, W, 1)
    agnostic_np_reshaped = agnostic_np_raw[np.newaxis, :, :, np.newaxis]
    
    # 4. Use decode_labels to apply the color map
    color_mask = decode_labels(agnostic_np_reshaped, num_classes=N_CLASSES)
    
    # 5. Convert the resulting colorful NumPy array back to a PIL Image
    final_color_image = Image.fromarray(color_mask[0])

    # --- Save the Final Color Output ---
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    final_color_image.save(args.output_path)
    print(f"✅ Colorful agnostic map saved successfully to '{args.output_path}'")
