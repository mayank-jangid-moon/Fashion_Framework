import numpy as np
from PIL import Image, ImageDraw
import argparse
import json
import os

def get_agnostic(im, im_parse, pose_data):
    """
    Creates an agnostic mask for a person's image by masking the upper body clothing.

    Args:
        im (PIL.Image.Image): The input person image.
        im_parse (PIL.Image.Image): The parsing mask of the person.
        pose_data (numpy.ndarray): A numpy array of shape (n, 2) containing the 2D pose keypoints.

    Returns:
        PIL.Image.Image: The generated agnostic mask image.
    """
    # Convert parsing mask to a numpy array
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

def main():
    """
    Main function to parse arguments and generate the agnostic mask.
    """
    parser = argparse.ArgumentParser(description="Generate an agnostic person mask from an image, a parse-mask, and pose data.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input person image file.")
    parser.add_argument("--parse_path", type=str, required=True, help="Path to the input parsing mask image file.")
    parser.add_argument("--pose_path", type=str, required=True, help="Path to the input pose data JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated agnostic image.")
    
    args = parser.parse_args()

    # --- Load Input Data ---
    try:
        # Load the person image
        person_image = Image.open(args.image_path).convert("RGBA")

        # Load the parsing mask
        parse_mask = Image.open(args.parse_path)

        # Load pose data from the JSON file
        with open(args.pose_path, 'r') as f:
            pose_label = json.load(f)
            # Assumes OpenPose JSON format
            pose_data = pose_label['people'][0]['pose_keypoints_2d'] 
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]

    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e.filename}")
        return
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing input file: {e}")
        return
    except (KeyError, IndexError):
        print("Error: Pose JSON file has an unexpected format. Ensure it follows the OpenPose standard.")
        return

    # --- Generate and Save the Agnostic Mask ---
    agnostic_result = get_agnostic(person_image, parse_mask, pose_data)
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    agnostic_result.save(args.output_path)
    print(f"Agnostic mask successfully generated and saved to: {args.output_path}")


if __name__ == "__main__":
    main()

