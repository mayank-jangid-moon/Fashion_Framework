import numpy as np
from PIL import Image
import argparse
import os

def create_mask_from_gray_patch(input_path: str, output_path: str, target_color: tuple = (128, 128, 128)):
    """
    Creates a binary mask from an image containing a specific gray patch.

    The script identifies all pixels of a target gray color, making them white,
    and turns all other pixels black. This is useful for isolating the
    masked region created by the agnostic.py script.

    Args:
        input_path (str): The path to the input image file (e.g., the output from agnostic.py).
        output_path (str): The path where the generated binary mask will be saved.
        target_color (tuple, optional): The RGB value of the gray color to be masked. 
                                        Defaults to (128, 128, 128).
    """
    try:
        # Load the input image and ensure it's in RGB format
        input_image = Image.open(input_path).convert('RGB')
        
        # Convert the image to a NumPy array for efficient processing
        image_array = np.array(input_image)
        
        # Create a boolean mask where True corresponds to pixels matching the target gray color
        # The np.all function checks for equality across the RGB channels (axis=-1)
        is_gray = np.all(image_array == target_color, axis=-1)
        
        # Create a new array for the output image, initialized to black (0)
        # The shape is based on the height and width of the input
        mask_array = np.zeros(is_gray.shape, dtype=np.uint8)
        
        # Where the boolean mask is True, set the pixel value to white (255)
        mask_array[is_gray] = 255
        
        # Convert the final NumPy array back into a PIL Image in grayscale ('L') mode
        output_mask = Image.fromarray(mask_array, 'L')
        
        # Ensure the output directory exists before saving
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Save the resulting image
        output_mask.save(output_path)
        print(f"✅ Mask created successfully and saved to: {output_path}")

    except FileNotFoundError:
        print(f"❌ Error: The input file was not found at '{input_path}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    """
    Main function to handle command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(
        description="Generate a binary mask (white on black) from the gray upper-body patch in an image."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input image file, which contains the gray mask."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the new binary mask image."
    )
    
    args = parser.parse_args()
    
    # The 'gray' color in PIL and agnostic.py corresponds to RGB (128, 128, 128)
    gray_color_value = (128, 128, 128)
    
    create_mask_from_gray_patch(args.input_path, args.output_path, target_color=gray_color_value)

if __name__ == "__main__":
    main()