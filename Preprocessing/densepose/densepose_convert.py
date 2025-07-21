import cv2
import os
import glob
import sys
from utils.helper import GetLogger, Predictor
from argparse import ArgumentParser

# --- Argument Parsing ---
parser = ArgumentParser()
parser.add_argument(
    "--input_path", type=str, help="Path to the input image or directory", required=True
)
parser.add_argument(
    "--output_path", type=str, help="Path to the output image or directory", required=True
)
args = parser.parse_args()

# --- Setup ---
logger = GetLogger.logger(__name__)
predictor = Predictor()

# --- Input Validation and File Discovery ---
image_files = []
if os.path.isfile(args.input_path):
    # Handle single file input
    logger.info("Processing a single image file.")
    image_files.append(args.input_path)
    # Ensure the output directory for the single file exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir: # Create directory only if a path is specified
        os.makedirs(output_dir, exist_ok=True)

elif os.path.isdir(args.input_path):
    # Handle directory input (original logic)
    logger.info("Processing a directory of images.")
    os.makedirs(args.output_path, exist_ok=True)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.input_path, extension)))
        image_files.extend(glob.glob(os.path.join(args.input_path, extension.upper())))

else:
    logger.error(f"Input path is not a valid file or directory: {args.input_path}")
    sys.exit(1)

total_images = len(image_files)
logger.info(f"Found {total_images} image(s) to process")

if total_images == 0:
    logger.error(f"No images found in {args.input_path}")
    sys.exit(1)

# --- Main Processing Loop ---
for idx, image_path in enumerate(image_files):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"Could not read image: {image_path}")
        continue

    # Process the image
    out_frame, out_frame_seg = predictor.predict(image)

    # Determine the correct output path and save the processed image
    if os.path.isdir(args.input_path):
        # If input was a directory, save to the output directory with the same filename
        filename = os.path.basename(image_path)
        output_save_path = os.path.join(args.output_path, filename)
    else:
        # If input was a single file, save to the specified output file path
        output_save_path = args.output_path
        
    cv2.imwrite(output_save_path, out_frame_seg)

    # Show progress
    done = idx + 1
    percent = int((done / total_images) * 100)
    # A more detailed progress bar is only useful for multiple files
    if total_images > 1:
        sys.stdout.write(
            f"\rProgress: [{'=' * percent}{' ' * (100 - percent)}] {percent}% ({done}/{total_images})"
        )
        sys.stdout.flush()

print("\nProcessing complete!")
logger.info(f"Processed image(s) have been saved.")
