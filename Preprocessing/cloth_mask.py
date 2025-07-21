from rembg import remove
from PIL import Image
import io

# Load input image
input_path = 'cloth/00098_00.jpg'  # Replace with your image path
output_path = 'output_image.png'

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

print(f"Background removed and binary mask created. Saved as {output_path}")
