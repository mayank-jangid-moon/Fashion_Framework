import os
import json
import numpy as np
import faiss
from PIL import Image
from fashion_clip.fashion_clip import FashionCLIP
from tqdm import tqdm

def build_faiss_index(image_directory="../Database/cloth/"):
    """
    Scans a directory of images, extracts features using FashionCLIP,
    and builds a Faiss index.
    """
    # 1. Initialize FashionCLIP model
    print("Loading FashionCLIP model...")
    fclip = FashionCLIP('fashion-clip')
    print("Model loaded.")

    # 2. Get all image paths
    image_paths = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_paths:
        print("No images found in the directory. Please add images to the 'images' folder.")
        return

    print(f"Found {len(image_paths)} images. Extracting features...")

    # 3. Extract image embeddings in batches
    # The notebook encodes all images at once; this is more memory-efficient for large datasets.
    image_embeddings = fclip.encode_images(image_paths, batch_size=32)
    print("Feature extraction complete.")

    # 4. Normalize embeddings for cosine similarity search
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)

    # 5. Build Faiss index
    print("Building Faiss index...")
    dimension = image_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Using IndexFlatIP for dot product (cosine similarity on normalized vectors)
    index.add(image_embeddings.astype('float32'))
    print(f"Index built successfully. Total vectors: {index.ntotal}")

    # 6. Save the index and the image path mapping
    faiss.write_index(index, "../Database/fashion.index")
    with open("../Database/image_paths.json", "w") as f:
        json.dump(image_paths, f)

    print("\nSetup complete!")
    print(" - Faiss index saved to 'fashion.index'")
    print(" - Image paths saved to 'image_paths.json'")

if __name__ == "__main__":
    build_faiss_index()