import faiss
import json
import numpy as np
from PIL import Image
from fashion_clip.fashion_clip import FashionCLIP
import argparse

def find_similar_items(query_image=None, query_text=None, top_k=10):
    """
    Searches the Faiss index for the top_k most similar items to a
    query image or text.
    """
    if not query_image and not query_text:
        print("Please provide either a query image or query text.")
        return

    # 1. Load the necessary files
    print("Loading index and model...")
    index = faiss.read_index("../Database/fashion.index")
    with open("../Database/image_paths.json", "r") as f:
        image_paths = json.load(f)
    fclip = FashionCLIP('fashion-clip')
    print("Loading complete.")

    query_embedding = None

    # 2. Encode the query (either image or text)
    if query_text:
        print(f"Encoding text query: '{query_text}'")
        query_embedding = fclip.encode_text([query_text], batch_size=1)

    elif query_image:
        print(f"Encoding image query: '{query_image}'")
        query_embedding = fclip.encode_images([query_image], batch_size=1)

    # 3. Normalize the query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding, ord=2, axis=-1, keepdims=True)

    # 4. Search the Faiss index
    print(f"Searching for top {top_k} similar items...")
    distances, indices = index.search(query_embedding.astype('float32'), top_k)

    # 5. Display results
    results = [image_paths[i] for i in indices[0]]
    print("\n--- Top 10 Recommendations ---")
    for i, path in enumerate(results):
        print(f"{i+1}: {path} (Similarity: {distances[0][i]:.4f})")

    # Optional: Display the images (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, top_k, figsize=(20, 4))
        for i, path in enumerate(results):
            img = Image.open(path)
            axes[i].imshow(img)
            axes[i].set_title(f"Rank {i+1}")
            axes[i].axis('off')
        plt.show()
    except ImportError:
        print("\nMatplotlib not found. Cannot display images. Please install it using: pip install matplotlib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find similar fashion items using FashionCLIP and Faiss.")
    parser.add_argument("--image", type=str, help="Path to the query image file.")
    parser.add_argument("--text", type=str, help="A text description of the clothing item.")
    parser.add_argument("-k", type=int, default=10, help="Number of similar items to return (default: 10).")

    args = parser.parse_args()

    find_similar_items(query_image=args.image, query_text=args.text, top_k=args.k)