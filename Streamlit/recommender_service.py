#!/usr/bin/env python3
"""
Recommender Service - Runs in FashionCLIP conda environment
This script handles the actual recommendation logic using FashionCLIP and FAISS
"""

import sys
import json
import argparse
import os
import tempfile
import numpy as np
import faiss
from PIL import Image

try:
    from fashion_clip.fashion_clip import FashionCLIP
except ImportError:
    print(json.dumps({"error": "FashionCLIP not found. Please install it in the FashionCLIP conda environment."}))
    sys.exit(1)

def load_model_and_index():
    """Load the FashionCLIP model and FAISS index"""
    try:
        print("Debug: Starting to load model and index", file=sys.stderr)
        
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Debug: Script directory: {script_dir}", file=sys.stderr)
        
        # Load FAISS index
        index_path = os.path.join(script_dir, '..', 'Recommender', 'Database', 'fashion.index')
        paths_json = os.path.join(script_dir, '..', 'Recommender', 'Database', 'image_paths.json')
        
        print(f"Debug: Index path: {index_path}", file=sys.stderr)
        print(f"Debug: Paths JSON: {paths_json}", file=sys.stderr)
        
        if not os.path.exists(index_path) or not os.path.exists(paths_json):
            error_msg = f"Index files not found. Index: {os.path.exists(index_path)}, Paths: {os.path.exists(paths_json)}"
            print(f"Debug: {error_msg}", file=sys.stderr)
            return None, None, None, error_msg
        
        print("Debug: Loading FAISS index", file=sys.stderr)    
        index = faiss.read_index(index_path)
        print(f"Debug: FAISS index loaded, size: {index.ntotal}", file=sys.stderr)
        
        with open(paths_json, "r") as f:
            image_paths = json.load(f)
        print(f"Debug: Loaded {len(image_paths)} image paths", file=sys.stderr)
        
        # Convert relative paths to absolute paths
        base_dir = os.path.join(script_dir, '..')
        absolute_paths = []
        missing_count = 0
        for path in image_paths:
            if path.startswith('../'):
                # Convert relative path to absolute
                abs_path = os.path.abspath(os.path.join(base_dir, path))
            else:
                # Already absolute or need to be made absolute
                abs_path = os.path.abspath(path)
            
            # Verify the file exists
            if os.path.exists(abs_path):
                absolute_paths.append(abs_path)
            else:
                # Try alternative path resolution
                filename = os.path.basename(abs_path)
                cloth_dir = os.path.join(script_dir, '..', 'Recommender', 'Database', 'cloth')
                alternative_path = os.path.join(cloth_dir, filename)
                if os.path.exists(alternative_path):
                    absolute_paths.append(alternative_path)
                else:
                    # Skip missing files but log them
                    missing_count += 1
                    absolute_paths.append(abs_path)  # Keep the path for index consistency
        
        print(f"Debug: Path resolution complete. Valid: {len(absolute_paths)}, Missing: {missing_count}", file=sys.stderr)
        
        # Load FashionCLIP model
        print("Debug: Loading FashionCLIP model", file=sys.stderr)
        fclip = FashionCLIP('fashion-clip')
        print("Debug: FashionCLIP model loaded successfully", file=sys.stderr)
        
        return index, absolute_paths, fclip, None
    except Exception as e:
        error_msg = f"Error loading model and index: {str(e)}"
        print(f"Debug: {error_msg}", file=sys.stderr)
        return None, None, None, error_msg

def find_similar_items(query_image=None, query_text=None, top_k=10):
    """Find similar items using text or image query"""
    index, image_paths, fclip, error = load_model_and_index()
    
    if error:
        return {"error": error}
    
    query_embedding = None
    
    try:
        # Encode the query
        if query_text:
            query_embedding = fclip.encode_text([query_text], batch_size=1)
        elif query_image:
            # Load image if path is provided
            if isinstance(query_image, str):
                query_embedding = fclip.encode_images([query_image], batch_size=1)
            else:
                return {"error": "Invalid image input"}
        
        if query_embedding is None:
            return {"error": "No valid query provided"}
        
        # Normalize the query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding, ord=2, axis=-1, keepdims=True)
        
        # Search the FAISS index
        distances, indices = index.search(query_embedding.astype('float32'), top_k)
        
        # Get results with similarity scores
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(image_paths):
                results.append({
                    'path': image_paths[idx],
                    'similarity': float(distances[0][i])
                })
        
        return {"results": results}
        
    except Exception as e:
        return {"error": f"Error during search: {str(e)}"}

def main():
    parser = argparse.ArgumentParser(description="Fashion Recommender Service")
    parser.add_argument("--text", type=str, help="Text query")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--top_k", type=int, default=10, help="Number of recommendations")
    
    # If called with JSON string as first argument (for backward compatibility)
    if len(sys.argv) == 2 and not sys.argv[1].startswith('-'):
        try:
            input_data = json.loads(sys.argv[1])
            query_text = input_data.get('query')
            top_k = input_data.get('top_k', 10)
            
            result = find_similar_items(query_text=query_text, top_k=top_k)
            print(json.dumps(result))
            return
        except json.JSONDecodeError:
            pass
    
    args = parser.parse_args()
    
    if not args.text and not args.image:
        print(json.dumps({"error": "Please provide either --text or --image parameter"}))
        sys.exit(1)
    
    # Add debug info
    print(f"Debug: Starting search with text='{args.text}', image='{args.image}', top_k={args.top_k}", file=sys.stderr)
    
    # Find similar items
    result = find_similar_items(
        query_text=args.text,
        query_image=args.image,
        top_k=args.top_k
    )
    
    # Output as JSON
    print(json.dumps(result))

if __name__ == "__main__":
    main()
