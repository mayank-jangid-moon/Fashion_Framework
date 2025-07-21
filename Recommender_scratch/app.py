# filename: app.py

import streamlit as st
import os
import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from typing import List, Dict, Any, Optional

# --- Configuration ---

FEATURE_DICT_PATH = "combined_features_and_logos_dict.pkl"
FEATURIZER_MODEL_PATH = "best_featurizer_model.pth"
YOLO_MODEL_PATH = "best.pt"
CLOTHES_IMAGE_FOLDER = "cloth"

# --- 1. Featurizer Model Definition (Autoencoder) ---
# This class MUST exactly match the architecture used during 'train_autoencoder.py'
# The model in featurizer.ipynb is a full autoencoder, not just an encoder.
class FeaturizerModel(nn.Module):
    def __init__(self):
        super(FeaturizerModel, self).__init__()
        # Encoder: (BS, 3, 128, 128) -> (BS, 512, 1, 1)
        self.encoder = nn.Sequential(
            # Block 1: Input (3, 128, 128)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), # Output: (64, 64, 64)

            # Block 2: Input (64, 64, 64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=2, padding=1), # Output (128, 32, 32)
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=0), # Output (128, 30, 30) - IMPORTANT: padding=0 here
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), # Output: (128, 15, 15)

            # Block 3: Input (128, 15, 15)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=2, padding=1), # Output (256, 8, 8)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), # Output: (256, 4, 4) - IMPORTANT: MaxPool2d here

            # Block 4: Input (256, 4, 4)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(True), # Added for symmetry with original block
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=0), # Output (512, 2, 2)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2) # Output: (512, 1, 1) - This is the bottleneck/latent space
        )

        # Decoder: (BS, 512, 1, 1) -> (BS, 3, 128, 128)
        self.decoder = nn.Sequential(
            # Reverse MaxPool2d(2,s=2) that went from (512, 2, 2) to (512, 1, 1)
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0), # Output: (512, 2, 2)
            nn.ReLU(True),

            # Reverse Conv2d(512, 512, k=3, s=1, p=0) that went from (512, 4, 4) to (512, 2, 2)
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0), # Output: (512, 4, 4)
            nn.ReLU(True),

            # Reverse Conv2d(512, 512, k=3, s=1, p=1) that went from (512, 4, 4) to (512, 4, 4)
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            # Reverse Conv2d(256, 512, k=3, s=1, p=1) and channel change (512, 4, 4) -> (256, 4, 4)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            # Reverse MaxPool2d(2,s=2) that went from (256, 8, 8) to (256, 4, 4)
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0), # Output: (256, 8, 8)
            nn.ReLU(True),

            # Reverse Conv2d(128, 256, k=3, s=2, p=1) that went from (128, 15, 15) to (256, 8, 8)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1), # Output: (128, 15, 15)
            nn.ReLU(True),

            # Reverse MaxPool2d(2,s=2) that went from (128, 30, 30) to (128, 15, 15)
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0), # Output: (128, 30, 30)
            nn.ReLU(True),

            # Reverse Conv2d(128, 128, k=3, s=1, p=0) that went from (128, 32, 32) to (128, 30, 30)
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0), # Output: (128, 32, 32)
            nn.ReLU(True),

            # Reverse Conv2d(64, 128, k=3, s=2, p=1) that went from (64, 64, 64) to (128, 32, 32)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (64, 64, 64)
            nn.ReLU(True),

            # Reverse MaxPool2d(2,s=2) that went from (64, 128, 128) to (64, 64, 64)
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0), # Output: (64, 128, 128)
            nn.ReLU(True),

            # Reverse Conv2d(64, 64, k=3, s=1, p=1)
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            # Reverse Conv2d(3, 64, k=3, s=1, p=1) to final output
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1), # Output: (3, 128, 128)
            nn.Tanh()
        )

    def forward(self, x):
        # For a full autoencoder, forward typically returns the decoded output
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_features(self, x):
        # This method is specifically for getting the latent features (encoder output)
        return self.encoder(x)

# --- Image Transformations ---
# Define the same transformations used during model training in featurizer.ipynb
transform = transforms.Compose([
    transforms.Resize((128, 128)), # Resize images to 128x128
    transforms.ToTensor(),         # Convert PIL Image to PyTorch Tensor
    # IMPORTANT: Normalize to [-1, 1] as used in featurizer.ipynb for Tanh activation
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Helper Functions ---

@st.cache_resource # Cache models and data to avoid reloading on every rerun
def load_models_and_data():
    """Loads the featurizer model, YOLO model, and feature dictionary."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    st.write("Loading featurizer model...")
    try:
        featurizer_model = FeaturizerModel()
        # Load the state_dict into the full autoencoder model
        featurizer_model.load_state_dict(torch.load(FEATURIZER_MODEL_PATH, map_location=device))
        featurizer_model.to(device)
        featurizer_model.eval() # Set to evaluation mode
        st.success("Featurizer model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading featurizer model: {e}")
        featurizer_model = None

    st.write("Loading YOLO model...")
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(YOLO_MODEL_PATH)
        st.success("YOLO model loaded successfully.")
    except ImportError:
        st.warning("Ultralytics library not found. YOLO model will not be loaded. Logo detection will be skipped.")
        yolo_model = None
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        yolo_model = None

    st.write("Loading feature dictionary...")
    try:
        with open(FEATURE_DICT_PATH, 'rb') as f:
            combined_all_clothing_features_dict = pickle.load(f)
            st.success("Feature dictionary loaded successfully.")
            # Convert NumPy arrays in dictionary to lists for consistent handling
            for key, value in combined_all_clothing_features_dict.items():
                if 'autoencoder_feature' in value and isinstance(value['autoencoder_feature'], np.ndarray):
                    value['autoencoder_feature'] = value['autoencoder_feature'].tolist()
    except FileNotFoundError:
        st.error(f"Error: Feature dictionary not found at {FEATURE_DICT_PATH}.")
        st.info("Please ensure you have run the Gemini labeling script to create/update this file and it's in the correct directory.")
        combined_all_clothing_features_dict = {}
    except Exception as e:
        st.error(f"Error loading feature dictionary: {e}")
        combined_all_clothing_features_dict = {}

    return featurizer_model, yolo_model, combined_all_clothing_features_dict, device

# Models and data are loaded once at app startup
loaded_featurizer_model, loaded_yolo_model, combined_all_clothing_features_dict, DEVICE = load_models_and_data()

def cosine_similarity_calc(vecA, vecB):
    """Calculates cosine similarity between two vectors."""
    # Removed 'not vecA or not vecB' as vecA and vecB are already numpy arrays here
    if len(vecA) == 0 or len(vecB) == 0 or len(vecA) != len(vecB):
        return 0
    # vecA_np = np.array(vecA) # Already numpy arrays from calling functions
    # vecB_np = np.array(vecB) # Already numpy arrays from calling functions
    dot_product = np.dot(vecA, vecB) # Use vecA, vecB directly as they are already numpy arrays
    norm_A = np.linalg.norm(vecA)
    norm_B = np.linalg.norm(vecB)
    if norm_A == 0 or norm_B == 0:
        return 0
    return dot_product / (norm_A * norm_B)

def get_image_path_for_display(item_original_path):
    """Constructs the full path to the image for display using the hardcoded base folder."""
    if not CLOTHES_IMAGE_FOLDER:
        return None # Cannot display without a base folder

    # Extract just the filename from the original path stored in the PKL
    filename = os.path.basename(item_original_path)
    full_image_path = os.path.join(CLOTHES_IMAGE_FOLDER, filename)

    if os.path.exists(full_image_path):
        return full_image_path
    else:
        st.warning(f"Image not found: {full_image_path}. Using placeholder.")
        return None # Indicate that image was not found

def get_recommendations_for_item(item_data, mode, num_recommendations=10):
    """Generates recommendations based on a given item's features."""
    if not loaded_featurizer_model or not combined_all_clothing_features_dict:
        st.error("Models or data not loaded. Cannot generate recommendations.")
        return []

    # Ensure input_feature is a numpy array for consistency
    input_feature = np.array(item_data['autoencoder_feature'])
    input_brand = item_data.get('brand', 'Unknown')
    input_item_path = item_data.get('original_path', '') # Use original path to exclude itself

    scores = []
    for item_path, current_item_data in combined_all_clothing_features_dict.items():
        if item_path == input_item_path: # Exclude the input item itself
            continue

        # Ensure current_item_feature is a numpy array
        current_item_feature = np.array(current_item_data['autoencoder_feature'])
        current_item_brand = current_item_data.get('brand', 'Unknown')

        score = cosine_similarity_calc(input_feature, current_item_feature)

        # Apply brand priority if mode is 'brand' and brands match
        if mode == 'brand' and input_brand != 'Unknown' and current_item_brand == input_brand:
            score += 0.2 # Boost score for matching brand

        scores.append({"item_path": item_path, "score": score})

    top_recommendations_raw = sorted(scores, key=lambda x: x['score'], reverse=True)[:num_recommendations]

    recommended_items = []
    for entry in top_recommendations_raw:
        item_path = entry['item_path']
        item_data = combined_all_clothing_features_dict[item_path]
        item_id = os.path.basename(item_path).split('.')[0]

        # Get actual image path or fallback to placeholder
        display_image_path = get_image_path_for_display(item_path)
        image_url_to_use = display_image_path if display_image_path else f"https://placehold.co/200x250/E0F2F7/000000?text={item_id}"

        recommended_items.append({
            "id": item_id,
            "imageUrl": image_url_to_use,
            "description": item_data.get('text_labels', 'No description'),
            "brand": item_data.get('brand', 'Unknown'),
            "autoencoder_feature": item_data['autoencoder_feature'],
            "detected_logos": item_data.get('detected_logos', [])
        })
    return recommended_items

def get_recommendations_by_uploaded_image(uploaded_file, mode, num_recommendations=10):
    """Generates recommendations based on an uploaded image."""
    if not loaded_featurizer_model or not combined_all_clothing_features_dict:
        st.error("Models or data not loaded. Cannot generate recommendations.")
        return []

    try:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        processed_image = transform(image).unsqueeze(0).to(DEVICE) # Add batch dimension
        with torch.no_grad():
            # Use get_features method to obtain the latent representation
            input_feature = loaded_featurizer_model.get_features(processed_image).cpu().numpy().flatten().tolist()

        # Mock logo detection for uploaded image if YOLO model is not loaded
        detected_logos = []
        if loaded_yolo_model:
            # In a real scenario, you'd run YOLO inference here
            # results = loaded_yolo_model(image)
            # detected_logos = [box.name for box in results.pred[0] if box.name in known_brands]
            st.info("YOLO model detected, but actual inference for uploaded image is a placeholder.")

        input_brand = detected_logos[0] if detected_logos else "Unknown"

        mock_item_data = {
            'autoencoder_feature': input_feature,
            'brand': input_brand,
            'original_path': 'uploaded_image_temp' # Unique identifier to exclude
        }
        # This function calls get_recommendations_for_item, which now includes imageUrl
        return get_recommendations_for_item(mock_item_data, mode, num_recommendations)

    except Exception as e:
        st.error(f"Error processing uploaded image: {e}")
        return []

def get_recommendations_by_text_query(text_query, num_recommendations=10):
    """Generates recommendations based on a text query."""
    if not combined_all_clothing_features_dict:
        st.error("Inventory data not loaded. Cannot generate recommendations.")
        return []

    text_query = text_query.lower().strip()
    if not text_query:
        st.warning("Text query cannot be empty.")
        return []

    query_keywords = set(text_query.split())

    scores = []
    for item_path, item_data in combined_all_clothing_features_dict.items():
        # Changed 'text_label' to 'text_labels' as per user's query
        item_description = item_data.get('text_labels', '').lower()
        score = 0
        for keyword in query_keywords:
            if keyword in item_description:
                score += 1 # Simple count of matching keywords
        if score > 0:
            scores.append({"item_path": item_path, "score": score})

    top_recommendations_raw = sorted(scores, key=lambda x: x['score'], reverse=True)[:num_recommendations]

    recommended_items = []
    for entry in top_recommendations_raw:
        item_path = entry['item_path']
        item_data = combined_all_clothing_features_dict[item_path]
        item_id = os.path.basename(item_path).split('.')[0]

        # Get actual image path or fallback to placeholder
        display_image_path = get_image_path_for_display(item_path)
        image_url_to_use = display_image_path if display_image_path else f"https://placehold.co/200x250/E0F2F7/000000?text={item_id}"

        recommended_items.append({
            "id": item_id,
            "imageUrl": image_url_to_use,
            "description": item_data.get('text_labels', 'No description'),
            "brand": item_data.get('brand', 'Unknown'),
            "autoencoder_feature": item_data['autoencoder_feature'],
            "detected_logos": item_data.get('detected_logos', [])
        })
    return recommended_items


# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="Fashion Recommender")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #E0F2F7, #C3DAF9);
        color: #333;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #1A237E; /* Deep Indigo */
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4A148C; /* Darker Purple */
        color: white;
        border-radius: 25px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background-color: #6A1B9A; /* Lighter Purple on hover */
        transform: translateY(-2px);
        box-shadow: 4px 4px 10px rgba(0,0,0,0.3);
    }
    .item-card {
        border: 2px solid #9FA8DA; /* Indigo 300 */
        border-radius: 10px;
        padding: 10px;
        background-color: #E8EAF6; /* Indigo 50 */
        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease-in-out;
        cursor: pointer;
    }
    .item-card:hover {
        transform: translateY(-5px);
        box-shadow: 0px 6px 12px rgba(0,0,0,0.15);
    }
    .item-card img {
        border-radius: 8px;
    }
    .stSelectbox>div>div {
        border-radius: 8px;
        border: 1px solid #7986CB; /* Indigo 400 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("👗 Fashion Recommender System")
st.markdown("Find your perfect style match!")

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_item' not in st.session_state:
    st.session_state.selected_item = None
if 'recommendation_mode' not in st.session_state:
    st.session_state.recommendation_mode = 'visual'
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []

# --- Home Page ---
if st.session_state.page == 'home':
    st.header("Our Inventory")
    st.write("Click on any item to see recommendations!")

    # Check if the hardcoded folder exists and provide feedback
    if not os.path.isdir(CLOTHES_IMAGE_FOLDER):
        st.error(f"**Error:** The hardcoded image folder path does not exist: `{CLOTHES_IMAGE_FOLDER}`. Please update the `CLOTHES_IMAGE_FOLDER` variable in `app.py` to your actual path.")
    elif not combined_all_clothing_features_dict:
        st.warning("Inventory data not loaded. Please check the backend logs for errors.")
    else:
        inventory_items = []
        for item_path, item_data in combined_all_clothing_features_dict.items():
            item_id = os.path.basename(item_path).split('.')[0]

            # Use the hardcoded image folder path
            display_image_path = get_image_path_for_display(item_path)
            image_url_to_use = display_image_path if display_image_path else f"https://placehold.co/200x250/E0F2F7/000000?text={item_id}"

            inventory_items.append({
                "id": item_id,
                "imageUrl": image_url_to_use,
                "description": item_data.get('text_labels', 'No description'),
                "brand": item_data.get('brand', 'Unknown'),
                "autoencoder_feature": item_data['autoencoder_feature'],
                "detected_logos": item_data.get('detected_logos', []),
                "original_path": item_path # Keep original path for exclusion
            })

        # Display inventory in a grid
        cols_per_row = 6 # Adjust for responsiveness
        num_rows = (len(inventory_items) + cols_per_row - 1) // cols_per_row

        for i in range(num_rows):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < len(inventory_items):
                    item = inventory_items[idx]
                    with cols[j]:
                        # Use st.image for local files, and markdown for external URLs
                        if item['imageUrl'].startswith('http'):
                            st.markdown(
                                f"""
                                <div class="item-card" onclick="document.getElementById('item_button_{item['id']}').click();">
                                    <img src="{item['imageUrl']}" alt="{item['description']}" style="width:100%; height:150px; object-fit:cover;">
                                    <h3 style="font-size:16px; margin-top:10px; margin-bottom:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                                        {item['brand'] if item['brand'] != 'Unknown' else ''} {item['description'].split(',')[0]}
                                    </h3>
                                    <p style="font-size:12px; color:#555; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                                        {item['description']}
                                    </p>
                                    <button id="item_button_{item['id']}" style="display:none;"></button>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            # For local files, use st.image directly
                            st.markdown(f"<div class='item-card' onclick=\"document.getElementById('item_button_{item['id']}').click();\">", unsafe_allow_html=True)
                            try:
                                st.image(item['imageUrl'], caption=None, use_container_width=True)
                            except Exception:
                                st.image(f"https://placehold.co/200x250/E0F2F7/000000?text=Error+Loading", caption="Error Loading Image", use_container_width=True)
                            st.markdown(
                                f"""
                                    <h3 style="font-size:16px; margin-top:10px; margin-bottom:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                                        {item['brand'] if item['brand'] != 'Unknown' else ''} {item['description'].split(',')[0]}
                                    </h3>
                                    <p style="font-size:12px; color:#555; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                                        {item['description']}
                                    </p>
                                    <button id="item_button_{item['id']}" style="display:none;"></button>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        # Hidden button to trigger Streamlit's callback
                        if st.button(" ", key=f"select_item_{item['id']}"):
                            st.session_state.selected_item = item
                            st.session_state.page = 'detail'
                            st.rerun() # Rerun to switch page immediately

# --- Detail/Recommendation Page ---
elif st.session_state.page == 'detail' and st.session_state.selected_item:
    selected_item = st.session_state.selected_item

    if st.button("← Back to Inventory"):
        st.session_state.page = 'home'
        st.session_state.selected_item = None
        st.session_state.recommendations = []
        st.rerun()

    st.header("Selected Item")
    col1, col2 = st.columns([1, 2])
    with col1:
        # Display the selected item's image
        if selected_item['imageUrl'].startswith('http'):
            st.image(selected_item['imageUrl'], caption=selected_item['description'], width=250)
        else:
            try:
                st.image(selected_item['imageUrl'], caption=selected_item['description'], width=250)
            except Exception:
                st.image(f"https://placehold.co/200x250/E0F2F7/000000?text=Error+Loading", caption="Error Loading Image", width=250)

    with col2:
        st.subheader(f"{selected_item['brand'] if selected_item['brand'] != 'Unknown' else ''} {selected_item['description'].split(',')[0]}")
        st.write(f"**Description:** {selected_item['description']}")
        if selected_item['brand'] != 'Unknown':
            st.write(f"**Brand:** {selected_item['brand']}")
        if selected_item['detected_logos']:
            st.write(f"**Detected Logos:** {', '.join(selected_item['detected_logos'])}")

    st.markdown("---")
    st.header("Recommendations")

    # Recommendation Mode Filter
    st.session_state.recommendation_mode = st.selectbox(
        "Select Recommendation Mode:",
        ['visual', 'brand'],
        format_func=lambda x: "Visual Similarity (Default)" if x == 'visual' else "Brand Priority",
        key='detail_recommendation_mode'
    )

    # Get recommendations based on selected item and mode
    st.session_state.recommendations = get_recommendations_for_item(
        selected_item,
        st.session_state.recommendation_mode,
        num_recommendations=10
    )

    if st.session_state.recommendations:
        st.subheader("Recommended for You:")
        cols_per_row = 5 # Display 5 recommendations per row
        num_rows = (len(st.session_state.recommendations) + cols_per_row - 1) // cols_per_row

        for i in range(num_rows):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < len(st.session_state.recommendations):
                    rec_item = st.session_state.recommendations[idx]
                    with cols[j]:
                        if rec_item['imageUrl'].startswith('http'):
                            st.markdown(
                                f"""
                                <div class="item-card" onclick="document.getElementById('rec_item_button_{rec_item['id']}').click();">
                                    <img src="{rec_item['imageUrl']}" alt="{rec_item['description']}" style="width:100%; height:150px; object-fit:cover;">
                                    <h3 style="font-size:16px; margin-top:10px; margin-bottom:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                                        {rec_item['brand'] if rec_item['brand'] != 'Unknown' else ''} {rec_item['description'].split(',')[0]}
                                    </h3>
                                    <p style="font-size:12px; color:#555; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                                        {rec_item['description']}
                                    </p>
                                    <button id="rec_item_button_{rec_item['id']}" style="display:none;"></button>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(f"<div class='item-card' onclick=\"document.getElementById('rec_item_button_{rec_item['id']}').click();\">", unsafe_allow_html=True)
                            try:
                                st.image(rec_item['imageUrl'], caption=None, use_container_width=True)
                            except Exception:
                                st.image(f"https://placehold.co/200x250/E0F2F7/000000?text=Error+Loading", caption="Error Loading Image", use_container_width=True)
                            st.markdown(
                                f"""
                                    <h3 style="font-size:16px; margin-top:10px; margin-bottom:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                                        {rec_item['brand'] if rec_item['brand'] != 'Unknown' else ''} {rec_item['description'].split(',')[0]}
                                    </h3>
                                    <p style="font-size:12px; color:#555; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                                        {rec_item['description']}
                                    </p>
                                    <button id="rec_item_button_{rec_item['id']}" style="display:none;"></button>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        # Allow clicking on recommended items to view their details
                        if st.button(" ", key=f"select_rec_item_{rec_item['id']}"):
                            st.session_state.selected_item = rec_item
                            st.session_state.page = 'detail'
                            st.session_state.recommendations = [] # Clear previous recs
                            st.rerun()
    else:
        st.info("No recommendations found for this item.")

    st.markdown("---")
    st.header("Or, Get Recommendations by Uploading an Image")
    uploaded_file = st.file_uploader("Upload an image of clothing:", type=["png", "jpg", "jpeg", "bmp", "gif"])
    if uploaded_file is not None:
        rec_mode_upload = st.selectbox(
            "Select Recommendation Mode for Uploaded Image:",
            ['visual', 'brand'],
            format_func=lambda x: "Visual Similarity (Default)" if x == 'visual' else "Brand Priority",
            key='upload_recommendation_mode'
        )
        if st.button("Get Recommendations for Uploaded Image"):
            with st.spinner("Processing image and finding recommendations..."):
                st.session_state.recommendations = get_recommendations_by_uploaded_image(uploaded_file, rec_mode_upload, num_recommendations=10)
                if st.session_state.recommendations:
                    st.subheader("Recommendations based on Uploaded Image:")
                    cols_per_row = 5
                    num_rows = (len(st.session_state.recommendations) + cols_per_row - 1) // cols_per_row
                    for i in range(num_rows):
                        cols = st.columns(cols_per_row)
                        for j in range(cols_per_row):
                            idx = i * cols_per_row + j
                            if idx < len(st.session_state.recommendations):
                                rec_item = st.session_state.recommendations[idx]
                                with cols[j]:
                                    if rec_item['imageUrl'].startswith('http'):
                                        st.markdown(
                                            f"""
                                            <div class="item-card">
                                                <img src="{rec_item['imageUrl']}" alt="{rec_item['description']}" style="width:100%; height:150px; object-fit:cover;">
                                                <h3 style="font-size:16px; margin-top:10px; margin-bottom:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                                                    {rec_item['brand'] if rec_item['brand'] != 'Unknown' else ''} {rec_item['description'].split(',')[0]}
                                                </h3>
                                                <p style="font-size:12px; color:#555; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                                                    {rec_item['description']}
                                                </p>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        st.markdown(f"<div class='item-card'>", unsafe_allow_html=True)
                                        try:
                                            st.image(rec_item['imageUrl'], caption=None, use_container_width=True)
                                        except Exception:
                                            st.image(f"https://placehold.co/200x250/E0F2F7/000000?text=Error+Loading", caption="Error Loading Image", use_container_width=True)
                                        st.markdown(
                                            f"""
                                                <h3 style="font-size:16px; margin-top:10px; margin-bottom:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                                                    {rec_item['brand'] if rec_item['brand'] != 'Unknown' else ''} {rec_item['description'].split(',')[0]}
                                                </h3>
                                                <p style="font-size:12px; color:#555; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                                                    {rec_item['description']}
                                                </p>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )
                else:
                    st.info("No recommendations found for the uploaded image.")

    st.markdown("---")
    st.header("Or, Get Recommendations by Text Query")
    text_query_input = st.text_input("Enter text description (e.g., 'casual blue t-shirt with a dog print'):")
    if st.button("Get Text Recommendations"):
        with st.spinner("Processing text query and finding recommendations..."):
            st.session_state.recommendations = get_recommendations_by_text_query(text_query_input, num_recommendations=10)
            if st.session_state.recommendations:
                st.subheader("Recommendations based on Text Query:")
                cols_per_row = 5
                num_rows = (len(st.session_state.recommendations) + cols_per_row - 1) // cols_per_row
                for i in range(num_rows):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        idx = i * cols_per_row + j
                        if idx < len(st.session_state.recommendations):
                            rec_item = st.session_state.recommendations[idx]
                            with cols[j]:
                                if rec_item['imageUrl'].startswith('http'):
                                    st.markdown(
                                        f"""
                                        <div class="item-card">
                                            <img src="{rec_item['imageUrl']}" alt="{rec_item['description']}" style="width:100%; height:150px; object-fit:cover;">
                                            <h3 style="font-size:16px; margin-top:10px; margin-bottom:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                                                {rec_item['brand'] if rec_item['brand'] != 'Unknown' else ''} {rec_item['description'].split(',')[0]}
                                            </h3>
                                            <p style="font-size:12px; color:#555; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                                                {rec_item['description']}
                                            </p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(f"<div class='item-card'>", unsafe_allow_html=True)
                                    try:
                                        st.image(rec_item['imageUrl'], caption=None, use_container_width=True)
                                    except Exception:
                                        st.image(f"https://placehold.co/200x250/E0F2F7/000000?text=Error+Loading", caption="Error Loading Image", use_container_width=True)
                                    st.markdown(
                                        f"""
                                            <h3 style="font-size:16px; margin-top:10px; margin-bottom:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                                                {rec_item['brand'] if rec_item['brand'] != 'Unknown' else ''} {rec_item['description'].split(',')[0]}
                                            </h3>
                                            <p style="font-size:12px; color:#555; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                                                {rec_item['description']}
                                            </p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
            else:
                st.info("No recommendations found for the text query.")
