#!/usr/bin/env python3
"""
Color Analysis Service
Analyzes uploaded images to determine personal color palette
"""
import argparse
import json
import os
import sys
import tempfile
from collections import Counter

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch.nn as nn


def add_facer_path():
    """Add facer directory to Python path"""
    facer_path = "/home/mayank/Vault/work_space/AIMS/Summer Project/Fashion/Fashion_Shit/Colour_Analysis/facer"
    if facer_path not in sys.path:
        sys.path.insert(0, facer_path)


def get_rgb_codes(path):
    """Extract RGB codes from lip region of the image"""
    add_facer_path()
    import facer
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = facer.hwc2bchw(facer.read_hwc(path)).to(device=device)
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    
    with torch.inference_mode():
        faces = face_detector(image)

    face_parser = facer.face_parser('farl/lapa/448', device=device)
    with torch.inference_mode():
        faces = face_parser(image, faces)

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
    seg_probs = seg_probs.cpu()  # if you are using GPU

    tensor = seg_probs.permute(0, 2, 3, 1)
    tensor = tensor.squeeze().numpy()

    llip = tensor[:, :, 7]
    ulip = tensor[:, :, 9]
    lips = llip + ulip
    binary_mask = (lips >= 0.5).astype(int)

    sample = cv2.imread(path)
    img = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

    indices = np.argwhere(binary_mask)   # binary mask location extraction
    rgb_codes = img[indices[:, 0], indices[:, 1], :]  # RGB color extraction by pixels
    return rgb_codes


def filter_lip_random(rgb_codes, randomNum=40):
    """Filter and randomly sample lip RGB codes"""
    blue_condition = (rgb_codes[:, 2] <= 227)
    red_condition = (rgb_codes[:, 0] >= 97)
    filtered_rgb_codes = rgb_codes[blue_condition & red_condition]
    
    if len(filtered_rgb_codes) == 0:
        return rgb_codes[:min(randomNum, len(rgb_codes))]
    
    random_index = np.random.randint(0, filtered_rgb_codes.shape[0], 
                                   min(randomNum, filtered_rgb_codes.shape[0]))
    random_rgb_codes = filtered_rgb_codes[random_index]
    return random_rgb_codes


def calc_dis(rgb_codes):
    """Calculate distance to seasonal color palettes"""
    spring = [[253, 183, 169], [247, 98, 77], [186, 33, 33]]
    summer = [[243, 184, 202], [211, 118, 155], [147, 70, 105]]
    autumn = [[210, 124, 110], [155, 70, 60], [97, 16, 28]]
    winter = [[237, 223, 227], [177, 47, 57], [98, 14, 37]]

    res = []
    for i in range(len(rgb_codes)):
        sp = np.inf
        su = np.inf
        au = np.inf
        win = np.inf
        
        for j in range(3):
            sp = min(sp, np.linalg.norm(rgb_codes[i] - spring[j]))
            su = min(su, np.linalg.norm(rgb_codes[i] - summer[j]))
            au = min(au, np.linalg.norm(rgb_codes[i] - autumn[j]))
            win = min(win, np.linalg.norm(rgb_codes[i] - winter[j]))

        min_type = min(sp, su, au, win)
        if min_type == sp:
            ctype = "sp"
        elif min_type == su:
            ctype = "su"
        elif min_type == au:
            ctype = "au"
        elif min_type == win:
            ctype = "wi"

        res.append(ctype)
    return res


def save_skin_mask(img_path):
    """Generate and save skin mask"""
    add_facer_path()
    import facer
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = facer.hwc2bchw(facer.read_hwc(img_path)).to(device=device)
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)

    with torch.inference_mode():
        faces = face_detector(image)

    image = facer.hwc2bchw(facer.read_hwc(img_path)).to(device=device)
    face_parser = facer.face_parser('farl/lapa/448', device=device)
    with torch.inference_mode():
        faces = face_parser(image, faces)

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)
    seg_probs = seg_probs.cpu()  # if you are using GPU
    tensor = seg_probs.permute(0, 2, 3, 1)
    tensor = tensor.squeeze().numpy()

    face_skin = tensor[:, :, 1]
    binary_mask = (face_skin >= 0.5).astype(int)

    sample = cv2.imread(img_path)
    img = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    masked_image = np.zeros_like(img)
    
    try:
        masked_image[binary_mask == 1] = img[binary_mask == 1]
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite("temp.jpg", masked_image)
        return True
    except Exception as e:
        print(f"Error occurred in skin mask generation: {e}")
        return False


def get_season(img_path):
    """Get season classification from skin tone"""
    model_path = "/home/mayank/Vault/work_space/AIMS/Summer Project/Fashion/Fashion_Shit/Colour_Analysis/facer/best_model_resnet_ALL.pth"
    
    # Load model
    model = models.resnet18(pretrained=True)
    num_classes = 4
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Load saved state dictionary
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    model.eval()

    with torch.no_grad():
        output = model(image)
    pred_index = output.argmax().item()
    
    return pred_index


def analyze_color_palette(image_path):
    """Main function to analyze color palette from image"""
    try:
        # Step 1: Generate skin mask
        if not save_skin_mask(image_path):
            return {"error": "Failed to generate skin mask"}
        
        # Step 2: Run season classifier
        season_result = get_season("temp.jpg")
        
        # Step 3: Extract lip RGB codes
        rgb_codes = get_rgb_codes(image_path)
        
        if len(rgb_codes) == 0:
            return {"error": "No lip region detected in the image"}
        
        # Step 4: Filter and sample lip RGB codes
        samples = filter_lip_random(rgb_codes, randomNum=40)
        
        # Step 5: Calculate closest palette
        distances = calc_dis(samples)
        counts = Counter(distances)
        dominant = counts.most_common(1)[0][0]
        
        # Map to color recommendations
        SEASON_COLOR_MAP = {
            'sp': ["Coral", "Pink", "Yellow", "Light Green", "Light Blue", "Peach"],   # Spring
            'su': ["Light Pink", "Purple", "Blue", "Mint Green", "Grey", "Rose"],      # Summer
            'au': ["Beige", "Brown", "Olive", "Dark Green", "Dark Orange", "Mustard"], # Autumn
            'wi': ["Dark Red", "Dark Green", "Dark Blue", "Charcoal", "Plum", "Teal"]  # Winter
        }
        
        COLOR_HEX_MAP = {
            # Spring colors
            "Coral": "#FF7F7F",
            "Pink": "#FFB6C1", 
            "Yellow": "#FFFF00",
            "Light Green": "#90EE90",
            "Light Blue": "#ADD8E6",
            "Peach": "#FFCBA4",
            
            # Summer colors
            "Light Pink": "#FFB6C1",
            "Purple": "#800080",
            "Blue": "#0000FF", 
            "Mint Green": "#98FF98",
            "Grey": "#808080",
            "Rose": "#FF007F",
            
            # Autumn colors
            "Beige": "#F5F5DC",
            "Brown": "#A52A2A",
            "Olive": "#808000",
            "Dark Green": "#006400",
            "Dark Orange": "#FF8C00",
            "Mustard": "#FFDB58",
            
            # Winter colors
            "Dark Red": "#8B0000",
            "Dark Blue": "#00008B", 
            "Charcoal": "#36454F",
            "Plum": "#DDA0DD",
            "Teal": "#008080"
        }
        
        recommended_colors = SEASON_COLOR_MAP[dominant]
        color_palette = []
        
        for color in recommended_colors:
            color_palette.append({
                "name": color,
                "hex": COLOR_HEX_MAP.get(color, "#CCCCCC")
            })
        
        # Cleanup temp file
        if os.path.exists("temp.jpg"):
            os.remove("temp.jpg")
            
        return {
            "success": True,
            "season_index": season_result,
            "dominant_palette": dominant,
            "color_palette": color_palette,
            "season_name": {"sp": "Spring", "su": "Summer", "au": "Autumn", "wi": "Winter"}[dominant]
        }
        
    except Exception as e:
        # Cleanup temp file on error
        if os.path.exists("temp.jpg"):
            os.remove("temp.jpg")
        return {"error": f"Color analysis failed: {str(e)}"}


def main():
    parser = argparse.ArgumentParser(description='Color Analysis Service')
    parser.add_argument('--image', required=True, help='Path to image file')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        result = {"error": f"Image file not found: {args.image}"}
    else:
        result = analyze_color_palette(args.image)
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
