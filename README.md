# AI Based Fashion Framework

A comprehensive AI-powered fashion system that combines virtual try-on, fashion recommendation, neural style transfer, color analysis, and preprocessing capabilities. This system provides end-to-end fashion solutions from image preprocessing to personalized recommendations.

## Key Features

### Virtual Try-On (SCW-VTON)
- **Shape-Guided Clothing Warping**: State-of-the-art virtual try-on using the SCW-VTON model
- **Paired/Unpaired Modes**: Try on original clothing or different items
- **High-Quality Results**: Realistic clothing warping and person synthesis
- **VITON-HD Compatible**: Works with VITON-HD datasets

### Fashion Recommendation System
- **Text-to-Fashion Search**: Find clothing items using natural language descriptions
- **Image-to-Fashion Search**: Upload clothing images to find similar items
- **FashionCLIP Integration**: Uses state-of-the-art fashion-specific CLIP model
- **FAISS-Powered**: Lightning-fast similarity search with vector indexing
- **Multi-modal Queries**: Combine text and image searches

### Neural Style Transfer
- **Fashion Style Transfer**: Apply artistic styles to clothing images
- **Customizable Parameters**: Adjustable iterations, weights, and quality settings
- **Background Removal**: Optional transparent backgrounds using rembg
- **VGG19-Based**: Professional-quality artistic transformations
- **Real-time Processing**: Optimized for interactive use

### Personal Color Analysis
- **Seasonal Color Matching**: Determines whether you're Spring, Summer, Autumn, or Winter
- **Skin Tone Analysis**: Uses ResNet-based deep learning for accurate classification
- **Lip Color Extraction**: Analyzes facial features for personalized recommendations
- **Color Palette Generation**: Provides matching clothing color suggestions
- **Fashion Integration**: Links color analysis to clothing recommendations

### Comprehensive Preprocessing Pipeline
- **Multi-Stage Processing**: Complete fashion image preprocessing workflow
- **DensePose Integration**: Human pose and shape estimation
- **OpenPose Support**: Skeletal pose detection and JSON generation
- **Image Segmentation**: Advanced parsing using PGN (Parsing Graph Network)
- **Mask Generation**: Agnostic masks, cloth masks, and binary segmentation
- **Background Removal**: Professional cloth mask creation

### Web Interface
- **Streamlit Dashboard**: User-friendly web interface for all features
- **Multi-Environment Architecture**: Isolated conda environments for optimal performance
- **Real-time Processing**: Interactive experience with progress tracking
- **Responsive Design**: Works on desktop and mobile devices
- **Professional UI**: Clean, intuitive interface with custom styling

## System Architecture

```
Fashion AI System
├── Virtual Try-On (SCW-VTON)
│   ├── Step 1: Clothing Warping
│   ├── Step 2: Person Synthesis  
│   └── Evaluation Metrics
├── Fashion Recommender
│   ├── FashionCLIP Embeddings
│   ├── FAISS Vector Database
│   └── Multi-modal Search
├── Neural Style Transfer
│   ├── VGG19 Feature Extraction
│   ├── Style/Content Loss
│   └── Artistic Rendering
├── Color Analysis
│   ├── Face Detection & Parsing
│   ├── Seasonal Classification
│   └── Color Palette Matching
├── Preprocessing Pipeline
│   ├── DensePose Generation
│   ├── OpenPose Extraction
│   ├── Image Segmentation
│   └── Mask Creation
└── Web Interface (Streamlit)
    ├── Multi-tab Interface
    ├── File Upload/Download
    └── Real-time Processing
```

## Quick Start

### Prerequisites
- **Python 3.8+**
- **CUDA-compatible GPU** (recommended for virtual try-on)
- **Miniconda/Anaconda** for environment management
- **Git** for cloning repositories

### 1. Clone the Repository
```bash
git clone https://github.com/mayank-jangid-moon/Fashion_Shit.git
cd Fashion_Shit
```

### 2. Setup Environments

#### Core Streamlit Environment
```bash
cd Streamlit
pip install -r requirements.txt
```

#### FashionCLIP Environment (for Recommendations)
```bash
cd Streamlit
./setup_fashionclip_env.sh
```

#### Color Analysis Environment
```bash
cd Streamlit
./setup_color_analysis_env.sh
```

#### SCW-VTON Environment (for Virtual Try-On)
```bash
cd SCW-VTON
conda create -n scw-vton python=3.8
conda activate scw-vton
bash environment.sh
```

#### DensePose Environment (for Preprocessing)
```bash
cd Preprocessing/densepose
conda create -n densepose python=3.8
conda activate densepose
pip install -r requirements.txt
```

### 3. Download Pre-trained Models

#### SCW-VTON Models
Download checkpoints from [Baidu Netdisk](https://pan.baidu.com/s/1-ww-bGwZQpFe-eUN1Nq-vg?pwd=4hde) or [Google Drive](https://drive.google.com/drive/folders/1v6pYpWOQC0cHatCCECIsof_LMNymG9YI?usp=drive_link)
- Place in `SCW-VTON/ckpts/`
- Download VGG19 checkpoint and place in `SCW-VTON/models/vgg/`

#### Color Analysis Models
- ResNet model should be at: `Colour_Analysis/facer/best_model_resnet_ALL.pth`

### 4. Build Fashion Database (Optional)
```bash
cd Recommender/FashionCLIP
python build_index.py --data_dir /path/to/your/fashion/images
```

### 5. Launch the Application
```bash
cd Streamlit
streamlit run app.py
```

Visit `http://localhost:8501` in your browser!

## Detailed Usage

### Virtual Try-On
```bash
cd SCW-VTON
# Step 1: Clothing Warping
python test_for_step1.py --dataroot ./data --ckpt_dir ./ckpts --pair_mode unpaired

# Step 2: Person Synthesis  
python test_for_step2.py --dataroot ./data --ckpt_dir ./ckpts --pair_mode unpaired --plms

# Or use the combined script
bash test.sh
```

### Fashion Recommendations
```bash
cd Recommender/FashionCLIP
# Text search
python recommend.py --query "red summer dress" --top_k 10

# Image search  
python recommend.py --image_path /path/to/image.jpg --top_k 10
```

### Neural Style Transfer
```bash
cd NST_Clothes
python NeuralStyleTransfer.py \
    --content /path/to/content.jpg \
    --style /path/to/style.jpg \
    --output /path/to/output.jpg \
    --iterations 100
```

### Preprocessing Pipeline
```bash
# Full preprocessing pipeline
python preprocessing_pipeline.py \
    --input_image person.jpg \
    --cloth_image shirt.jpg \
    --output_dir ./output/

# Individual preprocessing steps
cd Preprocessing
python agnostic_mask.py --input person.jpg --output mask.png
python cloth_mask.py --input cloth.jpg --output cloth_mask.png
```

### Color Analysis
```bash
cd Colour_Analysis
python main.py --input face_photo.jpg --output color_analysis.json
```

## Configuration

### Environment Variables
```bash
# Set CUDA device (optional)
export CUDA_VISIBLE_DEVICES=0

# Set model paths (if different from defaults)
export SCWVTON_MODEL_PATH=/path/to/scw-vton/models
export FASHIONCLIP_INDEX_PATH=/path/to/faiss/index
```

### Web Interface Settings
Edit `Streamlit/.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
maxUploadSize = 200
```

## Testing

### Run Individual Tests
```bash
# Test recommender system
cd Streamlit
python test_recommender.py

# Test style transfer
python test_style_transfer.py

# Test color analysis
python test_color_analysis.py

# Test preprocessing pipeline
cd ..
python preprocessing_pipeline.py --input_image sample_person.jpg --output_dir test_output
```

### Evaluation Metrics
```bash
cd SCW-VTON
python metrics.py  # Evaluates FID, SSIM, LPIPS scores
```

## Development

### Project Structure
```
Fashion_Shit/
├── SCW-VTON/                 # Virtual try-on system
│   ├── test_for_step1.py    # Clothing warping
│   ├── test_for_step2.py    # Person synthesis
│   ├── ckpts/               # Model checkpoints
│   └── configs/             # Configuration files
├── Streamlit/               # Web interface
│   ├── app.py              # Main application
│   ├── *_service.py        # Individual services
│   └── setup_*.sh          # Environment setup scripts
├── Recommender/             # Fashion recommendation
│   └── FashionCLIP/        # CLIP-based search
├── NST_Clothes/             # Neural style transfer
├── Colour_Analysis/         # Personal color analysis
├── Preprocessing/           # Image preprocessing
│   ├── densepose/          # Human pose estimation
│   ├── openpose/           # Pose detection
│   └── image_parse/        # Image segmentation
└── Dataset/                # Training/test data
```

## 📚 Research & Citations

This project combines several state-of-the-art research works:

### SCW-VTON (Virtual Try-On)
```bibtex
@inproceedings{han2024shape,
  title={Shape-Guided Clothing Warping for Virtual Try-On},
  author={Han, Xiaoyu and Zheng, Shunyuan and Li, Zonglin and Wang, Chenyang and Sun, Xin and Meng, Quanling},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={2593--2602},
  year={2024}
}
```

### FashionCLIP (Recommendations)
- Based on OpenAI CLIP adapted for fashion domain
- Combines text and image understanding for fashion

### Neural Style Transfer
- Based on Gatys et al. "A Neural Algorithm of Artistic Style"
- Uses VGG19 for feature extraction and style transfer

## 👥 Team
1. Mayank Jangid
2. Abhinav Rajput
3. Odwitiyo



---

