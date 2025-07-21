#!/bin/bash

# Fashion Recommender Streamlit App Startup Script

echo "🚀 Starting Fashion Recommender System..."

# Setup Color Analysis Environment if needed
if ! conda env list | grep -q "ColourAnalysis"; then
    echo "Setting up Color Analysis environment..."
    ./setup_color_analysis_env.sh
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if FAISS index exists
if [ ! -f "../Recommender/Database/fashion.index" ]; then
    echo "⚠️  FAISS index not found!"
    echo "Please run the build_index.py script first to create the search index."
    echo "From the Recommender/FashionCLIP directory, run:"
    echo "python build_index.py"
    exit 1
fi

echo "✅ All checks passed!"
echo "🌐 Starting Streamlit app..."
streamlit run app.py
