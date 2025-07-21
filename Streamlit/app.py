import streamlit as st
import json
import subprocess
from PIL import Image
import os
import tempfile
from io import BytesIO
import warnings
import time

# Suppress Streamlit deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set page config
st.set_page_config(
    page_title="Fashion Recommender System",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for light theme and better styling
st.markdown("""
<style>
    /* Light theme overrides */
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Hide Streamlit style elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Search section styling */
    .search-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 1px solid #e9ecef;
    }
    
    /* Results grid */
    .results-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin-top: 2rem;
    }
    
    /* Image container */
    .image-container {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        overflow: hidden;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
        .image-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Color palette styling */
    .color-palette-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    .color-swatch {
        display: inline-block;
        margin: 5px;
        text-align: center;
    }
    
    .color-box {
        width: 60px;
        height: 60px;
        border: 2px solid #ddd;
        border-radius: 8px;
        margin-bottom: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .color-label {
        font-size: 12px;
        font-weight: bold;
        color: #333;
    }
    
    .season-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_conda_environment_info():
    """Get information about the FashionCLIP conda environment"""
    try:
        # Test if the recommender service works
        script_path = os.path.join(os.path.dirname(__file__), 'recommender_service.py')
        conda_env = "FashionCLIP"
        
        # Test command
        cmd = [
            '/home/mayank/miniconda3/bin/conda', 'run', '-n', conda_env,
            'python', script_path, '--text', 'test', '--top_k', '1'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            if 'error' not in response:
                return True, "FashionCLIP environment is ready"
            else:
                return False, f"Service error: {response['error']}"
        else:
            return False, f"Failed to run recommender service: {result.stderr}"
            
    except Exception as e:
        return False, f"Error testing recommender service: {str(e)}"

def find_similar_items(query_image=None, query_text=None, top_k=10):
    """Find similar items using text or image query via subprocess"""
    try:
        script_path = os.path.join(os.path.dirname(__file__), 'recommender_service.py')
        conda_env = "FashionCLIP"
        
        # Build command
        cmd = [
            '/home/mayank/miniconda3/bin/conda', 'run', '-n', conda_env,
            'python', script_path, '--top_k', str(top_k)
        ]
        
        # Handle image query
        temp_image_path = None
        if query_image is not None:
            # Save uploaded file to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                # Read the uploaded file content and save it
                img = Image.open(query_image)
                img.save(tmp_file, format='JPEG')
                temp_image_path = tmp_file.name
            cmd.extend(['--image', temp_image_path])
        
        # Handle text query
        if query_text:
            cmd.extend(['--text', query_text])
        
        # Run the recommender service
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Clean up temporary file
        if temp_image_path:
            try:
                os.unlink(temp_image_path)
            except:
                pass
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            return response  # Return the full response dict with 'results' key
        else:
            return {"error": f"Service failed: {result.stderr}"}
            
    except subprocess.TimeoutExpired:
        return {"error": "Service timed out"}
    except Exception as e:
        return {"error": str(e)}


def analyze_color_palette(image_file):
    """Analyze color palette from uploaded image using conda environment"""
    try:
        script_path = os.path.join(os.path.dirname(__file__), 'color_analysis_service.py')
        conda_env = "ColourAnalysis"
        
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            img = Image.open(image_file)
            img.save(tmp_file, format='JPEG')
            temp_image_path = tmp_file.name
        
        # Build command
        cmd = [
            '/home/mayank/miniconda3/bin/conda', 'run', '-n', conda_env,
            'python', script_path, '--image', temp_image_path
        ]
        
        # Run the color analysis service
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        # Clean up temporary file
        try:
            os.unlink(temp_image_path)
        except:
            pass
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            return response
        else:
            return {"error": f"Color analysis failed: {result.stderr}"}
            
    except subprocess.TimeoutExpired:
        return {"error": "Color analysis timed out"}
    except Exception as e:
        return {"error": str(e)}


def get_clothes_by_color(color_name, top_k=5):
    """Get clothing recommendations for a specific color"""
    try:
        script_path = os.path.join(os.path.dirname(__file__), 'recommender_service.py')
        conda_env = "FashionCLIP"
        
        # Build search query for the color
        search_query = f"{color_name} clothes fashion"
        
        # Build command
        cmd = [
            '/home/mayank/miniconda3/bin/conda', 'run', '-n', conda_env,
            'python', script_path, '--text', search_query, '--top_k', str(top_k)
        ]
        
        # Run the recommender service
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            return response.get('results', [])
        else:
            return []
            
    except Exception as e:
        return []

def display_results(results, show_count=5):
    """Display search results with pagination - shows first 'show_count' items"""
    if 'error' in results:
        st.error(f"Error: {results['error']}")
        return
    
    if 'results' not in results or not results['results']:
        st.warning("No recommendations found.")
        return
    
    # Get the items to show based on show_count
    items_to_show = results['results'][:show_count]
    total_items = len(results['results'])
    
    st.markdown(f"## 🎯 Recommended Items")
    
    # Calculate number of columns (more columns for wider display)
    num_cols = min(5, len(items_to_show))
    cols = st.columns(num_cols)
    
    for idx, result in enumerate(items_to_show):
        col_idx = idx % num_cols
        with cols[col_idx]:
            try:
                # Check if image file exists before trying to load it
                if os.path.exists(result['path']):
                    # Load and display image
                    img = Image.open(result['path'])
                    st.image(img, use_container_width=True)
                else:
                    # Display placeholder for missing image
                    st.error(f"Image not found")
                    st.text(f"Path: {os.path.basename(result['path'])}")
                        
            except Exception as e:
                st.error(f"Error displaying image: {os.path.basename(result['path'])}")
                st.text(f"Details: {str(e)}")
    
    # Show "Load More" button if there are more items
    if show_count < total_items:
        remaining = total_items - show_count
        next_batch = min(5, remaining)
        
        # Center the button using columns
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Load More", key="load_more", type="secondary", use_container_width=True):
                # Update session state to show more items
                new_count = min(show_count + 5, total_items)
                st.session_state['show_count'] = new_count
                st.rerun()
    else:
        # Show completion message
        st.success(f"✨ All {total_items} recommendations loaded!")
    
    return len(items_to_show)

def run_style_transfer(content_image, style_image, iterations=50, remove_bg=True):
    """Run neural style transfer using the NSTclothes conda environment"""
    try:
        # Use the original NeuralStyleTransfer.py from NST_Clothes directory
        nst_script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'NST_Clothes', 'NeuralStyleTransfer.py')
        conda_env = "NSTclothes"
        
        # Create temporary files for input images
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as content_tmp:
            content_img = Image.open(content_image)
            content_img.save(content_tmp, format='JPEG')
            content_path = content_tmp.name
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as style_tmp:
            style_img = Image.open(style_image)
            style_img.save(style_tmp, format='JPEG')
            style_path = style_tmp.name
            
        # Create output directory for results
        output_dir = tempfile.mkdtemp()
        
        # Build command using conda environment
        # Note: The NST script ignores the --output parameter and saves to current directory
        # So we need to change directory to our output_dir when running
        cmd = [
            '/home/mayank/miniconda3/bin/conda', 'run', '-n', conda_env,
            'bash', '-c', 
            f'cd "{output_dir}" && python "{nst_script_path}" --content "{content_path}" --style "{style_path}" --output . --iterations {iterations} --checkpoint-iterations 10'
        ]
        
        # Run the style transfer
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # Increased timeout
        
        # Clean up input temp files
        try:
            os.unlink(content_path)
            os.unlink(style_path)
        except:
            pass
        
        if result.returncode == 0:
            # Debug: List all files in output directory
            try:
                output_files = os.listdir(output_dir)
                print(f"DEBUG: Output directory contains: {output_files}")
            except:
                output_files = []
            
            # Look for the generated stylized image
            stylized_image_path = os.path.join(output_dir, 'stylized-image.png')
            checkpoint_files = [f for f in output_files if f.endswith('_iter.png')]
            
            # Use the stylized image or the last checkpoint if available
            if os.path.exists(stylized_image_path):
                result_image_path = stylized_image_path
            elif checkpoint_files:
                # Sort checkpoint files by iteration number and get the last one
                checkpoint_files.sort(key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
                result_image_path = os.path.join(output_dir, checkpoint_files[-1])
            else:
                error_msg = f"No output images were generated. Output directory contains: {output_files}"
                if result.stderr:
                    error_msg += f"\nScript stderr: {result.stderr}"
                if result.stdout:
                    error_msg += f"\nScript stdout: {result.stdout}"
                return None, error_msg
            
            # Load the result image
            result_image = Image.open(result_image_path)
            
            # Apply background removal if requested
            if remove_bg:
                try:
                    # Save the result image temporarily for background removal
                    temp_result_path = os.path.join(output_dir, 'temp_result.png')
                    result_image.save(temp_result_path)
                    
                    # Use conda environment to run background removal
                    bg_removal_script = f"""
import sys
from rembg import remove
from PIL import Image
import io

# Load input image
with open('{temp_result_path}', 'rb') as inp_file:
    input_data = inp_file.read()

# Remove background
output_data = remove(input_data)

# Save the output
output_image = Image.open(io.BytesIO(output_data))
output_image.save('{temp_result_path.replace(".png", "_no_bg.png")}')
print("Background removal completed")
"""
                    
                    # Write the script to a temporary file
                    bg_script_path = os.path.join(output_dir, 'bg_removal.py')
                    with open(bg_script_path, 'w') as f:
                        f.write(bg_removal_script)
                    
                    # Run background removal in the same conda environment
                    bg_cmd = [
                        '/home/mayank/miniconda3/bin/conda', 'run', '-n', conda_env,
                        'python', bg_script_path
                    ]
                    
                    bg_result = subprocess.run(bg_cmd, capture_output=True, text=True, timeout=60)
                    
                    if bg_result.returncode == 0:
                        # Load the background-removed image
                        no_bg_path = temp_result_path.replace(".png", "_no_bg.png")
                        if os.path.exists(no_bg_path):
                            result_image = Image.open(no_bg_path)
                        else:
                            print("Background removal file not created, using original")
                    else:
                        print(f"Background removal failed: {bg_result.stderr}")
                        
                except Exception as e:
                    print(f"Background removal error: {e}")
                    # Continue with original image if background removal fails
            
            # Clean up output directory
            try:
                import shutil
                shutil.rmtree(output_dir)
            except:
                pass
            
            return result_image, None
        else:
            # Clean up output directory
            try:
                import shutil
                shutil.rmtree(output_dir)
            except:
                pass
            
            error_msg = f"Neural Style Transfer script failed (return code: {result.returncode})"
            if result.stderr:
                error_msg += f"\nStderr: {result.stderr}"
            if result.stdout:
                error_msg += f"\nStdout: {result.stdout}"
            
            # Check if the NST script exists
            if not os.path.exists(nst_script_path):
                error_msg += f"\nNST script not found at: {nst_script_path}"
            
            return None, error_msg
            
    except subprocess.TimeoutExpired:
        return None, "Style transfer timed out (>10 minutes)"
    except Exception as e:
        return None, f"Error running style transfer: {str(e)}"

def style_transfer_page():
    """Neural Style Transfer page"""
    st.title("🎨 Neural Style Transfer")
    st.markdown("Transform your fashion images with artistic styles!")
    
    # Instructions
    st.markdown("""
    ### How it works:
    1. Upload a **content image** (the clothing item you want to transform)
    2. Upload a **style image** (the artistic style you want to apply)  
    3. Choose your **quality level** (Light/Medium/High/Extreme)
    4. Click "Generate Styled Image" and wait for the magic! ✨
    
    """)
    
    # Create two columns for image uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Content Image")
        st.markdown("*Upload the clothing item to be transformed*")
        content_image = st.file_uploader(
            "Choose content image:", 
            type=['jpg', 'jpeg', 'png'],
            key="content_upload"
        )
        
        if content_image is not None:
            st.image(content_image, caption="Content Image", use_container_width=True)
    
    with col2:
        st.subheader("🎨 Style Image") 
        st.markdown("*Upload the style reference image*")
        style_image = st.file_uploader(
            "Choose style image:", 
            type=['jpg', 'jpeg', 'png'],
            key="style_upload"
        )
        
        if style_image is not None:
            st.image(style_image, caption="Style Image", use_container_width=True)
    
    # Settings section
    st.markdown("---")
    st.subheader("⚙️ Settings")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Style Transfer Quality Levels
        quality_options = {
            "🟢 Light (1 iteration)": 1,
            "🟡 Medium (5 iterations)": 5, 
            "🟠 High (10 iterations)": 10,
            "🔴 Extreme (20 iterations)": 20
        }
        
        selected_quality = st.selectbox(
            "Style Transfer Quality:",
            options=list(quality_options.keys()),
            index=1,  # Default to Medium
            help="Higher quality = better results but longer processing time"
        )
        
        iterations = quality_options[selected_quality]
        
        # Estimate processing time based on quality level
        time_estimates = {
            1: "~5 seconds",
            5: "~25 seconds", 
            10: "~50 seconds",
            20: "~1.5 minutes"
        }
            
    with col4:
        remove_bg = True
        st.markdown("""
        **Quality Levels:**
        - 🟢 **Light**: Light Style Blend
        - 🟡 **Medium**: Balanced Style Blend
        - 🟠 **High**: High Style Blend
        - 🔴 **Extreme**: Extreme Style Blend
        """)
    
    # Generate button
    st.markdown("---")
    
    if st.button("🎨 Generate Styled Image", type="primary", use_container_width=True):
        if content_image is None:
            st.error("❌ Please upload a content image!")
        elif style_image is None:
            st.error("❌ Please upload a style image!")
        else:
            # Create progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🚀 Starting neural style transfer...")
                progress_bar.progress(10)
                
                # Run style transfer
                with st.spinner("🎨 Applying neural style transfer... This may take a few minutes."):
                    result_image, error = run_style_transfer(
                        content_image, 
                        style_image, 
                        iterations=iterations,
                        remove_bg=remove_bg
                    )
                
                progress_bar.progress(100)
                
                if result_image is not None:
                    status_text.text("✅ Style transfer completed successfully!")
                    
                    # Display result
                    st.markdown("---")
                    st.subheader("🎉 Result")
                    
                    # Show before and after
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        st.markdown("**Original Content:**")
                        st.image(content_image, use_container_width=True)
                    
                    with result_col2:
                        st.markdown("**Styled Result:**")
                        st.image(result_image, use_container_width=True)
                    
                    # Download button
                    img_buffer = BytesIO()
                    result_image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download Styled Image",
                        data=img_buffer.getvalue(),
                        file_name=f"styled_image_{int(time.time())}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    
                else:
                    status_text.text("❌ Style transfer failed!")
                    st.error(f"Error: {error}")
                    
            except Exception as e:
                progress_bar.progress(0)
                status_text.text("❌ An error occurred!")
                st.error(f"Unexpected error: {str(e)}")
    
def main():
    """Main Streamlit app with navigation"""
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("🎯 Features")
        
        # Navigation
        page = st.selectbox(
            "Select Feature:",
            ["🔍 Fashion Recommender", "🎨 Neural Style Transfer", "🎨 Color Palette Analysis"],
            key="navigation"
        )
        
        st.markdown("---")
        
        if page == "🔍 Fashion Recommender":
            st.markdown("""
            ### Fashion Recommender
            - Find similar fashion items
            - Search by text or image
            - Browse recommendations
            """)
        elif page == "🎨 Neural Style Transfer":
            st.markdown("""
            ### Neural Style Transfer
            - Apply artistic styles to clothes
            - Upload content & style images  
            - 4 quality levels (Light/Medium/High/Extreme)
            - Fixed checkpoints every 10 iterations
            - Background removal option
            """)
        else:  # Color Palette Analysis
            st.markdown("""
            ### Color Palette Analysis
            - Upload your photo for analysis
            - Get personalized color recommendations
            - View matching fashion items
            - Discover your seasonal color palette
            """)
        
        st.markdown("---")
        st.markdown("""
        ### Coming Soon:
        - 👗 Virtual Try-On
        - 🎨 Custom Cloth Creator  
        """)
    
    # Route to appropriate page
    if page == "🔍 Fashion Recommender":
        fashion_recommender_page()
    elif page == "🎨 Neural Style Transfer":
        style_transfer_page()
    elif page == "🎨 Color Palette Analysis":
        color_palette_analysis_page()


def color_palette_analysis_page():
    """Color Palette Analysis page"""
    
    st.title("🎨 Personal Color Palette Analysis")
    
    st.markdown("""
    Discover your personal color palette based on your skin tone and facial features. 
    Upload a clear photo of yourself to get personalized color recommendations and matching fashion items.
    """)
    
    # Upload section
    st.markdown('<div class="search-section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Upload Your Photo")
        uploaded_file = st.file_uploader(
            "Choose a clear photo of your face:",
            type=['jpg', 'jpeg', 'png'],
            key="color_analysis_upload",
            help="For best results, use a photo with good lighting and minimal makeup"
        )
        
        if uploaded_file:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)
    
    with col2:
        st.subheader("ℹ️ Tips for Best Results")
        st.markdown("""
        - Use a photo with good, natural lighting
        - Face should be clearly visible
        - Minimal or no makeup for accurate analysis
        - Avoid heavy filters or editing
        - Front-facing photo works best
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis button
    if uploaded_file:
        analyze_btn = st.button("🎨 Analyze My Color Palette", type="primary", key="analyze_color")
        
        if analyze_btn:
            with st.spinner("🔍 Analyzing your color palette... This may take a moment."):
                result = analyze_color_palette(uploaded_file)
            
            if "error" in result:
                st.error(f"❌ Analysis failed: {result['error']}")
            else:
                # Display results
                st.success("✅ Color palette analysis complete!")
                
                # Show season and palette info
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader(f"🌺 Your Season: {result['season_name']}")
                    st.markdown(f"**Dominant Palette:** {result['dominant_palette'].upper()}")
                
                with col2:
                    st.subheader("🎨 Your Color Palette")
                    
                    # Display color swatches using columns instead of HTML
                    color_cols = st.columns(len(result['color_palette']))
                    
                    for idx, color in enumerate(result['color_palette']):
                        with color_cols[idx]:
                            # Create a color swatch using HTML in a more controlled way
                            color_html = f'''
                            <div style="text-align: center; margin: 10px;">
                                <div style="width: 70px; height: 70px; 
                                            background-color: {color['hex']}; 
                                            border: 3px solid #fff;
                                            border-radius: 12px; 
                                            margin: 0 auto 8px auto;
                                            box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                                </div>
                                <div style="font-size: 12px; font-weight: bold; color: #333; margin-bottom: 2px;">
                                    {color['name']}
                                </div>
                                <div style="font-size: 10px; color: #666;">
                                    {color['hex']}
                                </div>
                            </div>
                            '''
                            st.markdown(color_html, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Get clothing recommendations for each color
                st.subheader("👗 Fashion Recommendations by Color")
                
                for color_info in result['color_palette']:
                    color_name = color_info['name']
                    color_hex = color_info['hex']
                    
                    with st.expander(f"🎨 {color_name} Fashion Items", expanded=False):
                        # Show color swatch with better styling
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin-bottom: 20px; 
                                    background: linear-gradient(45deg, #f8f9fa, #ffffff);
                                    padding: 15px; border-radius: 10px; border: 1px solid #e9ecef;">
                            <div style="width: 50px; height: 50px; 
                                        background-color: {color_hex}; 
                                        border: 3px solid #fff;
                                        border-radius: 8px; 
                                        margin-right: 15px;
                                        box-shadow: 0 3px 10px rgba(0,0,0,0.2);"></div>
                            <div>
                                <h4 style="margin: 0; color: #333;">Fashion items in {color_name}</h4>
                                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">Color: {color_hex}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Get recommendations for this color
                        with st.spinner(f"Finding {color_name} fashion items..."):
                            color_results = get_clothes_by_color(color_name, top_k=4)
                        
                        if color_results:
                            # Display results in a grid
                            cols = st.columns(4)
                            for idx, item in enumerate(color_results):
                                with cols[idx % 4]:
                                    # Use 'path' key instead of 'image_path' (matching recommender service response)
                                    image_path = item.get('path', item.get('image_path', ''))
                                    if image_path and os.path.exists(image_path):
                                        st.image(image_path, use_container_width=True)
                                    else:
                                        st.error("Image not found")
                                        st.text(f"Path: {os.path.basename(image_path) if image_path else 'N/A'}")
                        else:
                            st.info(f"🔍 No specific {color_name} items found. Try searching for '{color_name} fashion' in the Fashion Recommender!")


def fashion_recommender_page():
    """Fashion Recommender page"""
    
    # Title for this page
    st.title("🌟 Fashion Recommender System")
    
    st.markdown("Find similar fashion items using text descriptions or image uploads")
        
    # Main search interface
    st.markdown('<div class="search-section">', unsafe_allow_html=True)
    
    # Create two columns for text and image search
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔍 Text Search")
        query_text = st.text_input(
            "Describe the clothing item:", 
            placeholder="e.g., black dress, blue jeans, red shirt...",
            key="text_search"
        )
        text_search_btn = st.button("Search by Text", key="text_btn", type="primary")
    
    with col2:
        st.subheader("📷 Image Search") 
        uploaded_file = st.file_uploader(
            "Upload an image:", 
            type=['jpg', 'jpeg', 'png'],
            key="image_upload"
        )
        image_search_btn = st.button("Search by Image", key="image_btn", type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize session state for pagination
    if 'search_results' not in st.session_state:
        st.session_state['search_results'] = None
    if 'show_count' not in st.session_state:
        st.session_state['show_count'] = 5
    if 'last_query' not in st.session_state:
        st.session_state['last_query'] = None
    
    # Process search requests
    if text_search_btn and query_text.strip():
        # Reset pagination for new search
        st.session_state['show_count'] = 5
        current_query = f"text:{query_text.strip()}"
        
        # Only fetch new results if query changed
        if st.session_state['last_query'] != current_query:
            with st.spinner("🔍 Finding similar fashion items..."):
                # Fetch more results initially to enable pagination (fetch 20 but show 5)
                results = find_similar_items(query_text=query_text.strip(), top_k=20)
                st.session_state['search_results'] = results
                st.session_state['last_query'] = current_query
        
        # Display results with current show_count
        if st.session_state['search_results']:
            display_results(st.session_state['search_results'], st.session_state['show_count'])
    
    elif image_search_btn and uploaded_file is not None:
        # Reset pagination for new search
        st.session_state['show_count'] = 5
        current_query = f"image:{uploaded_file.name}"
        
        # Only fetch new results if query changed
        if st.session_state['last_query'] != current_query:
            with st.spinner("📷 Analyzing image and finding similar items..."):
                # Fetch more results initially to enable pagination (fetch 20 but show 5)
                results = find_similar_items(query_image=uploaded_file, top_k=20)
                st.session_state['search_results'] = results
                st.session_state['last_query'] = current_query
        
        # Display results with current show_count
        if st.session_state['search_results']:
            display_results(st.session_state['search_results'], st.session_state['show_count'])
    
    # Display existing results if available (for pagination updates)
    elif st.session_state['search_results'] and st.session_state['last_query']:
        display_results(st.session_state['search_results'], st.session_state['show_count'])

if __name__ == "__main__":
    main()