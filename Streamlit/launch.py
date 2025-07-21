#!/usr/bin/env python3
"""
Fashion Recommender System Launcher
Checks dependencies and launches the Streamlit app
Now uses isolated environments - Streamlit in its own env, FashionCLIP in conda env
"""

import os
import sys
import subprocess
import importlib.util

def check_dependency(package_name, import_name=None):
    """Check if a Python package is installed"""
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    return spec is not None

def check_files():
    """Check if required files exist"""
    required_files = [
        "../Recommender/Database/fashion.index",
        "../Recommender/Database/image_paths.json",
        "recommender_service.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
    
    return missing_files

def check_conda_environment():
    """Check if FashionCLIP conda environment exists and is working"""
    try:
        # Test if we can run the recommender service
        script_path = os.path.join(os.path.dirname(__file__), 'recommender_service.py')
        cmd = [
            '/home/mayank/miniconda3/bin/conda', 'run', '-n', 'FashionCLIP',
            'python', script_path, '--text', 'test', '--top_k', '1'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return True, "FashionCLIP environment is ready"
        else:
            return False, f"Environment error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "Environment check timed out"
    except Exception as e:
        return False, f"Error checking environment: {str(e)}"

def main():
    print("🚀 Fashion Recommender System Launcher")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check Streamlit dependencies (minimal)
    streamlit_deps = [
        ("streamlit", "streamlit"),
        ("Pillow", "PIL"),
    ]
    
    missing_deps = []
    for package, import_name in streamlit_deps:
        if check_dependency(package, import_name):
            print(f"✅ {package} (Streamlit env)")
        else:
            print(f"❌ {package} (missing in Streamlit env)")
            missing_deps.append(package)
    
    # Check required files
    missing_files = check_files()
    
    if missing_files:
        print("\n❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        if "recommender_service.py" in missing_files:
            print("\n💡 The recommender_service.py should be in the Streamlit directory!")
        else:
            print("\n💡 Please run 'python build_index.py' from the Recommender/FashionCLIP directory first!")
        sys.exit(1)
    else:
        print("✅ All required files found")
    
    # Check FashionCLIP conda environment
    print("\n🔍 Checking FashionCLIP environment...")
    env_ready, env_message = check_conda_environment()
    
    if env_ready:
        print(f"✅ {env_message}")
    else:
        print(f"❌ {env_message}")
        print("\n💡 Run './setup_fashionclip_env.sh' to set up the FashionCLIP environment")
        sys.exit(1)
    
    if missing_deps:
        print(f"\n❌ Missing Streamlit dependencies: {', '.join(missing_deps)}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n🎉 All checks passed!")
    print("🌐 Architecture:")
    print("   - Streamlit App: Running in current environment")
    print("   - Recommender Service: Running in FashionCLIP conda environment")
    print("🌐 Starting Streamlit app...")
    print("-" * 50)
    
    # Launch Streamlit
    try:
        subprocess.run(["streamlit", "run", "app.py", "--server.port", "8501"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to start Streamlit. Please check the installation.")
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")

if __name__ == "__main__":
    main()
