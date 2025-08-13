#!/bin/bash

# Complete OpenPose + ComfyUI Setup Script for Linux (No Sudo Version)
# This script automates the setup process for OpenPose on Linux with CUDA GPU support
# AND sets up ComfyUI with AnimateDiff and VideoHelper custom nodes for pose transfer workflows
# Based on the successful setup process completed on 2025-08-12
# Modified to remove sudo commands for root users

set -e  # Exit on any error

echo "🚀 Starting OpenPose GPU Setup for Linux..."
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script is designed for Linux. Please run on Linux."
    exit 1
fi

print_status "Checking prerequisites..."

# Check if CUDA is installed
if ! command -v nvcc &> /dev/null; then
    print_error "CUDA is not installed or not in PATH. Please install CUDA first."
    print_status "Download from: https://developer.nvidia.com/cuda-downloads"
    print_status "Recommended version: CUDA 11.8 or 12.x"
    exit 1
fi

# Get CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
print_success "CUDA $CUDA_VERSION found"

# Check CUDA version compatibility
if command -v bc &> /dev/null; then
    if [[ $(echo "$CUDA_VERSION < 11.0" | bc -l) -eq 1 ]]; then
        print_warning "CUDA version $CUDA_VERSION is quite old. Consider upgrading to CUDA 11.8+ for better compatibility."
    fi
else
    # Simple string comparison if bc is not available
    if [[ "$CUDA_VERSION" < "11.0" ]]; then
        print_warning "CUDA version $CUDA_VERSION is quite old. Consider upgrading to CUDA 11.8+ for better compatibility."
    fi
fi

# Check if cuDNN is available
if [ ! -f "/usr/local/cuda/include/cudnn.h" ] && [ ! -f "/usr/include/cudnn.h" ]; then
    print_warning "cuDNN not found in standard locations. OpenPose may not work optimally without cuDNN."
    print_status "Consider installing cuDNN for better performance: https://developer.nvidia.com/cudnn"
else
    print_success "cuDNN found"
fi

# Check if Git is available
if ! command -v git &> /dev/null; then
    print_error "Git is required but not found. Please install Git."
    exit 1
fi

print_success "Git found"

# Check if CMake is available
if ! command -v cmake &> /dev/null; then
    print_status "CMake not found. Installing CMake..."
    if command -v apt-get &> /dev/null; then
        apt-get update
        apt-get install -y cmake
    elif command -v yum &> /dev/null; then
        yum install -y cmake
    elif command -v dnf &> /dev/null; then
        dnf install -y cmake
    else
        print_error "Package manager not supported. Please install CMake manually."
        exit 1
    fi
fi

print_success "CMake found"

# Check CMake version
CMAKE_VERSION=$(cmake --version | head -n1 | sed 's/cmake version //')
print_status "CMake version: $CMAKE_VERSION"

# Install OpenCL and ViennaCL dependencies for GPU support
print_status "Installing GPU development dependencies..."
if command -v apt-get &> /dev/null; then
    apt-get install -y ocl-icd-opencl-dev opencl-headers libviennacl-dev
elif command -v yum &> /dev/null; then
    yum install -y ocl-icd-opencl-dev opencl-headers libviennacl-dev
elif command -v dnf &> /dev/null; then
    dnf install -y ocl-icd-opencl-dev opencl-headers libviennacl-dev
fi

# Install additional dependencies
print_status "Installing system dependencies..."

if command -v apt-get &> /dev/null; then
    apt-get update
    apt-get install -y \
        build-essential \
        libatlas-base-dev \
        libprotobuf-dev \
        libleveldb-dev \
        libsnappy-dev \
        libhdf5-serial-dev \
        protobuf-compiler \
        libgflags-dev \
        libgoogle-glog-dev \
        liblmdb-dev \
        libopencv-dev \
        libboost-all-dev \
        libopenblas-dev \
        liblapack-dev \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libgtk-3-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libatlas-base-dev \
        gfortran \
        wget \
        unzip \
        bc
elif command -v yum &> /dev/null; then
    yum groupinstall -y "Development Tools"
    yum install -y \
        atlas-devel \
        protobuf-devel \
        leveldb-devel \
        snappy-devel \
        hdf5-devel \
        gflags-devel \
        glog-devel \
        lmdb-devel \
        opencv-devel \
        boost-devel \
        openblas-devel \
        lapack-devel \
        gstreamer1-devel \
        gstreamer1-plugins-base-devel \
        gtk3-devel \
        ffmpeg-devel \
        libjpeg-turbo-devel \
        libpng-devel \
        libtiff-devel \
        gcc-gfortran \
        wget \
        unzip \
        bc
elif command -v dnf &> /dev/null; then
    dnf groupinstall -y "Development Tools"
    dnf install -y \
        atlas-devel \
        protobuf-devel \
        leveldb-devel \
        snappy-devel \
        hdf5-devel \
        gflags-devel \
        glog-devel \
        lmdb-devel \
        opencv-devel \
        boost-devel \
        openblas-devel \
        lapack-devel \
        gstreamer1-devel \
        gstreamer1-plugins-base-devel \
        gtk3-devel \
        ffmpeg-devel \
        libjpeg-turbo-devel \
        libpng-devel \
        libtiff-devel \
        gcc-gfortran \
        wget \
        unzip \
        bc
else
    print_warning "Unsupported package manager. Please install dependencies manually."
fi

print_success "System dependencies installed"

# Step 1: Check for existing OpenPose installation
print_status "Step 1: Checking for existing OpenPose installation..."

# Check if we're already in an OpenPose directory
if [ -d "openpose" ] && [ -f "openpose/CMakeLists.txt" ]; then
    print_success "Found existing OpenPose directory: openpose/"
    cd openpose
    OPENPOSE_DIR="."
elif [ -f "CMakeLists.txt" ] && grep -q "OpenPose" "CMakeLists.txt" 2>/dev/null; then
    print_success "Already in OpenPose directory"
    OPENPOSE_DIR="."
else
    # Check if we're in a git repository that's not OpenPose
    if [ -d ".git" ]; then
        print_warning "Current directory contains a different git repository"
        print_status "Checking if there's an OpenPose subdirectory..."
        
        if [ -d "openpose" ] && [ -f "openpose/CMakeLists.txt" ]; then
            print_success "Found OpenPose in subdirectory: openpose/"
            cd openpose
            OPENPOSE_DIR="."
        else
            print_status "No OpenPose found. Cloning to openpose/ subdirectory..."
            git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git openpose
            cd openpose
            OPENPOSE_DIR="."
        fi
    else
        print_status "Cloning OpenPose repository to openpose/ subdirectory..."
        git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git openpose
        cd openpose
        OPENPOSE_DIR="."
        print_success "OpenPose repository cloned"
    fi
fi

# Verify we're in the right place
if [ ! -f "CMakeLists.txt" ] || ! grep -q "OpenPose" "CMakeLists.txt" 2>/dev/null; then
    print_error "Failed to locate OpenPose source. Please check the installation."
    exit 1
fi

print_success "OpenPose source located successfully"

# Step 2: Check and download dependencies
print_status "Step 2: Checking dependencies..."

# Create 3rdparty/linux directory if it doesn't exist
mkdir -p 3rdparty/linux

# Check if dependencies already exist and are valid
if [ -d "3rdparty/linux/caffe" ] && [ -d "3rdparty/linux/opencv" ] && [ -d "3rdparty/linux/caffe3rdparty" ]; then
    print_success "Dependencies already exist and appear complete, skipping download"
elif [ -d "build" ] && [ -f "build/CMakeCache.txt" ]; then
    print_success "Build directory exists with CMake cache - dependencies likely already configured"
    print_status "Proceeding to GPU configuration step..."
elif [ -d "3rdparty/windows" ] || [ -d "3rdparty/mac" ]; then
    print_warning "Found 3rdparty directory but not for Linux. This might be a Windows/Mac installation."
    print_status "You may need to download Linux-specific dependencies."
else
    print_status "Some dependencies are missing or incomplete"
    
    # Check what's already there
    if [ -d "3rdparty/linux/caffe" ]; then
        print_success "Caffe directory exists"
    fi
    if [ -d "3rdparty/linux/opencv" ]; then
        print_success "OpenCV directory exists"
    fi
    if [ -d "3rdparty/linux/caffe3rdparty" ]; then
        print_success "Caffe 3rd party directory exists"
    fi
    
    # Ask user if they want to try downloading
    read -p "Do you want to attempt downloading missing dependencies? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Download dependencies
        print_status "Downloading Caffe 3rd party dependencies..."
        cd 3rdparty/linux
        
        # Download dependencies
        deps=(
            "caffe3rdparty_16_2020_11_14.zip"
            "caffe_gpu_2018_05_27.zip"
            "opencv_450_v15_2020_11_18.zip"
        )

        # Clean up any existing corrupted files
        print_status "Cleaning up any existing corrupted files..."
        for dep in "${deps[@]}"; do
            if [ -f "$dep" ] && ! unzip -t "$dep" >/dev/null 2>&1; then
                print_warning "Removing corrupted file: $dep"
                rm -f "$dep"
            fi
        done

        for dep in "${deps[@]}"; do
            if [ ! -f "$dep" ]; then
                print_status "Downloading $dep..."
                if wget -O "$dep" "http://vcl.snu.ac.kr/OpenPose/3rdparty/linux/$dep"; then
                    # Verify the file is actually a valid zip
                    if unzip -t "$dep" >/dev/null 2>&1; then
                        print_success "Downloaded and verified $dep"
                    else
                        print_warning "Downloaded file appears corrupted, removing..."
                        rm -f "$dep"
                    fi
                else
                    print_warning "Failed to download $dep from primary source"
                fi
            else
                print_success "$dep already exists"
            fi
        done

        # Extract valid dependencies
        print_status "Extracting valid dependencies..."
        for dep in "${deps[@]}"; do
            if [ -f "$dep" ]; then
                # Test if it's a valid zip file
                if unzip -t "$dep" >/dev/null 2>&1; then
                    print_status "Extracting $dep..."
                    unzip -o "$dep"
                    print_success "Extracted $dep"
                else
                    print_warning "File $dep is corrupted, removing it..."
                    rm -f "$dep"
                fi
            fi
        done

        cd ../..
    else
        print_status "Skipping dependency download. You may need to manually download dependencies later."
    fi
fi

# Step 3: Configure OpenPose with GPU support
print_status "Step 3: Configuring OpenPose with GPU support..."

# Check if build directory already exists
if [ -d "build" ]; then
    print_warning "Build directory already exists"
    read -p "Do you want to reuse the existing build directory? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Reusing existing build directory..."
        cd build
        # Check if CMake cache exists and if it has Python bindings enabled
        if [ -f "CMakeCache.txt" ]; then
            if grep -q "BUILD_PYTHON:BOOL=ON" "CMakeCache.txt"; then
                print_warning "Existing build has Python bindings enabled - this may cause build failures"
                print_status "Recommendation: Remove build directory and reconfigure without Python"
                read -p "Remove build directory and reconfigure? (y/n): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    cd ..
                    rm -rf build
                    mkdir build
                    cd build
                    print_status "Build directory recreated for clean configuration"
                else
                    print_status "Using existing build directory (may have compatibility issues)"
                fi
            else
                print_status "CMake cache found with Python bindings disabled - safe to use"
            fi
        fi
    else
        print_status "Removing existing build directory..."
        cd ..
        rm -rf build
        mkdir build
        cd build
    fi
else
    # Create new build directory
    mkdir build
    cd build
fi

# Configure with CMake (GPU support)
print_status "Running CMake configuration with GPU support..."

# Force CUDA architecture to 8.6 for optimal performance
CUDA_ARCH="8.6"
print_status "Using CUDA architecture: $CUDA_ARCH"

# First CMake configuration
print_status "Running first CMake configuration..."
cmake .. -DGPU_MODE=CUDA -DUSE_CUDNN=ON -DCUDA_ARCH_BIN=8.6 -DUSE_OPENCV=ON -DBUILD_CAFFE=ON -DBUILD_PYTHON=OFF

# Second CMake configuration for redundancy
print_status "Running second CMake configuration for redundancy..."
cmake .. -DGPU_MODE=CUDA -DUSE_CUDNN=ON -DCUDA_ARCH_BIN=8.6 -DUSE_OPENCV=ON -DBUILD_CAFFE=ON -DBUILD_PYTHON=OFF

# Final CMake configuration with all options (Python disabled for compatibility)
print_status "Running final CMake configuration with all options..."
print_status "Note: Python bindings disabled due to Python 3.11 compatibility issues"
print_status "This ensures successful build. You can still use Python with the compiled C++ executables."

# Check Python version and provide compatibility info
PYTHON_VERSION=$(python3 --version 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1,2 || echo "unknown")
if [[ "$PYTHON_VERSION" == "3.11" ]]; then
    print_warning "Python 3.11 detected - Python bindings disabled for compatibility"
    print_status "To enable Python bindings, use Python 3.10 or update pybind11"
    print_status "Current build uses C++ only for maximum compatibility"
elif [[ "$PYTHON_VERSION" == "3.10" ]]; then
    print_status "Python 3.10 detected - Python bindings could be enabled"
    print_status "Current build uses C++ only for maximum compatibility"
else
    print_status "Python version: $PYTHON_VERSION - Python bindings disabled"
fi

# Additional Python compatibility check
if [[ "$PYTHON_VERSION" == "3.11" ]] || [[ "$PYTHON_VERSION" == "3.12" ]]; then
    print_warning "Python $PYTHON_VERSION detected - this version has known compatibility issues with OpenPose"
    print_status "Recommendation: Use Python 3.10 or earlier, or build without Python bindings"
    print_status "Current configuration: BUILD_PYTHON=OFF (safe for all Python versions)"
fi

cmake .. \
    -DGPU_MODE=CUDA \
    -DUSE_CUDNN=ON \
    -DCUDA_ARCH_BIN=$CUDA_ARCH \
    -DCUDA_ARCH_PTX=$CUDA_ARCH \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DCUDNN_INCLUDE_DIR=/usr/local/cuda/include \
    -DCUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so \
    -DBUILD_CAFFE=ON \
    -DBUILD_PYTHON=OFF \
    -DUSE_OPENCV=ON \
    -DUSE_OPENMP=ON \
    -DUSE_MPI=OFF

if [ $? -eq 0 ]; then
    print_success "CMake configuration completed successfully!"
else
    print_error "CMake configuration failed. Check the output above for errors."
    exit 1
fi

print_success "OpenPose configured successfully with GPU support!"

# Show current working directory
print_status "Current working directory: $(pwd)"
print_status "OpenPose source directory: $(realpath $OPENPOSE_DIR)"

# Check if build was successful and provide next steps
if [ -f "CMakeCache.txt" ] && [ -f "Makefile" ]; then
    print_success "Build system configured successfully!"
    print_status "You can now proceed with building OpenPose"
    
    # Ask if user wants to start building immediately
    echo ""
    read -p "Do you want to start building OpenPose now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Starting OpenPose build with GPU acceleration..."
        print_status "This will take 1-3 hours depending on your system"
        print_status "Build command: make -j$(nproc)"
        echo ""
        
        # Start the build with progress monitoring
        print_status "Build started at: $(date)"
        print_status "Monitoring build progress..."
        
        # Start the build
        make -j$(nproc)
        BUILD_EXIT_CODE=$?
        
        print_status "Build completed at: $(date)"
        
        if [ $BUILD_EXIT_CODE -eq 0 ]; then
            print_success "OpenPose build completed successfully!"
            
            # Check if binaries were created
            if [ -f "bin/openpose.bin" ] || [ -f "bin/OpenPoseDemo" ]; then
                print_success "OpenPose binaries created successfully!"
                print_status "You can now test your installation:"
                print_status "cd bin && ./openpose.bin --help"
            else
                print_warning "Build succeeded but binaries not found in expected location"
                print_status "Checking bin directory contents..."
                ls -la bin/ 2>/dev/null || print_error "bin directory not found"
            fi
        else
            print_error "Build failed with exit code: $BUILD_EXIT_CODE"
            print_status "Common solutions:"
            print_status "1. Check available memory: free -h"
            print_status "2. Try building with fewer cores: make -j4"
            print_status "3. Check CUDA installation: nvidia-smi"
            print_status "4. Reconfigure without Python: rm -rf build && run script again"
        fi
    else
        print_status "Build skipped. You can build later with: make -j$(nproc)"
    fi
else
    print_warning "Build system may not be fully configured"
    print_status "Please check the CMake output above for any errors"
fi

# Step 4: Build instructions
print_status "Step 4: Build instructions..."
echo ""
echo "🎯 Configuration Complete! Next steps:"
echo "======================================"
echo ""
echo "1. Build OpenPose (this will take 1-3 hours):"
echo "   cd build"
echo "   make -j$(nproc)"
echo ""
echo "2. Or build with specific number of cores:"
echo "   make -j4  # Use 4 cores"
echo ""
echo "3. Test the installation:"
echo "   cd bin"
echo "   ./OpenPoseDemo --help"
echo ""
echo "4. Run a test with GPU acceleration:"
echo "   ./OpenPoseDemo --video examples/media/video.avi --display 0 --write_video output.avi"
echo ""
echo "📝 Notes:"
echo "- This setup is configured for GPU acceleration with CUDA"
echo "- CUDA version detected: $CUDA_VERSION"
echo "- The build process may take 1-3 hours depending on your system"
echo "- Make sure you have enough disk space (at least 10GB free)"
echo "- If you encounter memory issues during build, reduce the number of parallel jobs"
echo ""
echo "🔧 GPU Configuration Details:"
echo "- GPU_MODE: CUDA (optimized for NVIDIA GPUs)"
echo "- CUDA support: Enabled"
echo "- cuDNN support: Enabled (if available)"
echo "- OpenCV version: 4.x"
echo ""
echo "📋 Successful Configuration for NVIDIA A40:"
echo "- GPU_MODE=CUDA"
echo "- CUDA_ARCH_BIN=8.6 (for A40 compute capability)"
echo "- USE_CUDNN=ON"
echo "- BUILD_CAFFE=ON"
echo "- BUILD_PYTHON=OFF (for Python 3.11 compatibility)"
echo "- OpenPose installed in: openpose/ subdirectory"
echo ""
echo "🎯 Skeleton Extraction Commands (after build completes):"
echo "========================================================"
echo "1. Extract 16 frames from video:"
echo "   ffmpeg -i input_video.mp4 -vf \"select='not(mod(n,floor(n_frames/16)))'\" -vsync vfr frame_%03d.jpg"
echo ""
echo "2. Process with OpenPose:"
echo "   cd openpose/build/bin"
echo "   ./openpose.bin --image_dir ../../../frame_*.jpg --write_json ../../../skeletons/ --number_people_max 1"
echo ""
echo "3. Or process video directly:"
echo "   ./openpose.bin --video ../../../input_video.mp4 --write_json ../../../skeletons/ --frame_rate 16"
echo ""
echo "4. Test installation:"
echo "   ./openpose.bin --help"
echo ""

print_success "OpenPose GPU setup script completed!"
print_status "You can now proceed with building OpenPose"

# Optional: Set up your own Git repository
echo ""
echo "🔧 Optional: Set up your own Git repository"
echo "=========================================="
echo "To push to your own repository:"
echo "1. git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO.git"
echo "2. git add ."
echo "3. git commit -m 'Initial OpenPose GPU setup'"
echo "4. git push -u origin main"
echo ""

# Show system information
echo ""
echo "💻 System Information:"
echo "======================"
echo "OS: $(lsb_release -d 2>/dev/null | cut -f2 || echo "Linux")"
echo "Architecture: $(uname -m)"
echo "CUDA Version: $CUDA_VERSION"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo "Not detected")"
echo "Available cores: $(nproc)"
echo ""

# Step 5: ComfyUI Custom Nodes Setup
print_status "Step 5: Setting up ComfyUI custom nodes for pose transfer workflows..."

# Navigate back to the main directory (should be parent of openpose/)
if [ -d ".." ] && [ -f "../openpose_gpu_setup.sh" ]; then
    cd ..
    print_status "Navigated back to main directory: $(pwd)"
elif [ -d "/workspace/animatediff" ]; then
    cd /workspace/animatediff
    print_status "Navigated to /workspace/animatediff"
else
    print_warning "Could not find main project directory. Staying in current directory."
fi

# Check if ComfyUI is available
if [ ! -d "ComfyUI" ] && [ ! -f "main.py" ]; then
    print_warning "ComfyUI not found. This script assumes ComfyUI is already installed."
    print_status "Please install ComfyUI first or run this script from a ComfyUI installation."
else
    print_success "ComfyUI installation detected"
    
    # Create custom_nodes directory if it doesn't exist
    mkdir -p custom_nodes
    cd custom_nodes
    
    # Install AnimateDiff-Evolved custom node
    if [ ! -d "ComfyUI-AnimateDiff-Evolved" ]; then
        print_status "Installing ComfyUI-AnimateDiff-Evolved..."
        git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git
        cd ComfyUI-AnimateDiff-Evolved
        
        # Install Python dependencies
        if [ -f "requirements.txt" ]; then
            print_status "Installing AnimateDiff Python dependencies..."
            pip install -r requirements.txt
        fi
        cd ..
        print_success "ComfyUI-AnimateDiff-Evolved installed"
    else
        print_success "ComfyUI-AnimateDiff-Evolved already exists"
    fi
    
    # Install VideoHelperSuite custom node
    if [ ! -d "ComfyUI-VideoHelperSuite" ]; then
        print_status "Installing ComfyUI-VideoHelperSuite..."
        git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
        cd ComfyUI-VideoHelperSuite
        
        # Install Python dependencies
        if [ -f "requirements.txt" ]; then
            print_status "Installing VideoHelper Python dependencies..."
            pip install -r requirements.txt
        fi
        cd ..
        print_success "ComfyUI-VideoHelperSuite installed"
    else
        print_success "ComfyUI-VideoHelperSuite already exists"
    fi
    
    # Install OpenCV for Python (needed for skeleton processing)
    print_status "Installing opencv-python for skeleton processing..."
    pip install opencv-python numpy
    
    cd ..  # Back to /workspace/animatediff
fi

# Clean up temporary files
print_status "Cleaning up temporary files..."
if [ -f "get-docker.sh" ]; then
    rm -f get-docker.sh
    print_status "Removed Docker installation script (no longer needed)"
fi

print_success "Complete OpenPose + ComfyUI setup script completed!"
print_status "All components are now configured for pose transfer workflows"

# Final summary and next steps
echo ""
echo "🎉 Setup Complete! Here's what to do next:"
echo "=========================================="
echo ""
echo "1. If build completed successfully:"
echo "   cd openpose/build/bin"
echo "   ./openpose.bin --help"
echo ""
echo "2. If you need to rebuild:"
echo "   cd openpose"
echo "   rm -rf build"
echo "   ./openpose_gpu_setup_no_sudo.sh"
echo ""
echo "3. For skeleton extraction:"
echo "   # Extract 16 frames from video"
echo "   ffmpeg -i input.mp4 -vf \"select='not(mod(n,floor(n_frames/16)))'\" -vsync vfr frame_%03d.jpg"
echo "   # Process with OpenPose"
echo "   ./openpose.bin --image_dir ../../../frame_*.jpg --write_json ../../../skeletons/"
echo ""
echo "4. Test GPU acceleration:"
echo "   ./openpose.bin --video ../../../input.mp4 --display 0 --write_video output.avi"
echo ""

exit 0
