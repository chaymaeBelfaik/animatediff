#!/bin/bash

# Complete OpenPose + ComfyUI Setup Script for Linux
# This script automates the setup process for OpenPose on Linux with CUDA GPU support
# AND sets up ComfyUI with AnimateDiff and VideoHelper custom nodes for pose transfer workflows
# Based on the successful setup process completed on 2025-08-12

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
if [ ! -d "/usr/local/cuda/include/cudnn.h" ] && [ ! -d "/usr/include/cudnn.h" ]; then
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

# Install OpenCL and ViennaCL dependencies for GPU support
print_status "Installing GPU development dependencies..."
if command -v apt-get &> /dev/null; then
    apt-get install -y ocl-icd-opencl-dev opencl-headers libviennacl-dev
elif command -v yum &> /dev/null; then
    yum install -y ocl-icd-opencl-dev opencl-headers libviennacl-dev
elif command -v dnf &> /dev/null; then
    dnf install -y ocl-icd-opencl-dev opencl-headers libviennacl-dev
fi

print_success "CMake found"

# Check CMake version
CMAKE_VERSION=$(cmake --version | head -n1 | sed 's/cmake version //')
print_status "CMake version: $CMAKE_VERSION"

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
            print_status "No OpenPose found. Cloning to new directory..."
            cd ..
            git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git openpose_setup
            cd openpose_setup
            OPENPOSE_DIR="."
        fi
    else
        print_status "Cloning OpenPose repository..."
        git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git .
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
        
        # Clean up any existing corrupted files
        print_status "Cleaning up any existing corrupted files..."
        for dep in "${deps[@]}"; do
            if [ -f "$dep" ] && ! unzip -t "$dep" >/dev/null 2>&1; then
                print_warning "Removing corrupted file: $dep"
                rm -f "$dep"
            fi
        done

        # Download dependencies
        deps=(
            "caffe3rdparty_16_2020_11_14.zip"
            "caffe_gpu_2018_05_27.zip"
            "opencv_450_v15_2020_11_18.zip"
        )

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
        # Check if CMake cache exists
        if [ -f "CMakeCache.txt" ]; then
            print_status "CMake cache found. You can run 'make' directly or reconfigure."
            print_status "To reconfigure, run: rm -rf build && ./openpose_gpu_setup.sh"
        fi
    else
        print_status "Removing existing build directory..."
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

# Detect CUDA architecture based on installed GPU
CUDA_ARCH=""
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1)
    if [[ "$GPU_NAME" == "8.9" ]]; then
        CUDA_ARCH="8.9"
    elif [[ "$GPU_NAME" == "8.6" ]]; then
        CUDA_ARCH="8.6"
    elif [[ "$GPU_NAME" == "8.0" ]]; then
        CUDA_ARCH="8.0"
    elif [[ "$GPU_NAME" == "7.5" ]]; then
        CUDA_ARCH="7.5"
    elif [[ "$GPU_NAME" == "7.0" ]]; then
        CUDA_ARCH="7.0"
    elif [[ "$GPU_NAME" == "6.1" ]]; then
        CUDA_ARCH="6.1"
    elif [[ "$GPU_NAME" == "6.0" ]]; then
        CUDA_ARCH="6.0"
    else
        # Default to common architectures
        CUDA_ARCH="6.0,6.1,7.0,7.5,8.0,8.6,8.9"
    fi
    print_success "Detected GPU compute capability: $GPU_NAME"
else
    # Default to common architectures if nvidia-smi not available
    CUDA_ARCH="6.0,6.1,7.0,7.5,8.0,8.6,8.9"
    print_warning "nvidia-smi not available, using default CUDA architectures"
fi

print_status "Using CUDA architecture: $CUDA_ARCH"

cmake .. \
    -DGPU_MODE=CUDA \
    -DUSE_CUDNN=ON \
    -DCUDA_ARCH_BIN=$CUDA_ARCH \
    -DCUDA_ARCH_PTX=$CUDA_ARCH \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DCUDNN_INCLUDE_DIR=/usr/local/cuda/include \
    -DCUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so \
    -DBUILD_CAFFE=ON \
    -DBUILD_PYTHON=ON \
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
        
        # Start the build
        make -j$(nproc)
        
        if [ $? -eq 0 ]; then
            print_success "OpenPose build completed successfully!"
            print_status "You can now test your installation:"
            print_status "cd bin && ./OpenPoseDemo --help"
        else
            print_error "Build failed. Check the output above for errors."
            print_status "You can try building again with: make -j$(nproc)"
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

# Navigate back to the main directory (should be /workspace/animatediff)
cd /workspace/animatediff

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

print_success "Complete OpenPose + ComfyUI setup script completed!"
print_status "All components are now configured for pose transfer workflows"

exit 0
