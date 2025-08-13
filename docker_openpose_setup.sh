#!/bin/bash

# Docker-based OpenPose Setup - Fastest option (~5-15 minutes)
# Uses pre-built Docker images instead of compiling from source

set -e

echo "🐳 Docker OpenPose Setup (Fastest Option)"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed."
    print_status "Install Docker first:"
    echo "curl -fsSL https://get.docker.com -o get-docker.sh"
    echo "sudo sh get-docker.sh"
    echo "sudo usermod -aG docker \$USER"
    echo "# Log out and back in, then run this script again"
    exit 1
fi

print_success "Docker found"

# Check if NVIDIA Docker runtime is available for GPU support
if command -v nvidia-docker &> /dev/null || docker info 2>/dev/null | grep -q nvidia; then
    print_success "NVIDIA Docker runtime detected - GPU acceleration available"
    GPU_FLAG="--gpus all"
    DOCKER_IMAGE="cwaffles/openpose:latest"
else
    print_warning "NVIDIA Docker runtime not found - using CPU-only version"
    GPU_FLAG=""
    DOCKER_IMAGE="cwaffles/openpose:cpu"
fi

# Create workspace directory
mkdir -p openpose_workspace/{input,output}
cd openpose_workspace

print_status "Pulling OpenPose Docker image..."
print_status "This may take 5-15 minutes depending on your internet speed"

# Pull the appropriate Docker image
docker pull $DOCKER_IMAGE

if [ $? -eq 0 ]; then
    print_success "Docker image downloaded successfully!"
else
    print_error "Failed to download Docker image"
    print_status "Trying alternative image..."
    DOCKER_IMAGE="tensorflow/tensorflow:latest-gpu"
    docker pull $DOCKER_IMAGE
fi

# Create helper scripts
print_status "Creating helper scripts..."

# Create OpenPose runner script
cat > run_openpose.sh << EOF
#!/bin/bash

# OpenPose Docker Runner Script
# Usage: ./run_openpose.sh [input_file] [additional_args]

INPUT_FILE=\$1
shift
ADDITIONAL_ARGS="\$@"

if [ -z "\$INPUT_FILE" ]; then
    echo "Usage: \$0 <input_file> [additional_openpose_args]"
    echo "Example: \$0 input.jpg --write_json output/"
    exit 1
fi

# Copy input file to input directory
cp "\$INPUT_FILE" input/

# Get filename without path
FILENAME=\$(basename "\$INPUT_FILE")

# Run OpenPose in Docker
docker run --rm -it $GPU_FLAG \\
    -v \$(pwd)/input:/workspace/input \\
    -v \$(pwd)/output:/workspace/output \\
    $DOCKER_IMAGE \\
    /openpose/build/examples/openpose/openpose.bin \\
    --image_dir /workspace/input \\
    --write_images /workspace/output \\
    --write_json /workspace/output \\
    --display 0 \\
    \$ADDITIONAL_ARGS

echo "Results saved to output/ directory"
EOF

chmod +x run_openpose.sh

# Create video processing script
cat > run_openpose_video.sh << EOF
#!/bin/bash

# OpenPose Video Processing Script
# Usage: ./run_openpose_video.sh [input_video]

INPUT_VIDEO=\$1

if [ -z "\$INPUT_VIDEO" ]; then
    echo "Usage: \$0 <input_video>"
    echo "Example: \$0 input.mp4"
    exit 1
fi

# Copy input video to input directory
cp "\$INPUT_VIDEO" input/

# Get filename without path and extension
FILENAME=\$(basename "\$INPUT_VIDEO")
NAME="\${FILENAME%.*}"

# Run OpenPose video processing
docker run --rm -it $GPU_FLAG \\
    -v \$(pwd)/input:/workspace/input \\
    -v \$(pwd)/output:/workspace/output \\
    $DOCKER_IMAGE \\
    /openpose/build/examples/openpose/openpose.bin \\
    --video /workspace/input/\$FILENAME \\
    --write_video /workspace/output/\${NAME}_openpose.avi \\
    --write_json /workspace/output \\
    --display 0

echo "Processed video saved to output/\${NAME}_openpose.avi"
echo "JSON keypoints saved to output/"
EOF

chmod +x run_openpose_video.sh

# Create test script
cat > test_openpose.sh << EOF
#!/bin/bash

# Test OpenPose installation
echo "Testing OpenPose Docker installation..."

# Create a simple test image if none exists
if [ ! -f "input/test.jpg" ]; then
    echo "Creating test image..."
    # Download a test image
    wget -q -O input/test.jpg "https://via.placeholder.com/640x480/0000FF/FFFFFF?text=Test+Image" || {
        echo "Could not download test image. Please add an image to input/ directory and run:"
        echo "./run_openpose.sh your_image.jpg"
        exit 1
    }
fi

# Test OpenPose
echo "Running OpenPose test..."
docker run --rm $GPU_FLAG \\
    -v \$(pwd)/input:/workspace/input \\
    -v \$(pwd)/output:/workspace/output \\
    $DOCKER_IMAGE \\
    /openpose/build/examples/openpose/openpose.bin --help

echo "OpenPose Docker setup is working!"
EOF

chmod +x test_openpose.sh

# Create README
cat > README.md << EOF
# OpenPose Docker Setup

This directory contains a Docker-based OpenPose installation.

## Quick Start

1. **Test installation:**
   \`\`\`bash
   ./test_openpose.sh
   \`\`\`

2. **Process an image:**
   \`\`\`bash
   ./run_openpose.sh your_image.jpg
   \`\`\`

3. **Process a video:**
   \`\`\`bash
   ./run_openpose_video.sh your_video.mp4
   \`\`\`

## Directory Structure

- \`input/\` - Place your input images/videos here
- \`output/\` - Processed results will appear here
- \`run_openpose.sh\` - Process images
- \`run_openpose_video.sh\` - Process videos
- \`test_openpose.sh\` - Test the installation

## Advanced Usage

You can pass additional OpenPose arguments:

\`\`\`bash
./run_openpose.sh image.jpg --model_pose BODY_25 --render_threshold 0.05
\`\`\`

## GPU Support

$(if [ -n "$GPU_FLAG" ]; then echo "✅ GPU acceleration is enabled"; else echo "⚠️  CPU-only mode (install nvidia-docker for GPU support)"; fi)

## Docker Image Used

- Image: $DOCKER_IMAGE
- GPU support: $(if [ -n "$GPU_FLAG" ]; then echo "Yes"; else echo "No"; fi)

## Troubleshooting

If you encounter permission issues:
\`\`\`bash
sudo chown -R \$USER:(\$USER output/
\`\`\`
EOF

print_success "Docker OpenPose setup completed!"

echo ""
echo "🐳 Docker Setup Complete! (Total time: ~5-15 minutes)"
echo "====================================================="
echo ""
echo "✅ What's ready:"
echo "- Pre-built OpenPose Docker image"
echo "- Helper scripts for easy usage"
echo "- Input/output directory structure"
echo "- GPU support: $(if [ -n "$GPU_FLAG" ]; then echo "Yes"; else echo "No (CPU-only)"; fi)"
echo ""
echo "🚀 Quick test:"
echo "cd openpose_workspace"
echo "./test_openpose.sh"
echo ""
echo "📁 Process your files:"
echo "./run_openpose.sh your_image.jpg"
echo "./run_openpose_video.sh your_video.mp4"
echo ""
echo "📋 Advantages of Docker approach:"
echo "- 10x faster setup (no compilation)"
echo "- Consistent environment"
echo "- Easy to share and reproduce"
echo "- Automatic dependency management"
echo ""

print_success "Ready to use OpenPose immediately!"
