# Pose Transfer Workflow Usage Guide

## Overview
This setup allows you to transfer poses from one image to another using OpenPose skeleton detection and ComfyUI with ControlNet.

## Prerequisites
- OpenPose built and working with GPU support
- ComfyUI installed with custom nodes:
  - ComfyUI-AnimateDiff-Evolved
  - ComfyUI-VideoHelperSuite
- ControlNet model: `control_v11p_sd15_openpose.pth`

## Step-by-Step Usage

### 1. Generate Skeleton from Source Image
```bash
# Go to OpenPose directory
cd openpose

# Create input directory if it doesn't exist
mkdir -p input_images

# Copy your source image (the one with the pose you want to extract)
cp /path/to/your/pose_source.jpg input_images/

# Run OpenPose to extract skeleton
./build/examples/openpose/openpose.bin \
    --image_dir input_images/ \
    --write_json output/ \
    --display 0 \
    --render_pose 0

# This creates JSON files in output/ directory
```

### 2. Convert JSON to Skeleton Image
```bash
# Go back to main directory
cd /workspace/animatediff

# Convert the JSON to a skeleton image
python3 create_skeleton_from_json.py \
    openpose/output/pose_source_keypoints.json \
    input/skeleton.png
```

### 3. Prepare Input Images for ComfyUI
```bash
# Copy your target image (the people you want to re-pose)
cp /path/to/your/target_image.jpg input/input_image.jpg

# The skeleton image should already be at input/skeleton.png
```

### 4. Run ComfyUI Workflow
1. Start ComfyUI:
   ```bash
   python3 main.py
   ```

2. Open the browser at `http://localhost:8188`

3. Load the workflow:
   - Drag and drop `working_hug_workflow.json` into ComfyUI

4. Check that the input images are loaded correctly:
   - Node 4 should load `hug-img.jpg` (your target image)
   - Node 5 should load `perfect_hug_skeleton.png` (your skeleton)

5. Adjust parameters if needed:
   - ControlNet strength (default: 1.2)
   - KSampler steps (default: 25)
   - CFG scale (default: 8)

6. Click "Queue Prompt" to generate the pose transfer

### 5. Output
The result will be saved as `hug_result_00001.png` in ComfyUI's output directory.

## Troubleshooting

### Common Issues:
1. **"No people detected"**: Ensure your source image has clear, visible people
2. **"Prompt has no outputs"**: Make sure SaveImage node is connected properly
3. **Poor pose transfer**: Try adjusting ControlNet strength (0.8-1.5 range)
4. **Blurry results**: Increase KSampler steps or adjust CFG scale
5. **Wrong poses**: Verify the skeleton image shows the desired pose clearly

### File Locations:
- OpenPose: `/workspace/animatediff/openpose/`
- ComfyUI inputs: `/workspace/animatediff/input/`
- ComfyUI outputs: `/workspace/animatediff/output/`
- Working workflow: `/workspace/animatediff/working_hug_workflow.json`

## Files Included in This Setup:

### Core Scripts:
- `openpose_gpu_setup.sh` - Complete setup script for OpenPose + ComfyUI
- `create_skeleton_from_json.py` - Converts OpenPose JSON to skeleton images

### ComfyUI Workflows:
- `working_hug_workflow.json` - Main pose transfer workflow (img2img with ControlNet)

### Input Images:
- `input/hug-img.jpg` - Target image (people to re-pose)
- `input/perfect_hug_skeleton.png` - Skeleton reference image

### Custom Nodes:
- `custom_nodes/ComfyUI-AnimateDiff-Evolved/` - For animation and video generation
- `custom_nodes/ComfyUI-VideoHelperSuite/` - For video processing utilities

### OpenPose Build:
- `openpose/` - OpenPose source and build directory
- `openpose/build/` - Compiled OpenPose binaries
- `openpose/input_images/` - Input images for pose detection
- `openpose/output/` - JSON keypoint output from OpenPose

## Example Command Sequence:
```bash
# 1. Extract pose from source image
cd /workspace/animatediff/openpose
mkdir -p input_images
cp /path/to/source_image.jpg input_images/
./build/examples/openpose/openpose.bin --image_dir input_images/ --write_json output/ --display 0 --render_pose 0

# 2. Convert to skeleton
cd /workspace/animatediff
python3 create_skeleton_from_json.py openpose/output/source_image_keypoints.json input/skeleton.png

# 3. Prepare target image
cp /path/to/target_image.jpg input/input_image.jpg

# 4. Start ComfyUI
python3 main.py
```

Then use the workflow in the ComfyUI interface.

## Git Repository Setup
To save your work and push to your own repository:

```bash
# Initialize git if not already done
git init

# Add all files
git add .

# Commit changes
git commit -m "Complete OpenPose + ComfyUI pose transfer setup"

# Set up your remote repository
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push to your repository
git push -u origin main
```

## Project Structure
```
/workspace/animatediff/
├── openpose_gpu_setup.sh           # Main setup script
├── create_skeleton_from_json.py    # Skeleton conversion utility
├── working_hug_workflow.json       # ComfyUI pose transfer workflow
├── POSE_TRANSFER_USAGE.md          # This usage guide
├── input/                          # Input images directory
│   ├── hug-img.jpg                # Target image
│   └── perfect_hug_skeleton.png   # Skeleton reference
├── custom_nodes/                   # ComfyUI custom nodes
│   ├── ComfyUI-AnimateDiff-Evolved/
│   └── ComfyUI-VideoHelperSuite/
└── openpose/                       # OpenPose installation
    ├── build/                      # Compiled binaries
    ├── input_images/              # Pose detection input
    └── output/                    # JSON keypoint output
```
