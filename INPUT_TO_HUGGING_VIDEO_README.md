# Input Image to Hugging Video Generator

Complete workflow that transforms any input image into a realistic hugging video using MediaPipe skeleton generation, ControlNet pose guidance, and AnimateDiff.

## 🎬 Overview

This pipeline takes a user's input image and generates a smooth video showing two people walking toward each other and embracing in a warm hug. The motion is guided by clean skeleton frames generated with MediaPipe and transformed into realistic video using ControlNet and Stable Diffusion.

## ✨ Features

- **📷 Input Image Processing**: Analyzes and preprocesses any input image
- **🎭 Skeleton Generation**: Creates 16 clean skeleton frames for walking and hugging motion
- **🎯 ControlNet Integration**: Uses OpenPose ControlNet for precise pose guidance
- **🎨 Realistic Generation**: Generates high-quality realistic frames
- **🎥 Video Assembly**: Combines frames into smooth video output
- **⚙️ Customizable Settings**: Full control over generation parameters

## 🚀 Quick Start

### Basic Usage

```bash
# Generate hugging video from input image
python input_to_hugging_video_generator.py input/my_image.jpg output/my_hugging_video/

# Run demo with sample settings
python demo_input_to_hugging_video.py
```

### Advanced Usage

```bash
# Custom prompt and settings
python input_to_hugging_video_generator.py input/photo.jpg output/custom/ \
    --prompt "two people in formal wear embracing at sunset, romantic, cinematic" \
    --strength 1.2 \
    --steps 30 \
    --cfg 8.0 \
    --seed 123
```

## 📋 Workflow Steps

### 1. Input Image Analysis 📷
- Loads and resizes input image to 512x512
- Detects existing poses (if any) using MediaPipe
- Prepares image for style transfer and generation

### 2. Skeleton Generation 🎭
- Generates 16 clean skeleton frames showing:
  - Two people walking toward each other
  - Alternating leg movement and arm motion
  - Progressive approach and final embrace
- Uses MediaPipe-based motion generation

### 3. ControlNet Preparation 🎯
- Converts skeleton frames to ControlNet format
- Prepares pose guidance inputs
- Sets up batch processing structure

### 4. Frame Generation 🎨
- Applies ControlNet OpenPose guidance
- Generates realistic frames using Stable Diffusion
- Maintains consistent style across all frames
- Uses input image for style reference

### 5. Video Assembly 🎥
- Combines generated frames into smooth video
- Applies FFmpeg encoding for high quality
- Creates final MP4 output

## 🛠️ Requirements

### Python Dependencies
```bash
pip install opencv-python numpy mediapipe
```

### AI Models Required
- **Stable Diffusion Model**: `stable-diffusion-v1-5.ckpt`
- **ControlNet OpenPose**: `control_v11p_sd15_openpose.pth`
- **AnimateDiff (Optional)**: `mm_sd_v15.ckpt`

### External Tools
- **FFmpeg**: For video encoding (optional, OpenCV fallback available)
- **ComfyUI**: For ControlNet processing

## 📁 File Structure

```
animatediff/
├── input_to_hugging_video_generator.py    # Main workflow script
├── demo_input_to_hugging_video.py         # Demo script
├── mediapipe_pose_detector.py             # MediaPipe integration
├── clean_skeleton_hugging_generator.py    # Skeleton generation
├── input/                                  # Input images
│   └── hug-img.jpg
└── output/                                 # Generated outputs
    └── input_to_hugging_video_demo/
        ├── hugging_video.mp4              # Final video
        ├── skeleton_frames/               # Generated skeletons
        ├── processed_frames/              # ControlNet inputs
        ├── comfyui_process.py            # ComfyUI script
        ├── processing_instructions.md     # Manual setup guide
        └── workflow_metadata.json        # Generation settings
```

## ⚙️ Configuration

### Generation Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | "two people hugging..." | Positive prompt for generation |
| `negative_prompt` | "text, watermark..." | Negative prompt |
| `controlnet_strength` | 1.0 | ControlNet influence (0.0-2.0) |
| `steps` | 25 | Diffusion steps |
| `cfg` | 7.5 | CFG scale |
| `seed` | 42 | Random seed |

### Model Paths
```python
generator = InputToHuggingVideoGenerator(
    controlnet_model_path="control_v11p_sd15_openpose.pth",
    sd_model_path="stable-diffusion-v1-5.ckpt",
    animatediff_model_path="mm_sd_v15.ckpt"
)
```

## 🎯 ComfyUI Integration

The workflow generates ready-to-use ComfyUI scripts and instructions:

### Generated Files
- `comfyui_process.py`: Python script for API processing
- `processing_instructions.md`: Manual setup guide

### Manual ComfyUI Setup
1. Load the skeleton frames as ControlNet inputs
2. Set up OpenPose ControlNet with specified strength
3. Use provided prompts and settings
4. Process all 16 frames in sequence
5. Export frames for video assembly

## 🔧 Troubleshooting

### Common Issues

**No poses detected in input image**
- This is normal - the workflow uses input for style only
- Skeleton motion is generated independently

**Missing ControlNet model**
- Download `control_v11p_sd15_openpose.pth`
- Place in ComfyUI models/controlnet/ directory

**FFmpeg not found**
- Install FFmpeg or use OpenCV fallback
- Video will be created with alternative method

**ComfyUI connection failed**
- Check ComfyUI is running on localhost:8188
- Use manual processing instructions instead

## 📊 Performance Tips

### Speed Optimization
- Use lower steps (15-20) for faster generation
- Reduce ControlNet strength for less rigid poses
- Process fewer frames for shorter videos

### Quality Enhancement
- Increase steps (30-40) for better quality
- Higher CFG (8-10) for stronger prompt adherence
- Use AnimateDiff for smoother motion

## 🎨 Creative Examples

### Romantic Scene
```bash
python input_to_hugging_video_generator.py romantic.jpg output/romantic/ \
    --prompt "couple embracing at sunset, romantic atmosphere, golden hour lighting, cinematic" \
    --strength 1.0 --steps 30 --cfg 8.0
```

### Family Reunion
```bash
python input_to_hugging_video_generator.py family.jpg output/family/ \
    --prompt "family members hugging, warm reunion, emotional moment, natural lighting" \
    --strength 1.2 --steps 25 --cfg 7.5
```

### Professional Portrait
```bash
python input_to_hugging_video_generator.py business.jpg output/business/ \
    --prompt "business colleagues in formal wear, professional embrace, office setting" \
    --strength 0.8 --steps 20 --cfg 7.0
```

## 🔄 Workflow Variations

### Different Motion Styles
Modify the skeleton generator for different embrace styles:
- Quick hug vs. slow embrace
- Side hug vs. front hug
- Different walking speeds
- Various approach distances

### Alternative ControlNets
Use different ControlNet models:
- Depth for 3D guidance
- Canny for edge control
- Lineart for sketch-like input

## 📈 Future Enhancements

- **Multi-person support**: Generate videos with more than 2 people
- **Custom motions**: User-defined motion sequences
- **Style transfer**: More sophisticated style matching
- **Real-time processing**: Faster generation pipeline
- **Mobile support**: iOS/Android app integration

## 🤝 Contributing

This workflow combines several AI technologies:
- **MediaPipe**: Pose detection and skeleton generation
- **ControlNet**: Pose-guided image generation
- **Stable Diffusion**: High-quality image synthesis
- **AnimateDiff**: Smooth video animation

Contributions welcome for:
- New motion sequences
- Better ControlNet integration
- Performance optimizations
- Mobile deployment

## 📄 License

This project integrates multiple open-source components. Please check individual licenses for:
- MediaPipe (Apache 2.0)
- Stable Diffusion (CreativeML OpenRAIL-M)
- ControlNet (Apache 2.0)

---

**🎬 Transform any image into a beautiful hugging video with AI-powered motion synthesis!**
