#!/bin/bash

# AnimateDiff Models Download Script
# This script downloads all necessary models for pose-guided animation in ComfyUI

set -e  # Exit on any error

echo "🚀 Starting AnimateDiff models download..."

# Create necessary directories
echo "📁 Creating model directories..."
mkdir -p /workspace/ComfyUI/models/{checkpoints,controlnet,animatediff_models,vae}

# Base directory for models
MODELS_DIR="/workspace/ComfyUI/models"

echo "⬇️  Downloading models..."

# 1. Stable Diffusion 1.5 Base Model (if not present)
if [ ! -f "$MODELS_DIR/checkpoints/v1-5-pruned-emaonly.safetensors" ]; then
    echo "📥 Downloading SD 1.5 base model..."
    wget -O "$MODELS_DIR/checkpoints/v1-5-pruned-emaonly.safetensors" \
        "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
else
    echo "✅ SD 1.5 base model already exists"
fi

# 2. AnimateDiff Motion Modules
echo "📥 Downloading AnimateDiff motion modules..."

# Less motion (subtle animation)
if [ ! -f "$MODELS_DIR/animatediff_models/motion_checkpoint_less_motion.ckpt" ]; then
    wget -O "$MODELS_DIR/animatediff_models/motion_checkpoint_less_motion.ckpt" \
        "https://huggingface.co/crishhh/animatediff_controlnet/resolve/main/motion_checkpoint_less_motion.ckpt"
else
    echo "✅ Less motion module already exists"
fi

# More motion (stronger animation)
if [ ! -f "$MODELS_DIR/animatediff_models/motion_checkpoint_more_motion.ckpt" ]; then
    wget -O "$MODELS_DIR/animatediff_models/motion_checkpoint_more_motion.ckpt" \
        "https://huggingface.co/crishhh/animatediff_controlnet/resolve/main/motion_checkpoint_more_motion.ckpt"
else
    echo "✅ More motion module already exists"
fi

# 3. ControlNet Models
echo "📥 Downloading ControlNet models..."

# OpenPose ControlNet (v1.1 - smaller and better)
if [ ! -f "$MODELS_DIR/controlnet/control_v11p_sd15_openpose.pth" ]; then
    wget -O "$MODELS_DIR/controlnet/control_v11p_sd15_openpose.pth" \
        "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth"
else
    echo "✅ OpenPose ControlNet v1.1 already exists"
fi

# Depth ControlNet (for body overlap/occlusion)
if [ ! -f "$MODELS_DIR/controlnet/control_sd15_depth.pth" ]; then
    echo "📥 Downloading Depth ControlNet..."
    wget -O "$MODELS_DIR/controlnet/control_sd15_depth.pth" \
        "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_depth.pth"
else
    echo "✅ Depth ControlNet already exists"
fi

# AnimateDiff ControlNet checkpoint (original repo)
if [ ! -f "$MODELS_DIR/controlnet/animatediff_controlnet.ckpt" ]; then
    wget -O "$MODELS_DIR/controlnet/animatediff_controlnet.ckpt" \
        "https://huggingface.co/crishhh/animatediff_controlnet/resolve/main/controlnet_checkpoint.ckpt"
else
    echo "✅ AnimateDiff ControlNet already exists"
fi

# 4. VAE (optional but recommended for better quality)
if [ ! -f "$MODELS_DIR/vae/vae-ft-mse-840000-ema-pruned.safetensors" ]; then
    echo "📥 Downloading VAE for better image quality..."
    wget -O "$MODELS_DIR/vae/vae-ft-mse-840000-ema-pruned.safetensors" \
        "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors"
else
    echo "✅ VAE already exists"
fi

echo ""
echo "🎉 Download complete! Models summary:"
echo "📁 Checkpoints: $(ls -1 $MODELS_DIR/checkpoints/ | wc -l) files"
echo "📁 ControlNet: $(ls -1 $MODELS_DIR/controlnet/ | wc -l) files"
echo "📁 AnimateDiff: $(ls -1 $MODELS_DIR/animatediff_models/ | wc -l) files"
echo "📁 VAE: $(ls -1 $MODELS_DIR/vae/ | wc -l) files"

echo ""
echo "💾 Total disk usage:"
du -sh $MODELS_DIR/checkpoints/ $MODELS_DIR/controlnet/ $MODELS_DIR/animatediff_models/ $MODELS_DIR/vae/ 2>/dev/null || true

echo ""
echo "✨ All models downloaded successfully!"
echo "🔄 Please restart ComfyUI to load the new models."
echo ""
echo "📝 Usage notes:"
echo "   • Use SD 1.5 base model: v1-5-pruned-emaonly.safetensors"
echo "   • AnimateDiff motion: motion_checkpoint_more_motion.ckpt (stronger) or less_motion (subtle)"
echo "   • OpenPose: control_v11p_sd15_openpose.pth"
echo "   • Depth: control_sd15_depth.pth (for hugging occlusion)"
echo "   • VAE: vae-ft-mse-840000-ema-pruned.safetensors (optional, for better quality)"