#!/bin/bash

# Quick OpenPose Setup Script - Minimal installation for testing
# This version skips the full build and uses pre-built alternatives

set -e

echo "🚀 Quick OpenPose Setup for Testing..."
echo "====================================="

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

# Check if we're on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script is designed for Linux."
    exit 1
fi

# Install minimal dependencies only
print_status "Installing minimal dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y \
        python3-pip \
        python3-opencv \
        git \
        wget
elif command -v yum &> /dev/null; then
    sudo yum install -y \
        python3-pip \
        opencv-python3 \
        git \
        wget
fi

# Install MediaPipe as OpenPose alternative (much faster)
print_status "Installing MediaPipe as OpenPose alternative..."
pip3 install mediapipe opencv-python numpy

# Create a simple pose detection script
print_status "Creating pose detection wrapper..."
cat > pose_detector.py << 'EOF'
#!/usr/bin/env python3
"""
Quick pose detection using MediaPipe (OpenPose alternative)
Much faster installation and execution than full OpenPose
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import os

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def process_image(self, image_path, output_path=None):
        """Process a single image"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Pose detection saved to: {output_path}")
        
        return image, results

    def process_video(self, video_path, output_path=None):
        """Process a video file"""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            if output_path:
                out.write(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        if output_path:
            out.release()
            print(f"Video processing complete: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Quick pose detection using MediaPipe')
    parser.add_argument('--input', required=True, help='Input image or video file')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--display', action='store_true', help='Display result')
    
    args = parser.parse_args()
    
    detector = PoseDetector()
    
    # Check if input is image or video
    if args.input.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        result_image, results = detector.process_image(args.input, args.output)
        if args.display and result_image is not None:
            cv2.imshow('Pose Detection', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        detector.process_video(args.input, args.output)

if __name__ == "__main__":
    main()
EOF

chmod +x pose_detector.py

# Set up ComfyUI custom nodes (quick version)
print_status "Setting up ComfyUI custom nodes..."

if [ -d "ComfyUI" ] || [ -f "main.py" ]; then
    mkdir -p custom_nodes
    cd custom_nodes
    
    # Only install essential custom nodes without heavy dependencies
    if [ ! -d "ComfyUI-AnimateDiff-Evolved" ]; then
        print_status "Cloning AnimateDiff (without installing heavy dependencies)..."
        git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git
    fi
    
    if [ ! -d "ComfyUI-VideoHelperSuite" ]; then
        print_status "Cloning VideoHelper (lightweight setup)..."
        git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
    fi
    
    cd ..
fi

print_success "Quick setup completed!"
echo ""
echo "🎯 Quick Setup Complete! (Total time: ~5-10 minutes)"
echo "=================================================="
echo ""
echo "✅ What's installed:"
echo "- MediaPipe pose detection (OpenPose alternative)"
echo "- Basic ComfyUI custom nodes"
echo "- Python pose detection script"
echo ""
echo "🚀 Quick test:"
echo "python3 pose_detector.py --input your_image.jpg --output result.jpg --display"
echo ""
echo "📝 Notes:"
echo "- This uses MediaPipe instead of full OpenPose"
echo "- 10x faster installation and execution"
echo "- Good for testing and prototyping"
echo "- For production use, consider the full OpenPose build"
echo ""
print_success "You can now test pose detection immediately!"
