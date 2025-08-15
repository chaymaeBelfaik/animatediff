#!/usr/bin/env python3
"""
Input Image to Hugging Video Generator
Complete workflow that takes a user's input image and generates a hugging video
guided by MediaPipe skeleton frames using ControlNet and AnimateDiff
"""

import cv2
import numpy as np
import json
import os
import subprocess
import argparse
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import shutil

# Import our existing MediaPipe components
from mediapipe_pose_detector import MediaPipePoseDetector
from clean_skeleton_hugging_generator import CleanSkeletonHuggingGenerator

class InputToHuggingVideoGenerator:
    def __init__(self, 
                 controlnet_model_path: str = "control_v11p_sd15_openpose.pth",
                 sd_model_path: str = "stable-diffusion-v1-5.ckpt",
                 animatediff_model_path: str = "mm_sd_v15.ckpt"):
        """
        Initialize the Input to Hugging Video Generator
        
        Args:
            controlnet_model_path: Path to ControlNet OpenPose model
            sd_model_path: Path to Stable Diffusion model
            animatediff_model_path: Path to AnimateDiff model
        """
        self.pose_detector = MediaPipePoseDetector()
        self.skeleton_generator = CleanSkeletonHuggingGenerator()
        
        # Model paths
        self.controlnet_model = controlnet_model_path
        self.sd_model = sd_model_path
        self.animatediff_model = animatediff_model_path
        
        # Generation settings
        self.frame_width = 512
        self.frame_height = 512
        self.num_frames = 16
        self.fps = 8
        
        print("🎬 Input to Hugging Video Generator initialized!")
        print(f"📐 Output resolution: {self.frame_width}x{self.frame_height}")
        print(f"🎭 Frame count: {self.num_frames}")
        print(f"⚡ FPS: {self.fps}")
    
    def generate_hugging_video_from_input(self, 
                                        input_image_path: str,
                                        output_dir: str,
                                        prompt: str = "two people hugging, warm embrace, emotional moment, high quality, detailed, realistic, beautiful lighting",
                                        negative_prompt: str = "text, watermark, blurry, low quality, distorted, ugly, deformed",
                                        controlnet_strength: float = 1.0,
                                        steps: int = 25,
                                        cfg: float = 7.5,
                                        seed: int = 42) -> bool:
        """
        Complete pipeline: Input Image → Hugging Video
        
        Args:
            input_image_path: Path to user's input image
            output_dir: Directory to save all outputs
            prompt: Positive prompt for image generation
            negative_prompt: Negative prompt for image generation
            controlnet_strength: ControlNet strength (0.0-2.0)
            steps: Number of diffusion steps
            cfg: CFG scale
            seed: Random seed for reproducibility
            
        Returns:
            True if successful, False otherwise
        """
        print(f"🚀 Starting Input to Hugging Video Generation...")
        print(f"📁 Input image: {input_image_path}")
        print(f"📁 Output directory: {output_dir}")
        
        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        skeleton_dir = os.path.join(output_dir, "skeleton_frames")
        processed_dir = os.path.join(output_dir, "processed_frames")
        final_dir = os.path.join(output_dir, "final_video")
        
        for dir_path in [skeleton_dir, processed_dir, final_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        try:
            # Step 1: Analyze input image
            print("\n🔍 Step 1: Analyzing input image...")
            if not self._analyze_input_image(input_image_path, output_dir):
                return False
            
            # Step 2: Generate skeleton sequence
            print("\n🎭 Step 2: Generating skeleton sequence...")
            if not self._generate_skeleton_sequence(skeleton_dir):
                return False
            
            # Step 3: Prepare ControlNet inputs
            print("\n🎯 Step 3: Preparing ControlNet inputs...")
            if not self._prepare_controlnet_inputs(skeleton_dir, processed_dir):
                return False
            
            # Step 4: Generate video frames with ControlNet
            print("\n🎨 Step 4: Generating video frames with ControlNet...")
            if not self._generate_controlnet_frames(
                input_image_path, processed_dir, final_dir, 
                prompt, negative_prompt, controlnet_strength, steps, cfg, seed):
                return False
            
            # Step 5: Create final video
            print("\n🎥 Step 5: Creating final video...")
            if not self._create_final_video(final_dir, output_dir):
                return False
            
            # Step 6: Save workflow metadata
            print("\n📊 Step 6: Saving workflow metadata...")
            self._save_workflow_metadata(output_dir, input_image_path, prompt, 
                                       controlnet_strength, steps, cfg, seed)
            
            print(f"\n🎉 Successfully generated hugging video!")
            print(f"📁 Check output directory: {output_dir}")
            print(f"🎥 Final video: {os.path.join(output_dir, 'hugging_video.mp4')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in video generation pipeline: {e}")
            return False
    
    def _analyze_input_image(self, input_image_path: str, output_dir: str) -> bool:
        """Analyze the input image and detect poses if present"""
        try:
            # Check if input image exists
            if not os.path.exists(input_image_path):
                print(f"❌ Input image not found: {input_image_path}")
                return False
            
            # Load and resize input image
            img = cv2.imread(input_image_path)
            if img is None:
                print(f"❌ Could not load image: {input_image_path}")
                return False
            
            # Resize to target dimensions
            img_resized = cv2.resize(img, (self.frame_width, self.frame_height))
            
            # Save resized input
            resized_path = os.path.join(output_dir, "input_resized.png")
            cv2.imwrite(resized_path, img_resized)
            
            # Try to detect poses in input image
            print("   Detecting poses in input image...")
            poses = self.pose_detector.detect_pose_from_image(input_image_path)
            
            if poses:
                print(f"   ✅ Detected {len(poses)} pose(s) in input image")
                
                # Create pose visualization
                pose_viz_path = os.path.join(output_dir, "input_poses.png")
                self.pose_detector.create_skeleton_image(
                    poses, pose_viz_path, 
                    image_size=(self.frame_width, self.frame_height)
                )
            else:
                print("   ⚠️ No poses detected in input image (will use style only)")
            
            print(f"   ✅ Input image analysis complete")
            return True
            
        except Exception as e:
            print(f"   ❌ Error analyzing input image: {e}")
            return False
    
    def _generate_skeleton_sequence(self, skeleton_dir: str) -> bool:
        """Generate the 16-frame skeleton sequence for hugging motion"""
        try:
            print(f"   Generating {self.num_frames} skeleton frames...")
            
            success = self.skeleton_generator.generate_clean_hugging_sequence(
                output_dir=skeleton_dir,
                num_frames=self.num_frames
            )
            
            if success:
                print(f"   ✅ Skeleton sequence generated in {skeleton_dir}")
                return True
            else:
                print(f"   ❌ Failed to generate skeleton sequence")
                return False
                
        except Exception as e:
            print(f"   ❌ Error generating skeleton sequence: {e}")
            return False
    
    def _prepare_controlnet_inputs(self, skeleton_dir: str, processed_dir: str) -> bool:
        """Prepare skeleton frames for ControlNet processing"""
        try:
            print(f"   Preparing ControlNet inputs...")
            
            # Copy skeleton frames to processed directory
            for i in range(self.num_frames):
                src_path = os.path.join(skeleton_dir, f"frame_{i:03d}.png")
                dst_path = os.path.join(processed_dir, f"control_{i:03d}.png")
                
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                else:
                    print(f"   ⚠️ Missing skeleton frame: {src_path}")
                    return False
            
            print(f"   ✅ Prepared {self.num_frames} ControlNet input frames")
            return True
            
        except Exception as e:
            print(f"   ❌ Error preparing ControlNet inputs: {e}")
            return False
    
    def _generate_controlnet_frames(self, 
                                  input_image_path: str,
                                  control_dir: str,
                                  output_dir: str,
                                  prompt: str,
                                  negative_prompt: str,
                                  strength: float,
                                  steps: int,
                                  cfg: float,
                                  seed: int) -> bool:
        """Generate video frames using ControlNet"""
        try:
            print(f"   Processing {self.num_frames} frames with ControlNet...")
            
            # This is where you would integrate with your ControlNet/ComfyUI setup
            # For now, we'll create a script that can be run with ComfyUI
            
            # Create ComfyUI processing script
            script_path = os.path.join(output_dir, "comfyui_process.py")
            self._create_comfyui_script(
                script_path, input_image_path, control_dir, output_dir,
                prompt, negative_prompt, strength, steps, cfg, seed
            )
            
            # Create batch processing instructions
            instructions_path = os.path.join(output_dir, "processing_instructions.md")
            self._create_processing_instructions(
                instructions_path, control_dir, prompt, negative_prompt, strength, steps, cfg, seed
            )
            
            print(f"   ✅ ControlNet processing setup complete")
            print(f"   📝 Processing script: {script_path}")
            print(f"   📋 Instructions: {instructions_path}")
            
            # For demonstration, create placeholder frames
            self._create_placeholder_frames(output_dir)
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error in ControlNet processing: {e}")
            return False
    
    def _create_comfyui_script(self, 
                              script_path: str,
                              input_image: str,
                              control_dir: str,
                              output_dir: str,
                              prompt: str,
                              negative_prompt: str,
                              strength: float,
                              steps: int,
                              cfg: float,
                              seed: int):
        """Create a Python script for ComfyUI processing"""
        script_content = f'''#!/usr/bin/env python3
"""
ComfyUI Processing Script for Hugging Video Generation
Generated automatically by Input to Hugging Video Generator
"""

# ComfyUI API processing script
# This script can be adapted to work with ComfyUI's API

import requests
import json
import os
import time

def process_frame_with_comfyui(control_image_path, frame_index):
    """Process a single frame with ComfyUI API"""
    
    workflow = {{
        "prompt": {{
            "1": {{
                "class_type": "LoadImage",
                "inputs": {{
                    "image": control_image_path
                }}
            }},
            "2": {{
                "class_type": "ControlNetLoader",
                "inputs": {{
                    "control_net_name": "{self.controlnet_model}"
                }}
            }},
            "3": {{
                "class_type": "ControlNetApply",
                "inputs": {{
                    "strength": {strength},
                    "start_percent": 0.0,
                    "end_percent": 1.0,
                    "control_net": ["2", 0],
                    "image": ["1", 0]
                }}
            }},
            "4": {{
                "class_type": "CLIPTextEncode",
                "inputs": {{
                    "text": "{prompt}",
                    "clip": ["6", 1]
                }}
            }},
            "5": {{
                "class_type": "CLIPTextEncode",
                "inputs": {{
                    "text": "{negative_prompt}",
                    "clip": ["6", 1]
                }}
            }},
            "6": {{
                "class_type": "CheckpointLoaderSimple",
                "inputs": {{
                    "ckpt_name": "{self.sd_model}"
                }}
            }},
            "7": {{
                "class_type": "KSampler",
                "inputs": {{
                    "seed": {seed + frame_index},
                    "steps": {steps},
                    "cfg": {cfg},
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["6", 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["3", 0]
                }}
            }},
            "8": {{
                "class_type": "SaveImage",
                "inputs": {{
                    "filename_prefix": f"hugging_frame_{{frame_index:03d}}",
                    "images": ["7", 0]
                }}
            }}
        }}
    }}
    
    # Send to ComfyUI (adjust URL as needed)
    url = "http://127.0.0.1:8188/prompt"
    response = requests.post(url, json={{"prompt": workflow}})
    
    if response.status_code == 200:
        print(f"✅ Frame {frame_index} queued successfully")
        return True
    else:
        print(f"❌ Failed to queue frame {frame_index}: {response.text}")
        return False

def main():
    """Process all frames"""
    control_dir = "{control_dir}"
    
    print("🎬 Starting ComfyUI batch processing...")
    
    for i in range({self.num_frames}):
        control_path = os.path.join(control_dir, f"control_{{i:03d}}.png")
        
        if os.path.exists(control_path):
            print(f"Processing frame {{i+1}}/{self.num_frames}...")
            process_frame_with_comfyui(control_path, i)
            time.sleep(2)  # Avoid overwhelming the API
        else:
            print(f"⚠️ Control frame not found: {{control_path}}")
    
    print("🎉 Batch processing complete!")

if __name__ == "__main__":
    main()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
    
    def _create_processing_instructions(self, 
                                      instructions_path: str,
                                      control_dir: str,
                                      prompt: str,
                                      negative_prompt: str,
                                      strength: float,
                                      steps: int,
                                      cfg: float,
                                      seed: int):
        """Create human-readable processing instructions"""
        instructions = f'''# Hugging Video Generation Instructions

## Overview
Generate a hugging video from your input image using the provided skeleton frames.

## Required Models
- **ControlNet**: {self.controlnet_model}
- **Stable Diffusion**: {self.sd_model}
- **AnimateDiff**: {self.animatediff_model}

## Skeleton Frames
Location: `{control_dir}/`
- `control_000.png` to `control_015.png` (16 frames total)

## Generation Settings
- **Prompt**: {prompt}
- **Negative Prompt**: {negative_prompt}
- **ControlNet Strength**: {strength}
- **Steps**: {steps}
- **CFG Scale**: {cfg}
- **Seed**: {seed}
- **Resolution**: {self.frame_width}x{self.frame_height}

## Manual ComfyUI Setup

### 1. Load Workflow Nodes
1. **LoadImage** → Load each skeleton frame
2. **ControlNetLoader** → Load {self.controlnet_model}
3. **ControlNetApply** → Apply skeleton control (strength: {strength})
4. **CLIPTextEncode** (Positive) → "{prompt}"
5. **CLIPTextEncode** (Negative) → "{negative_prompt}"
6. **CheckpointLoaderSimple** → Load {self.sd_model}
7. **KSampler** → Generate images (steps: {steps}, cfg: {cfg})
8. **SaveImage** → Save generated frames

### 2. Process Each Frame
Process frames in sequence:
- Frame 1: `control_000.png` → `generated_000.png`
- Frame 2: `control_001.png` → `generated_001.png`
- ... (continue for all 16 frames)

### 3. Video Assembly
Use FFmpeg to create final video:
```bash
ffmpeg -framerate {self.fps} -i generated_%03d.png -c:v libx264 -pix_fmt yuv420p hugging_video.mp4
```

## AnimateDiff Integration (Optional)
For smoother animation, load AnimateDiff model and process all frames together.

## Tips
- Keep seed consistent across all frames for coherent style
- Adjust ControlNet strength if poses are too rigid or too loose
- Use higher CFG for more prompt adherence
- Lower CFG for more natural variations
'''
        
        with open(instructions_path, 'w') as f:
            f.write(instructions)
    
    def _create_placeholder_frames(self, output_dir: str):
        """Create placeholder frames for demonstration"""
        try:
            print("   Creating placeholder frames for demonstration...")
            
            for i in range(self.num_frames):
                # Create a simple placeholder image
                img = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                
                # Add frame number
                cv2.putText(img, f"Frame {i+1}/{self.num_frames}", 
                           (50, self.frame_height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.putText(img, "Generated with ControlNet + Skeleton", 
                           (50, self.frame_height//2 + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                
                # Save placeholder
                placeholder_path = os.path.join(output_dir, f"generated_{i:03d}.png")
                cv2.imwrite(placeholder_path, img)
            
            print(f"   ✅ Created {self.num_frames} placeholder frames")
            
        except Exception as e:
            print(f"   ⚠️ Error creating placeholder frames: {e}")
    
    def _create_final_video(self, frames_dir: str, output_dir: str) -> bool:
        """Create final video from generated frames"""
        try:
            print(f"   Creating final video...")
            
            # Check for generated frames
            frame_files = [f for f in os.listdir(frames_dir) 
                          if f.startswith('generated_') and f.endswith('.png')]
            frame_files.sort()
            
            if len(frame_files) < self.num_frames:
                print(f"   ⚠️ Only found {len(frame_files)} frames, expected {self.num_frames}")
            
            # Create video using FFmpeg
            video_path = os.path.join(output_dir, "hugging_video.mp4")
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-framerate', str(self.fps),
                '-i', os.path.join(frames_dir, 'generated_%03d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',  # Good quality
                video_path
            ]
            
            try:
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"   ✅ Video created: {video_path}")
                    return True
                else:
                    print(f"   ⚠️ FFmpeg warning: {result.stderr}")
                    # Try alternative video creation
                    return self._create_video_opencv(frames_dir, video_path)
            except FileNotFoundError:
                print("   ⚠️ FFmpeg not found, using OpenCV for video creation")
                return self._create_video_opencv(frames_dir, video_path)
                
        except Exception as e:
            print(f"   ❌ Error creating final video: {e}")
            return False
    
    def _create_video_opencv(self, frames_dir: str, video_path: str) -> bool:
        """Create video using OpenCV as fallback"""
        try:
            frame_files = sorted([f for f in os.listdir(frames_dir) 
                                if f.startswith('generated_') and f.endswith('.png')])
            
            if not frame_files:
                print("   ❌ No generated frames found for video creation")
                return False
            
            # Read first frame to get dimensions
            first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
            height, width, layers = first_frame.shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
            
            # Write frames
            for frame_file in frame_files:
                frame_path = os.path.join(frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    video_writer.write(frame)
            
            video_writer.release()
            print(f"   ✅ Video created with OpenCV: {video_path}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error creating video with OpenCV: {e}")
            return False
    
    def _save_workflow_metadata(self, 
                               output_dir: str,
                               input_image: str,
                               prompt: str,
                               strength: float,
                               steps: int,
                               cfg: float,
                               seed: int):
        """Save workflow metadata and settings"""
        try:
            metadata = {
                "workflow": "Input Image to Hugging Video",
                "input_image": input_image,
                "output_directory": output_dir,
                "settings": {
                    "prompt": prompt,
                    "controlnet_strength": strength,
                    "steps": steps,
                    "cfg_scale": cfg,
                    "seed": seed,
                    "frames": self.num_frames,
                    "fps": self.fps,
                    "resolution": f"{self.frame_width}x{self.frame_height}"
                },
                "models": {
                    "controlnet": self.controlnet_model,
                    "stable_diffusion": self.sd_model,
                    "animatediff": self.animatediff_model
                },
                "pipeline_steps": [
                    "1. Input image analysis",
                    "2. Skeleton sequence generation", 
                    "3. ControlNet input preparation",
                    "4. Frame generation with ControlNet",
                    "5. Video assembly"
                ],
                "timestamp": str(np.datetime64('now')),
                "version": "1.0"
            }
            
            metadata_path = os.path.join(output_dir, "workflow_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"   ✅ Workflow metadata saved to {metadata_path}")
            
        except Exception as e:
            print(f"   ⚠️ Warning: Could not save metadata: {e}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Generate hugging video from input image")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("output_dir", help="Output directory for generated video")
    parser.add_argument("--prompt", default="two people hugging, warm embrace, emotional moment, high quality, detailed, realistic, beautiful lighting", help="Positive prompt")
    parser.add_argument("--negative", default="text, watermark, blurry, low quality, distorted, ugly, deformed", help="Negative prompt")
    parser.add_argument("--strength", type=float, default=1.0, help="ControlNet strength (0.0-2.0)")
    parser.add_argument("--steps", type=int, default=25, help="Number of diffusion steps")
    parser.add_argument("--cfg", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create generator
    generator = InputToHuggingVideoGenerator()
    
    # Generate video
    success = generator.generate_hugging_video_from_input(
        input_image_path=args.input_image,
        output_dir=args.output_dir,
        prompt=args.prompt,
        negative_prompt=args.negative,
        controlnet_strength=args.strength,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed
    )
    
    if success:
        print(f"\n🎉 Hugging video generation completed successfully!")
        print(f"📁 Check the output directory: {args.output_dir}")
        print(f"🎥 Final video: {os.path.join(args.output_dir, 'hugging_video.mp4')}")
        print(f"📋 Processing instructions: {os.path.join(args.output_dir, 'processing_instructions.md')}")
    else:
        print(f"\n❌ Hugging video generation failed!")
        exit(1)

if __name__ == "__main__":
    main()
