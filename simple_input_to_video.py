#!/usr/bin/env python3
"""
Simple Input to Hugging Video Generator
Uses existing skeleton frames from clean_skeleton_hug folder
"""

import cv2
import numpy as np
import json
import os
import shutil
from pathlib import Path
from mediapipe_pose_detector import MediaPipePoseDetector

class SimpleInputToVideoGenerator:
    def __init__(self):
        """Initialize the simple generator"""
        self.pose_detector = MediaPipePoseDetector()
        self.frame_width = 512
        self.frame_height = 512
        
    def generate_hugging_video_simple(self, 
                                    input_image_path: str,
                                    skeleton_frames_dir: str,
                                    output_dir: str,
                                    prompt: str = "two people hugging, warm embrace, emotional moment, high quality, detailed, realistic, beautiful lighting") -> bool:
        """
        Simple pipeline using existing skeleton frames
        
        Args:
            input_image_path: Path to user's input image
            skeleton_frames_dir: Path to existing skeleton frames (clean_skeleton_hug)
            output_dir: Directory to save outputs
            prompt: Prompt for generation
            
        Returns:
            True if successful
        """
        print(f"🎬 Simple Input to Hugging Video Generation")
        print(f"📁 Input image: {input_image_path}")
        print(f"🎭 Skeleton frames: {skeleton_frames_dir}")
        print(f"📁 Output directory: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Step 1: Analyze input image
            print(f"\n🔍 Step 1: Analyzing input image...")
            if not self._analyze_input_image(input_image_path, output_dir):
                return False
            
            # Step 2: Copy skeleton frames
            print(f"\n🎭 Step 2: Preparing skeleton frames...")
            if not self._prepare_skeleton_frames(skeleton_frames_dir, output_dir):
                return False
            
            # Step 3: Create ComfyUI instructions
            print(f"\n📋 Step 3: Creating processing instructions...")
            if not self._create_instructions(input_image_path, output_dir, prompt):
                return False
            
            # Step 4: Create demonstration frames
            print(f"\n🎨 Step 4: Creating demonstration output...")
            if not self._create_demo_output(output_dir, prompt):
                return False
            
            print(f"\n🎉 Simple workflow setup complete!")
            print(f"📁 Check output directory: {output_dir}")
            print(f"📋 Instructions: {output_dir}/INSTRUCTIONS.md")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in simple workflow: {e}")
            return False
    
    def _analyze_input_image(self, input_image_path: str, output_dir: str) -> bool:
        """Analyze the input image"""
        try:
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
            
            # Try to detect poses
            print("   Detecting poses in input image...")
            poses = self.pose_detector.detect_pose_from_image(input_image_path)
            
            if poses:
                print(f"   ✅ Detected {len(poses)} pose(s) in input image")
            else:
                print("   ⚠️ No poses detected in input image (will use style only)")
            
            # Always create 2-person pose visualization for hugging workflow
            print("   Creating 2-person pose visualization for hugging workflow...")
            pose_viz_path = os.path.join(output_dir, "input_poses.png")
            self._create_two_person_input_pose(pose_viz_path)
            
            print(f"   ✅ Input image analysis complete")
            return True
            
        except Exception as e:
            print(f"   ❌ Error analyzing input image: {e}")
            return False
    
    def _prepare_skeleton_frames(self, skeleton_dir: str, output_dir: str) -> bool:
        """Copy skeleton frames to output directory"""
        try:
            if not os.path.exists(skeleton_dir):
                print(f"❌ Skeleton directory not found: {skeleton_dir}")
                return False
            
            # Create skeleton frames directory in output
            output_skeleton_dir = os.path.join(output_dir, "skeleton_frames")
            os.makedirs(output_skeleton_dir, exist_ok=True)
            
            # Copy skeleton frames
            frame_count = 0
            for i in range(16):
                src_frame = os.path.join(skeleton_dir, f"frame_{i:03d}.png")
                dst_frame = os.path.join(output_skeleton_dir, f"control_{i:03d}.png")
                
                if os.path.exists(src_frame):
                    shutil.copy2(src_frame, dst_frame)
                    frame_count += 1
                else:
                    print(f"   ⚠️ Missing frame: {src_frame}")
            
            print(f"   ✅ Copied {frame_count} skeleton frames to {output_skeleton_dir}")
            
            # Also copy the video if it exists
            src_video = os.path.join(skeleton_dir, "clean_hugging_sequence.mp4")
            if os.path.exists(src_video):
                dst_video = os.path.join(output_dir, "skeleton_sequence.mp4")
                shutil.copy2(src_video, dst_video)
                print(f"   ✅ Copied skeleton video to {dst_video}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error preparing skeleton frames: {e}")
            return False
    
    def _create_instructions(self, input_image_path: str, output_dir: str, prompt: str) -> bool:
        """Create detailed processing instructions"""
        try:
            instructions = f'''# Hugging Video Generation Instructions

## Input Image
- **Original**: {input_image_path}
- **Resized**: input_resized.png (512x512)
- **2-Person Pose Reference**: input_poses.png (shows 2 people in initial standing position)

## Skeleton Frames
- **Location**: skeleton_frames/
- **Files**: control_000.png to control_015.png (16 frames)
- **Type**: Clean stick figure skeletons showing walking to hugging motion

## ComfyUI Manual Setup

### Required Models
- **ControlNet**: control_v11p_sd15_openpose.pth
- **Stable Diffusion**: stable-diffusion-v1-5.ckpt or any SD 1.5 model

### Node Setup
1. **LoadImage** → Load each skeleton frame (control_000.png to control_015.png)
2. **ControlNetLoader** → Load OpenPose ControlNet model
3. **ControlNetApply** → Apply skeleton control
   - Strength: 1.0-1.2 (adjust as needed)
   - Start: 0.0, End: 1.0
4. **CLIPTextEncode** (Positive) → "{prompt}"
5. **CLIPTextEncode** (Negative) → "text, watermark, blurry, low quality, distorted"
6. **CheckpointLoaderSimple** → Load your SD model
7. **KSampler** → Generate images
   - Steps: 25-30
   - CFG: 7.5-8.0
   - Sampler: euler_ancestral
   - Scheduler: normal
8. **SaveImage** → Save generated frames

### Processing Steps
1. Process each skeleton frame individually
2. Use the same prompt and settings for all frames
3. Keep seed consistent for style coherence
4. Save frames as: generated_000.png to generated_015.png

### Video Creation
After generating all frames, create video with FFmpeg:
```bash
ffmpeg -framerate 8 -i generated_%03d.png -c:v libx264 -pix_fmt yuv420p hugging_video.mp4
```

## Settings Recommendations
- **ControlNet Strength**: 1.0 (increase for stricter pose following)
- **Steps**: 25-30 (higher for better quality)
- **CFG Scale**: 7.5-8.0 (higher for stronger prompt adherence)
- **Resolution**: 512x512 (matches skeleton frames)

## Tips
- Use the input image as style reference
- Adjust ControlNet strength if poses are too rigid or loose
- Keep lighting and clothing consistent in prompt
- Consider using img2img with input image as base

## Troubleshooting
- **Poses don't match**: Increase ControlNet strength
- **Poor quality**: Increase steps and CFG scale
- **Inconsistent style**: Use same seed across all frames
- **Wrong poses**: Check skeleton frame order

---
Generated by Simple Input to Hugging Video Generator
'''
            
            instructions_path = os.path.join(output_dir, "INSTRUCTIONS.md")
            with open(instructions_path, 'w') as f:
                f.write(instructions)
            
            print(f"   ✅ Instructions saved to {instructions_path}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error creating instructions: {e}")
            return False
    
    def _create_demo_output(self, output_dir: str, prompt: str) -> bool:
        """Create demonstration output showing the workflow"""
        try:
            # Create demo frames directory
            demo_dir = os.path.join(output_dir, "demo_generated_frames")
            os.makedirs(demo_dir, exist_ok=True)
            
            # Create 16 demo frames
            for i in range(16):
                # Create demo frame
                img = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                
                # Add frame info
                cv2.putText(img, f"Generated Frame {i+1}/16", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.putText(img, "ControlNet + Skeleton Guided", 
                           (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 255, 128), 2)
                
                cv2.putText(img, f"Prompt: {prompt[:40]}...", 
                           (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Add progress indicator
                progress = i / 15
                bar_width = 400
                bar_height = 20
                cv2.rectangle(img, (50, 250), (50 + bar_width, 250 + bar_height), (100, 100, 100), -1)
                cv2.rectangle(img, (50, 250), (50 + int(bar_width * progress), 250 + bar_height), (0, 255, 0), -1)
                
                cv2.putText(img, f"Motion: {self._get_motion_phase(progress)}", 
                           (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Save demo frame
                demo_frame_path = os.path.join(demo_dir, f"demo_generated_{i:03d}.png")
                cv2.imwrite(demo_frame_path, img)
            
            print(f"   ✅ Created 16 demo frames in {demo_dir}")
            
            # Create demo video
            self._create_demo_video(demo_dir, output_dir)
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error creating demo output: {e}")
            return False
    
    def _get_motion_phase(self, progress: float) -> str:
        """Get motion phase description"""
        if progress < 0.3:
            return "Walking toward each other"
        elif progress < 0.7:
            return "Approaching, arms extending"
        else:
            return "Embracing in warm hug"
    
    def _create_demo_video(self, frames_dir: str, output_dir: str):
        """Create demo video from frames"""
        try:
            frame_files = sorted([f for f in os.listdir(frames_dir) 
                                if f.startswith('demo_generated_') and f.endswith('.png')])
            
            if not frame_files:
                return
            
            # Read first frame to get dimensions
            first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
            height, width, layers = first_frame.shape
            
            # Create video writer
            video_path = os.path.join(output_dir, "demo_hugging_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 8, (width, height))
            
            # Write frames
            for frame_file in frame_files:
                frame_path = os.path.join(frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    video_writer.write(frame)
            
            video_writer.release()
            print(f"   ✅ Demo video created: {video_path}")
            
        except Exception as e:
            print(f"   ⚠️ Could not create demo video: {e}")
    
    def _create_two_person_input_pose(self, output_path: str) -> bool:
        """Create a proper 2-person pose visualization for input reference"""
        try:
            # Create black background
            img = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            
            # Define initial standing poses for 2 people
            # Person 1 (left side) - standing upright
            person1_pose = {
                'head': (0.25, 0.15),
                'left_shoulder': (0.17, 0.25),
                'right_shoulder': (0.33, 0.25),
                'left_elbow': (0.13, 0.40),
                'right_elbow': (0.37, 0.40),
                'left_wrist': (0.10, 0.55),
                'right_wrist': (0.40, 0.55),
                'left_hip': (0.19, 0.55),
                'right_hip': (0.31, 0.55),
                'left_knee': (0.19, 0.75),
                'right_knee': (0.31, 0.75),
                'left_ankle': (0.19, 0.90),
                'right_ankle': (0.31, 0.90)
            }
            
            # Person 2 (right side) - standing upright, facing person 1
            person2_pose = {
                'head': (0.75, 0.15),
                'left_shoulder': (0.67, 0.25),
                'right_shoulder': (0.83, 0.25),
                'left_elbow': (0.63, 0.40),
                'right_elbow': (0.87, 0.40),
                'left_wrist': (0.60, 0.55),
                'right_wrist': (0.90, 0.55),
                'left_hip': (0.69, 0.55),
                'right_hip': (0.81, 0.55),
                'left_knee': (0.69, 0.75),
                'right_knee': (0.81, 0.75),
                'left_ankle': (0.69, 0.90),
                'right_ankle': (0.81, 0.90)
            }
            
            # Colors for different people
            person1_color = (255, 0, 255)  # Magenta
            person2_color = (0, 255, 255)  # Yellow
            joint_color = (255, 255, 255)  # White joints
            
            # Draw both people
            self._draw_person_pose(img, person1_pose, person1_color, joint_color)
            self._draw_person_pose(img, person2_pose, person2_color, joint_color)
            
            # Add labels
            cv2.putText(img, "Person 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, person1_color, 2)
            cv2.putText(img, "Person 2", (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, person2_color, 2)
            
            cv2.putText(img, "Initial Standing Pose", (150, self.frame_height - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # Save the image
            cv2.imwrite(output_path, img)
            print(f"   ✅ 2-person input pose created: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error creating 2-person input pose: {e}")
            return False
    
    def _draw_person_pose(self, img, pose, person_color, joint_color):
        """Draw a single person's pose on the image"""
        
        # Define bone connections
        connections = [
            # Head and torso
            ('head', 'left_shoulder'),
            ('head', 'right_shoulder'),
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            
            # Arms
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            
            # Legs
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle')
        ]
        
        # Draw bones
        for start_joint, end_joint in connections:
            if start_joint in pose and end_joint in pose:
                start_pos = self._normalize_pose_position(pose[start_joint])
                end_pos = self._normalize_pose_position(pose[end_joint])
                cv2.line(img, start_pos, end_pos, person_color, 2)
        
        # Draw joints
        for joint_name, position in pose.items():
            if position:
                pos = self._normalize_pose_position(position)
                cv2.circle(img, pos, 4, joint_color, -1)
    
    def _normalize_pose_position(self, pos):
        """Convert normalized coordinates (0-1) to pixel coordinates"""
        x = int(pos[0] * self.frame_width)
        y = int(pos[1] * self.frame_height)
        return (x, y)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Input to Hugging Video Generator")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("skeleton_frames", help="Path to skeleton frames directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--prompt", default="two people hugging, warm embrace, emotional moment, high quality, detailed, realistic, beautiful lighting", help="Generation prompt")
    
    args = parser.parse_args()
    
    # Create generator
    generator = SimpleInputToVideoGenerator()
    
    # Generate workflow
    success = generator.generate_hugging_video_simple(
        input_image_path=args.input_image,
        skeleton_frames_dir=args.skeleton_frames,
        output_dir=args.output_dir,
        prompt=args.prompt
    )
    
    if success:
        print(f"\n🎉 Simple workflow setup completed successfully!")
        print(f"📁 Check the output directory: {args.output_dir}")
        print(f"📋 Follow the instructions in: {args.output_dir}/INSTRUCTIONS.md")
    else:
        print(f"\n❌ Simple workflow setup failed!")

if __name__ == "__main__":
    main()
