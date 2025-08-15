#!/usr/bin/env python3
"""
Clean Skeleton Hugging Generator
Creates clean skeleton frames without text, just pure stick figures like the reference image
"""

import cv2
import numpy as np
import json
import os
import math
from typing import List, Tuple, Dict

class CleanSkeletonHuggingGenerator:
    def __init__(self):
        """Initialize the clean skeleton hugging generator"""
        self.frame_width = 512
        self.frame_height = 512
        self.joint_radius = 4
        self.bone_thickness = 2
        
        # Colors for different people (matching reference image)
        self.person1_color = (255, 0, 255)    # Magenta (like reference)
        self.person2_color = (0, 255, 255)    # Yellow (like reference)
        
        # Joint colors (white outlined)
        self.joint_colors = {
            'head': (255, 255, 255),      # White
            'shoulder': (255, 255, 255),   # White
            'elbow': (255, 255, 255),      # White
            'wrist': (255, 255, 255),      # White
            'hip': (255, 255, 255),        # White
            'knee': (255, 255, 255),       # White
            'ankle': (255, 255, 255)       # White
        }
        
    def generate_clean_hugging_sequence(self, output_dir: str, num_frames: int = 16) -> bool:
        """
        Generate a complete clean hugging sequence without text
        
        Args:
            output_dir: Directory to save generated frames
            num_frames: Number of frames to generate (default: 16)
            
        Returns:
            True if successful, False otherwise
        """
        print(f"🎬 Generating clean skeleton hugging sequence with {num_frames} frames...")
        print("👥 Two people walking toward each other with alternating leg movement")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate motion keyframes
        print("🎭 Creating clean walking and hugging motion for TWO people...")
        
        for frame_idx in range(num_frames):
            # Calculate progress through the sequence
            progress = frame_idx / (num_frames - 1)
            
            # Generate poses for both people
            person1_pose = self._generate_person1_walking_pose(progress)
            person2_pose = self._generate_person2_walking_pose(progress)
            
            # Create clean skeleton image (no text)
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
            self._create_clean_skeleton_frame(person1_pose, person2_pose, frame_path)
            
            # Progress indicator (only in console, not on image)
            if frame_idx % 4 == 0:
                phase = self._get_motion_phase(progress)
                print(f"   Frame {frame_idx:2d}: {phase} ({progress:.1%})")
        
        # Create video from frames
        video_path = os.path.join(output_dir, "clean_hugging_sequence.mp4")
        self._create_video_from_frames(output_dir, video_path, fps=8)
        
        # Save metadata
        self._save_metadata(output_dir, num_frames)
        
        print(f"🎉 Successfully generated {num_frames} clean skeleton frames in {output_dir}")
        return True
    
    def _get_motion_phase(self, progress: float) -> str:
        """Get the current motion phase description"""
        if progress < 0.3:
            return "Two people walking toward each other"
        elif progress < 0.7:
            return "Approaching with arms extending"
        else:
            return "Embracing in warm hug"
    
    def _generate_person1_walking_pose(self, progress: float) -> Dict:
        """Generate pose for person 1 (left side) with proper walking motion"""
        # Base positions for person 1
        base_x = 0.25  # Start further left
        base_y = 0.5
        
        # Walking motion: move toward center
        walk_distance = 0.20 * progress  # Move more distance
        current_x = base_x + walk_distance
        
        # Arm motion based on progress
        if progress < 0.3:
            # Walking phase: subtle arm swing with alternating motion
            step_cycle = progress * 8  # More step cycles
            left_arm_swing = math.sin(step_cycle * math.pi) * 0.03
            right_arm_swing = math.sin(step_cycle * math.pi + math.pi) * 0.03
            
            left_arm_angle = 0.05 + left_arm_swing
            right_arm_angle = -0.05 + right_arm_swing
        elif progress < 0.7:
            # Approaching phase: arms extending forward
            arm_extend = (progress - 0.3) / 0.4
            left_arm_angle = 0.05 + arm_extend * 0.25
            right_arm_angle = -0.05 + arm_extend * 0.25
        else:
            # Hugging phase: arms wrapping around (like reference image)
            hug_progress = (progress - 0.7) / 0.3
            left_arm_angle = 0.30 + hug_progress * 0.20
            right_arm_angle = 0.10 + hug_progress * 0.30
        
        # Leg motion: alternating steps with realistic walking
        if progress < 0.3:
            step_cycle = progress * 8  # More realistic step cycles
            left_leg_forward = math.sin(step_cycle * math.pi) * 0.03
            right_leg_forward = math.sin(step_cycle * math.pi + math.pi) * 0.03
            
            # Add subtle up/down motion for walking
            left_leg_up = abs(math.sin(step_cycle * math.pi)) * 0.02
            right_leg_up = abs(math.sin(step_cycle * math.pi + math.pi)) * 0.02
        else:
            # Stop walking, prepare for hug
            left_leg_forward = 0.01
            right_leg_forward = -0.01
            left_leg_up = 0.0
            right_leg_up = 0.0
        
        # Torso lean toward center
        torso_lean = progress * 0.08
        
        # Head motion: slight forward lean
        head_lean = progress * 0.05
        
        return {
            'head': (current_x, base_y - 0.25 + head_lean),
            'left_shoulder': (current_x - 0.08, base_y - 0.15),
            'right_shoulder': (current_x + 0.08, base_y - 0.15),
            'left_elbow': (current_x - 0.12, base_y - 0.05 + left_arm_angle),
            'right_elbow': (current_x + 0.12, base_y - 0.05 + right_arm_angle),
            'left_wrist': (current_x - 0.16, base_y + 0.05 + left_arm_angle * 2),
            'right_wrist': (current_x + 0.16, base_y + 0.05 + right_arm_angle * 2),
            'left_hip': (current_x - 0.06, base_y + 0.05 + torso_lean),
            'right_hip': (current_x + 0.06, base_y + 0.05 + torso_lean),
            'left_knee': (current_x - 0.06, base_y + 0.25 + left_leg_forward),
            'right_knee': (current_x + 0.06, base_y + 0.25 + right_leg_forward),
            'left_ankle': (current_x - 0.06, base_y + 0.45 + left_leg_forward * 2 - left_leg_up),
            'right_ankle': (current_x + 0.06, base_y + 0.45 + right_leg_forward * 2 - right_leg_up)
        }
    
    def _generate_person2_walking_pose(self, progress: float) -> Dict:
        """Generate pose for person 2 (right side) with proper walking motion"""
        # Base positions for person 2 (mirrored)
        base_x = 0.75  # Start further right
        base_y = 0.5
        
        # Walking motion: move toward center
        walk_distance = 0.20 * progress  # Move more distance
        current_x = base_x - walk_distance
        
        # Arm motion (mirrored from person 1)
        if progress < 0.3:
            # Walking phase: subtle arm swing with alternating motion
            step_cycle = progress * 8 + math.pi  # Opposite phase
            left_arm_swing = math.sin(step_cycle * math.pi) * 0.03
            right_arm_swing = math.sin(step_cycle * math.pi + math.pi) * 0.03
            
            left_arm_angle = -0.05 + left_arm_swing
            right_arm_angle = 0.05 + right_arm_swing
        elif progress < 0.7:
            # Approaching phase: arms extending forward
            arm_extend = (progress - 0.3) / 0.4
            left_arm_angle = -0.05 + arm_extend * 0.25
            right_arm_angle = 0.05 + arm_extend * 0.25
        else:
            # Hugging phase: arms wrapping around (like reference image)
            hug_progress = (progress - 0.7) / 0.3
            left_arm_angle = -0.10 + hug_progress * 0.30
            right_arm_angle = -0.30 + hug_progress * 0.20
        
        # Leg motion: alternating steps (opposite phase from person 1)
        if progress < 0.3:
            step_cycle = progress * 8 + math.pi  # Opposite phase
            left_leg_forward = math.sin(step_cycle * math.pi) * 0.03
            right_leg_forward = math.sin(step_cycle * math.pi + math.pi) * 0.03
            
            # Add subtle up/down motion for walking
            left_leg_up = abs(math.sin(step_cycle * math.pi)) * 0.02
            right_leg_up = abs(math.sin(step_cycle * math.pi + math.pi)) * 0.02
        else:
            # Stop walking, prepare for hug
            left_leg_forward = -0.01
            right_leg_forward = 0.01
            left_leg_up = 0.0
            right_leg_up = 0.0
        
        # Torso lean toward center
        torso_lean = progress * 0.08
        
        # Head motion: slight forward lean
        head_lean = progress * 0.05
        
        return {
            'head': (current_x, base_y - 0.25 + head_lean),
            'left_shoulder': (current_x - 0.08, base_y - 0.15),
            'right_shoulder': (current_x + 0.08, base_y - 0.15),
            'left_elbow': (current_x - 0.12, base_y - 0.05 + left_arm_angle),
            'right_elbow': (current_x + 0.12, base_y - 0.05 + right_arm_angle),
            'left_wrist': (current_x - 0.16, base_y + 0.05 + left_arm_angle * 2),
            'right_wrist': (current_x + 0.16, base_y + 0.05 + right_arm_angle * 2),
            'left_hip': (current_x - 0.06, base_y + 0.05 + torso_lean),
            'right_hip': (current_x + 0.06, base_y + 0.05 + torso_lean),
            'left_knee': (current_x - 0.06, base_y + 0.25 + left_leg_forward),
            'right_knee': (current_x + 0.06, base_y + 0.25 + right_leg_forward),
            'left_ankle': (current_x - 0.06, base_y + 0.45 + left_leg_forward * 2 - left_leg_up),
            'right_ankle': (current_x + 0.06, base_y + 0.45 + right_leg_forward * 2 - right_leg_up)
        }
    
    def _create_clean_skeleton_frame(self, person1_pose: Dict, person2_pose: Dict, output_path: str):
        """Create a clean skeleton frame with TWO people - NO TEXT"""
        # Create black background
        img = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Draw person 1 (left side) - Magenta (like reference)
        self._draw_person(img, person1_pose, self.person1_color)
        
        # Draw person 2 (right side) - Yellow (like reference)
        self._draw_person(img, person2_pose, self.person2_color)
        
        # Save the clean image (no text, no progress bars, no labels)
        cv2.imwrite(output_path, img)
    
    def _draw_person(self, img: np.ndarray, pose: Dict, color: Tuple[int, int, int]):
        """Draw a complete person skeleton"""
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
                start_pos = self._normalize_position(pose[start_joint])
                end_pos = self._normalize_position(pose[end_joint])
                cv2.line(img, start_pos, end_pos, color, self.bone_thickness)
        
        # Draw joints with white outline (like reference image)
        for joint_name, position in pose.items():
            if position:
                pos = self._normalize_position(position)
                # White joint marker
                cv2.circle(img, pos, self.joint_radius, (255, 255, 255), -1)
                # White outline (already white, so just the circle)
    
    def _normalize_position(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert normalized coordinates (0-1) to pixel coordinates"""
        x = int(pos[0] * self.frame_width)
        y = int(pos[1] * self.frame_height)
        return (x, y)
    
    def _create_video_from_frames(self, frames_dir: str, output_video: str, fps: int = 8):
        """Create video from generated frames"""
        try:
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.png')])
            
            if not frame_files:
                print("❌ No frame files found for video creation")
                return False
            
            first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
            height, width, layers = first_frame.shape
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            for frame_file in frame_files:
                frame_path = os.path.join(frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                video_writer.write(frame)
            
            video_writer.release()
            print(f"🎥 Video created: {output_video}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating video: {e}")
            return False
    
    def _save_metadata(self, output_dir: str, num_frames: int):
        """Save metadata about the generated sequence"""
        try:
            metadata = {
                "frames_generated": num_frames,
                "frame_width": self.frame_width,
                "frame_height": self.frame_height,
                "people_count": 2,
                "motion_phases": [
                    "Two people walking toward each other",
                    "Approaching with arms extending", 
                    "Embracing in warm hug"
                ],
                "walking_features": [
                    "Alternating leg movement",
                    "Subtle forward arm motion",
                    "Natural step progression"
                ],
                "hugging_features": [
                    "Arms wrap around shoulders and backs",
                    "Heads gently inclined toward each other",
                    "Natural embrace motion"
                ],
                "joint_types": [
                    "head", "shoulder", "elbow", "wrist", 
                    "hip", "knee", "ankle"
                ],
                "style": "clean_stick_figure_no_text",
                "colors": {
                    "person1": "magenta",
                    "person2": "yellow",
                    "joints": "white"
                },
                "motion_description": "Two people taking small steps toward each other with alternating leg movement and subtle forward arm motion, progressing to warm hug",
                "video_created": "clean_hugging_sequence.mp4",
                "timestamp": str(np.datetime64('now')),
                "method": "clean_skeleton_no_text"
            }
            
            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"📊 Clean metadata saved to {output_dir}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save metadata: {e}")

def main():
    """Main function for command-line usage"""
    import argparse
    parser = argparse.ArgumentParser(description="Generate clean skeleton hugging sequence without text")
    parser.add_argument("output_dir", help="Output directory for generated frames and video")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames to generate (default: 16)")
    
    args = parser.parse_args()
    
    # Create generator and run
    generator = CleanSkeletonHuggingGenerator()
    
    success = generator.generate_clean_hugging_sequence(
        output_dir=args.output_dir,
        num_frames=args.frames
    )
    
    if success:
        print(f"\n🎉 Clean skeleton hugging sequence generation completed successfully!")
        print(f"📁 Check the output directory: {args.output_dir}")
        print(f"🎥 Video file: {os.path.join(args.output_dir, 'clean_hugging_sequence.mp4')}")
        print(f"📊 Frames generated: {args.frames}")
        print(f"👥 Two people: Walking → Approaching → Hugging")
        print(f"🚶 Walking features: Alternating leg movement, subtle arm motion")
        print(f"🤗 Hugging features: Arms around shoulders, heads inclined")
        print(f"✨ Style: Clean skeleton frames with NO TEXT (like reference image)")
    else:
        print(f"\n❌ Clean skeleton hugging sequence generation failed!")
        exit(1)

if __name__ == "__main__":
    main()
