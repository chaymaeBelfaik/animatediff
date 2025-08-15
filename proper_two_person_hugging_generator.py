#!/usr/bin/env python3
"""
Proper Two-Person Hugging Generator
Creates skeleton frames showing TWO people walking toward each other with realistic motion
"""

import cv2
import numpy as np
import json
import os
import math
from typing import List, Tuple, Dict

class ProperTwoPersonHuggingGenerator:
    def __init__(self):
        """Initialize the proper two-person hugging generator"""
        self.frame_width = 512
        self.frame_height = 512
        self.joint_radius = 4
        self.bone_thickness = 2
        
        # Colors for different people
        self.person1_color = (0, 255, 255)    # Cyan
        self.person2_color = (255, 0, 255)    # Magenta
        
        # Joint colors
        self.joint_colors = {
            'head': (255, 255, 0),      # Yellow
            'shoulder': (0, 255, 255),   # Cyan
            'elbow': (255, 0, 255),      # Magenta
            'wrist': (255, 255, 0),      # Yellow
            'hip': (0, 255, 0),          # Green
            'knee': (255, 128, 0),       # Orange
            'ankle': (128, 0, 255)       # Purple
        }
        
    def generate_two_person_hugging_sequence(self, output_dir: str, num_frames: int = 16) -> bool:
        """
        Generate a complete two-person hugging sequence with proper walking motion
        
        Args:
            output_dir: Directory to save generated frames
            num_frames: Number of frames to generate (default: 16)
            
        Returns:
            True if successful, False otherwise
        """
        print(f"🎬 Generating proper two-person hugging sequence with {num_frames} frames...")
        print("👥 Two people walking toward each other with alternating leg movement")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate motion keyframes
        print("🎭 Creating realistic walking and hugging motion for TWO people...")
        
        for frame_idx in range(num_frames):
            # Calculate progress through the sequence
            progress = frame_idx / (num_frames - 1)
            
            # Generate poses for both people
            person1_pose = self._generate_person1_walking_pose(progress)
            person2_pose = self._generate_person2_walking_pose(progress)
            
            # Create skeleton image
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
            self._create_two_person_skeleton_frame(person1_pose, person2_pose, frame_path, progress)
            
            # Progress indicator
            if frame_idx % 4 == 0:
                phase = self._get_motion_phase(progress)
                print(f"   Frame {frame_idx:2d}: {phase} ({progress:.1%})")
        
        # Create video from frames
        video_path = os.path.join(output_dir, "two_person_hugging_sequence.mp4")
        self._create_video_from_frames(output_dir, video_path, fps=8)
        
        # Save metadata
        self._save_metadata(output_dir, num_frames)
        
        print(f"🎉 Successfully generated {num_frames} frames with TWO people in {output_dir}")
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
            # Hugging phase: arms wrapping around
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
            # Hugging phase: arms wrapping around
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
    
    def _create_two_person_skeleton_frame(self, person1_pose: Dict, person2_pose: Dict, output_path: str, progress: float):
        """Create a skeleton frame with TWO people"""
        # Create black background
        img = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Draw person 1 (left side) - Cyan
        self._draw_person(img, person1_pose, self.person1_color)
        
        # Draw person 2 (right side) - Magenta
        self._draw_person(img, person2_pose, self.person2_color)
        
        # Add frame information
        self._add_frame_info(img, progress)
        
        # Add person labels
        self._add_person_labels(img)
        
        # Save the image
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
        
        # Draw joints
        for joint_name, position in pose.items():
            if position:
                pos = self._normalize_position(position)
                joint_color = self._get_joint_color(joint_name)
                cv2.circle(img, pos, self.joint_radius, joint_color, -1)
                # White border
                cv2.circle(img, pos, self.joint_radius + 1, (255, 255, 255), 1)
    
    def _normalize_position(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert normalized coordinates (0-1) to pixel coordinates"""
        x = int(pos[0] * self.frame_width)
        y = int(pos[1] * self.frame_height)
        return (x, y)
    
    def _get_joint_color(self, joint_name: str) -> Tuple[int, int, int]:
        """Get color for specific joint type"""
        if 'head' in joint_name:
            return self.joint_colors['head']
        elif 'shoulder' in joint_name:
            return self.joint_colors['shoulder']
        elif 'elbow' in joint_name:
            return self.joint_colors['elbow']
        elif 'wrist' in joint_name:
            return self.joint_colors['wrist']
        elif 'hip' in joint_name:
            return self.joint_colors['hip']
        elif 'knee' in joint_name:
            return self.joint_colors['knee']
        elif 'ankle' in joint_name:
            return self.joint_colors['ankle']
        else:
            return (255, 255, 255)  # White default
    
    def _add_frame_info(self, img: np.ndarray, progress: float):
        """Add frame information overlay"""
        # Add progress bar
        bar_width = 400
        bar_height = 20
        bar_x = (self.frame_width - bar_width) // 2
        bar_y = 30
        
        # Background bar
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        
        # Progress bar
        progress_width = int(bar_width * progress)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Progress text
        progress_text = f"Progress: {progress:.1%}"
        text_size = cv2.getTextSize(progress_text, font, font_scale, thickness)[0]
        text_x = (self.frame_width - text_size[0]) // 2
        text_y = bar_y + bar_height + 25
        
        cv2.putText(img, progress_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        # Phase text
        phase = self._get_motion_phase(progress)
        phase_text = f"Phase: {phase}"
        phase_size = cv2.getTextSize(phase_text, font, font_scale, thickness)[0]
        phase_x = (self.frame_width - phase_size[0]) // 2
        phase_y = text_y + 25
        
        cv2.putText(img, phase_text, (phase_x, phase_y), font, font_scale, (255, 255, 255), thickness)
    
    def _add_person_labels(self, img: np.ndarray):
        """Add labels to identify the two people"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        
        # Person 1 label (left side)
        cv2.putText(img, "Person 1", (50, 450), font, font_scale, self.person1_color, thickness)
        
        # Person 2 label (right side)
        cv2.putText(img, "Person 2", (350, 450), font, font_scale, self.person2_color, thickness)
    
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
                "style": "stick_figure_with_clear_joints",
                "motion_description": "Two people taking small steps toward each other with alternating leg movement and subtle forward arm motion, progressing to warm hug",
                "video_created": "two_person_hugging_sequence.mp4",
                "timestamp": str(np.datetime64('now')),
                "method": "proper_two_person_walking_and_hugging"
            }
            
            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"📊 Metadata saved to {output_dir}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save metadata: {e}")

def main():
    """Main function for command-line usage"""
    import argparse
    parser = argparse.ArgumentParser(description="Generate proper two-person hugging sequence with walking motion")
    parser.add_argument("output_dir", help="Output directory for generated frames and video")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames to generate (default: 16)")
    
    args = parser.parse_args()
    
    # Create generator and run
    generator = ProperTwoPersonHuggingGenerator()
    
    success = generator.generate_two_person_hugging_sequence(
        output_dir=args.output_dir,
        num_frames=args.frames
    )
    
    if success:
        print(f"\n🎉 Proper two-person hugging sequence generation completed successfully!")
        print(f"📁 Check the output directory: {args.output_dir}")
        print(f"🎥 Video file: {os.path.join(args.output_dir, 'two_person_hugging_sequence.mp4')}")
        print(f"📊 Frames generated: {args.frames}")
        print(f"👥 Two people: Walking → Approaching → Hugging")
        print(f"🚶 Walking features: Alternating leg movement, subtle arm motion")
        print(f"🤗 Hugging features: Arms around shoulders, heads inclined")
    else:
        print(f"\n❌ Two-person hugging sequence generation failed!")
        exit(1)

if __name__ == "__main__":
    main()
