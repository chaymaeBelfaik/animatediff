#!/usr/bin/env python3
"""
Hugging Effect Generator using MediaPipe
Creates smooth hugging animations from static images of people
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from mediapipe_pose_detector import MediaPipePoseDetector

class HuggingEffectGenerator:
    def __init__(self):
        """Initialize the hugging effect generator"""
        self.pose_detector = MediaPipePoseDetector(static_mode=True, model_complexity=2)
        
    def generate_hugging_sequence(self, 
                                 source_image_path: str,
                                 target_hug_pose_path: str,
                                 output_dir: str,
                                 num_frames: int = 24,
                                 easing: str = "ease_in_out") -> bool:
        """
        Generate a complete hugging sequence
        
        Args:
            source_image_path: Path to source image with people standing
            target_hug_pose_path: Path to target hugging pose image
            output_dir: Directory to save generated frames
            num_frames: Number of frames to generate
            easing: Easing function for smooth animation
            
        Returns:
            True if successful, False otherwise
        """
        print(f"🎬 Generating hugging sequence with {num_frames} frames...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Detect poses from both images
        print("🔍 Detecting poses from source image...")
        source_poses = self.pose_detector.detect_pose_from_image(source_image_path)
        
        print("🔍 Detecting poses from target hugging pose...")
        target_poses = self.pose_detector.detect_pose_from_image(target_hug_pose_path)
        
        if not source_poses:
            print("❌ No poses detected in source image")
            return False
            
        if not target_poses:
            print("❌ No poses detected in target hugging pose")
            return False
        
        print(f"✅ Detected {len(source_poses)} people in source, {len(target_poses)} in target")
        
        # Generate smooth transition frames
        print("🎭 Generating smooth pose transitions...")
        success = self._generate_smooth_transitions(
            source_poses, target_poses, output_dir, num_frames, easing
        )
        
        if success:
            print(f"🎉 Successfully generated {num_frames} frames in {output_dir}")
            
            # Create a summary video
            video_path = os.path.join(output_dir, "hugging_sequence.mp4")
            self._create_video_from_frames(output_dir, video_path, fps=8)
            
            # Save pose data for further processing
            self._save_pose_sequence_data(source_poses, target_poses, output_dir)
            
        return success
    
    def _generate_smooth_transitions(self, 
                                   source_poses: List[dict],
                                   target_poses: List[dict],
                                   output_dir: str,
                                   num_frames: int,
                                   easing: str) -> bool:
        """
        Generate smooth transitions between poses
        """
        try:
            for frame_idx in range(num_frames):
                # Calculate interpolation factor with easing
                t = self._apply_easing(frame_idx / (num_frames - 1), easing)
                
                # Interpolate poses
                interpolated_poses = self._interpolate_poses_advanced(
                    source_poses, target_poses, t
                )
                
                # Create skeleton image for this frame
                frame_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
                self.pose_detector.create_skeleton_image(
                    interpolated_poses, frame_path, 
                    image_size=(512, 512),
                    show_landmarks=True,
                    show_connections=True
                )
                
                # Progress indicator
                if frame_idx % 5 == 0:
                    progress = (frame_idx + 1) / num_frames * 100
                    print(f"   Progress: {progress:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating transitions: {e}")
            return False
    
    def _interpolate_poses_advanced(self, 
                                   source_poses: List[dict],
                                   target_poses: List[dict],
                                   t: float) -> List[dict]:
        """
        Advanced pose interpolation with better handling of multiple people
        """
        interpolated_poses = []
        
        # Match people between source and target based on position similarity
        matched_pairs = self._match_people(source_poses, target_poses)
        
        for source_idx, target_idx in matched_pairs:
            if source_idx < len(source_poses) and target_idx < len(target_poses):
                source_pose = source_poses[source_idx]
                target_pose = target_poses[target_idx]
                
                # Interpolate individual pose
                interpolated_pose = self._interpolate_single_pose(source_pose, target_pose, t)
                interpolated_poses.append(interpolated_pose)
        
        return interpolated_poses
    
    def _match_people(self, source_poses: List[dict], target_poses: List[dict]) -> List[Tuple[int, int]]:
        """
        Match people between source and target images based on position similarity
        """
        matches = []
        
        # Simple matching based on center of mass
        for i, source_pose in enumerate(source_poses):
            source_center = self._calculate_pose_center(source_pose)
            best_match = -1
            best_distance = float('inf')
            
            for j, target_pose in enumerate(target_poses):
                target_center = self._calculate_pose_center(target_pose)
                distance = np.linalg.norm(np.array(source_center) - np.array(target_center))
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = j
            
            if best_match != -1:
                matches.append((i, best_match))
        
        return matches
    
    def _calculate_pose_center(self, pose: dict) -> Tuple[float, float]:
        """Calculate the center of mass of a pose"""
        landmarks = pose['landmarks']
        
        # Use key body points (shoulders, hips) for center calculation
        key_points = [11, 12, 23, 24]  # Shoulders and hips
        
        x_coords = []
        y_coords = []
        
        for point_idx in key_points:
            if point_idx < len(landmarks):
                landmark = landmarks[point_idx]
                if landmark['visibility'] > 0.5:
                    x_coords.append(landmark['x'])
                    y_coords.append(landmark['y'])
        
        if x_coords and y_coords:
            return (np.mean(x_coords), np.mean(y_coords))
        else:
            # Fallback to all visible landmarks
            visible_x = [lm['x'] for lm in landmarks if lm['visibility'] > 0.5]
            visible_y = [lm['y'] for lm in landmarks if lm['visibility'] > 0.5]
            
            if visible_x and visible_y:
                return (np.mean(visible_x), np.mean(visible_y))
            else:
                return (0.5, 0.5)  # Default center
    
    def _interpolate_single_pose(self, source_pose: dict, target_pose: dict, t: float) -> dict:
        """
        Interpolate a single pose between source and target
        """
        source_landmarks = source_pose['landmarks']
        target_landmarks = target_pose['landmarks']
        
        # Ensure both have the same number of landmarks
        min_landmarks = min(len(source_landmarks), len(target_landmarks))
        
        interpolated_landmarks = []
        for i in range(min_landmarks):
            source_lm = source_landmarks[i]
            target_lm = target_landmarks[i]
            
            # Interpolate coordinates
            x = source_lm['x'] * (1 - t) + target_lm['x'] * t
            y = source_lm['y'] * (1 - t) + target_lm['y'] * t
            z = source_lm['z'] * (1 - t) + target_lm['z'] * t
            
            # Interpolate visibility
            visibility = source_lm['visibility'] * (1 - t) + target_lm['visibility'] * t
            
            interpolated_landmarks.append({
                'x': x, 'y': y, 'z': z, 'visibility': visibility
            })
        
        return {
            'landmarks': interpolated_landmarks,
            'image_width': 512,
            'image_height': 512
        }
    
    def _apply_easing(self, t: float, easing: str) -> float:
        """
        Apply easing function for smooth animation
        """
        if easing == "linear":
            return t
        elif easing == "ease_in":
            return t * t
        elif easing == "ease_out":
            return 1 - (1 - t) * (1 - t)
        elif easing == "ease_in_out":
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - 2 * (1 - t) * (1 - t)
        elif easing == "bounce":
            if t < 1/2.75:
                return 7.5625 * t * t
            elif t < 2/2.75:
                t = t - 1.5/2.75
                return 7.5625 * t * t + 0.75
            elif t < 2.5/2.75:
                t = t - 2.25/2.75
                return 7.5625 * t * t + 0.9375
            else:
                t = t - 2.625/2.75
                return 7.5625 * t * t + 0.984375
        else:
            return t
    
    def _create_video_from_frames(self, frames_dir: str, output_video: str, fps: int = 8):
        """
        Create a video from generated frames
        """
        try:
            # Get all frame files
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.png')])
            
            if not frame_files:
                print("❌ No frame files found for video creation")
                return False
            
            # Read first frame to get dimensions
            first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
            height, width, layers = first_frame.shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            # Add frames to video
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
    
    def _save_pose_sequence_data(self, source_poses: List[dict], target_poses: List[dict], output_dir: str):
        """
        Save pose sequence data for further processing
        """
        try:
            # Save source poses
            source_path = os.path.join(output_dir, "source_poses.json")
            with open(source_path, 'w') as f:
                json.dump(source_poses, f, indent=2)
            
            # Save target poses
            target_path = os.path.join(output_dir, "target_poses.json")
            with open(target_path, 'w') as f:
                json.dump(target_poses, f, indent=2)
            
            # Create metadata file
            metadata = {
                "source_image": "source_poses.json",
                "target_image": "target_poses.json",
                "frames_generated": len([f for f in os.listdir(output_dir) if f.startswith('frame_')]),
                "video_created": "hugging_sequence.mp4",
                "timestamp": str(np.datetime64('now'))
            }
            
            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"📊 Pose data saved to {output_dir}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save pose data: {e}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Generate hugging effects using MediaPipe")
    parser.add_argument("source_image", help="Path to source image with people standing")
    parser.add_argument("target_hug_pose", help="Path to target hugging pose image")
    parser.add_argument("output_dir", help="Output directory for generated frames and video")
    parser.add_argument("--frames", type=int, default=24, help="Number of frames to generate (default: 24)")
    parser.add_argument("--easing", default="ease_in_out", 
                       choices=["linear", "ease_in", "ease_out", "ease_in_out", "bounce"],
                       help="Easing function for animation (default: ease_in_out)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.source_image):
        print(f"❌ Source image not found: {args.source_image}")
        sys.exit(1)
    
    if not os.path.exists(args.target_hug_pose):
        print(f"❌ Target hugging pose not found: {args.target_hug_pose}")
        sys.exit(1)
    
    # Create generator and run
    generator = HuggingEffectGenerator()
    
    success = generator.generate_hugging_sequence(
        source_image_path=args.source_image,
        target_hug_pose_path=args.target_hug_pose,
        output_dir=args.output_dir,
        num_frames=args.frames,
        easing=args.easing
    )
    
    if success:
        print("\n🎉 Hugging effect generation completed successfully!")
        print(f"📁 Check the output directory: {args.output_dir}")
        print(f"🎥 Video file: {os.path.join(args.output_dir, 'hugging_sequence.mp4')}")
    else:
        print("\n❌ Hugging effect generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
