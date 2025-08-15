#!/usr/bin/env python3
"""
Video-Based Hugging Effect Generator using MediaPipe
Creates smooth hugging animations from static images to video targets
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from mediapipe_pose_detector import MediaPipePoseDetector

class VideoHuggingEffectGenerator:
    def __init__(self):
        """Initialize the video-based hugging effect generator"""
        self.pose_detector = MediaPipePoseDetector(static_mode=False, model_complexity=2)
        
    def extract_poses_from_video(self, video_path: str, max_frames: int = 30) -> List[Dict]:
        """
        Extract poses from a video file
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of pose data for each frame
        """
        print(f"🎬 Extracting poses from video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        print(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
        
        # Calculate frame sampling interval
        if total_frames <= max_frames:
            frame_interval = 1
        else:
            frame_interval = total_frames // max_frames
        
        poses_per_frame = []
        frame_count = 0
        extracted_count = 0
        
        while extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect poses in this frame
                results = self.pose_detector.pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    # Convert landmarks to our format
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        })
                    
                    poses_per_frame.append({
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps,
                        'landmarks': landmarks,
                        'image_width': frame.shape[1],
                        'image_height': frame.shape[0]
                    })
                    
                    extracted_count += 1
                    print(f"   Extracted pose from frame {frame_count} ({extracted_count}/{max_frames})")
                
            frame_count += 1
            
            if frame_count >= total_frames:
                break
        
        cap.release()
        print(f"✅ Extracted {len(poses_per_frame)} poses from video")
        return poses_per_frame
    
    def generate_video_hugging_sequence(self, 
                                      source_image_path: str,
                                      target_video_path: str,
                                      output_dir: str,
                                      num_frames: int = 48,
                                      easing: str = "ease_in_out",
                                      video_sampling: int = 30) -> bool:
        """
        Generate hugging sequence from image to video target
        
        Args:
            source_image_path: Path to source image with people standing
            target_video_path: Path to target hugging video
            output_dir: Directory to save generated frames
            num_frames: Number of output frames to generate
            easing: Easing function for smooth animation
            video_sampling: Number of poses to extract from target video
            
        Returns:
            True if successful, False otherwise
        """
        print(f"🎬 Generating video-based hugging sequence with {num_frames} frames...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Detect poses from source image
        print("🔍 Detecting poses from source image...")
        source_poses = self.pose_detector.detect_pose_from_image(source_image_path)
        
        if not source_poses:
            print("❌ No poses detected in source image")
            return False
        
        # Step 2: Extract poses from target video
        print("🎥 Extracting poses from target video...")
        target_video_poses = self.extract_poses_from_video(target_video_path, video_sampling)
        
        if not target_video_poses:
            print("❌ No poses extracted from target video")
            return False
        
        print(f"✅ Detected {len(source_poses)} people in source, {len(target_video_poses)} poses in video")
        
        # Step 3: Generate smooth transitions
        print("🎭 Generating smooth pose transitions...")
        success = self._generate_video_transitions(
            source_poses, target_video_poses, output_dir, num_frames, easing
        )
        
        if success:
            print(f"🎉 Successfully generated {num_frames} frames in {output_dir}")
            
            # Create video from frames
            video_path = os.path.join(output_dir, "hugging_sequence.mp4")
            self._create_video_from_frames(output_dir, video_path, fps=12)
            
            # Save metadata
            self._save_video_metadata(source_poses, target_video_poses, output_dir, target_video_path)
            
        return success
    
    def _generate_video_transitions(self, 
                                  source_poses: List[dict],
                                  target_video_poses: List[dict],
                                  output_dir: str,
                                  num_frames: int,
                                  easing: str) -> bool:
        """
        Generate smooth transitions using video pose data
        """
        try:
            # Create interpolation between source and video poses
            for frame_idx in range(num_frames):
                # Calculate interpolation factor with easing
                t = self._apply_easing(frame_idx / (num_frames - 1), easing)
                
                # Interpolate between source and video poses
                interpolated_poses = self._interpolate_with_video_poses(
                    source_poses, target_video_poses, t
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
                if frame_idx % 10 == 0:
                    progress = (frame_idx + 1) / num_frames * 100
                    print(f"   Progress: {progress:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating transitions: {e}")
            return False
    
    def _interpolate_with_video_poses(self, 
                                     source_poses: List[dict],
                                     target_video_poses: List[dict],
                                     t: float) -> List[dict]:
        """
        Interpolate between source poses and video poses
        """
        interpolated_poses = []
        
        # Match people between source and video
        matched_pairs = self._match_people_advanced(source_poses, target_video_poses)
        
        for source_idx, target_pose in matched_pairs:
            if source_idx < len(source_poses):
                source_pose = source_poses[source_idx]
                
                # Interpolate between source and video pose
                interpolated_pose = self._interpolate_single_pose(source_pose, target_pose, t)
                interpolated_poses.append(interpolated_pose)
        
        return interpolated_poses
    
    def _match_people_advanced(self, source_poses: List[dict], target_video_poses: List[dict]) -> List[Tuple[int, dict]]:
        """
        Advanced people matching between source and video poses
        """
        matches = []
        
        # Use the first video pose as reference for matching
        if target_video_poses:
            reference_video_pose = target_video_poses[0]
            
            for i, source_pose in enumerate(source_poses):
                source_center = self._calculate_pose_center(source_pose)
                video_center = self._calculate_pose_center(reference_video_pose)
                
                # Calculate distance between pose centers
                distance = np.linalg.norm(np.array(source_center) - np.array(video_center))
                
                # Match based on position similarity
                if distance < 0.3:  # Threshold for matching
                    matches.append((i, reference_video_pose))
        
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
    
    def _create_video_from_frames(self, frames_dir: str, output_video: str, fps: int = 12):
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
    
    def _save_video_metadata(self, source_poses: List[dict], target_video_poses: List[dict], 
                            output_dir: str, target_video_path: str):
        """
        Save metadata for video-based generation
        """
        try:
            # Save source poses
            source_path = os.path.join(output_dir, "source_poses.json")
            with open(source_path, 'w') as f:
                json.dump(source_poses, f, indent=2)
            
            # Save video pose data
            video_poses_path = os.path.join(output_dir, "video_poses.json")
            with open(video_poses_path, 'w') as f:
                json.dump(target_video_poses, f, indent=2)
            
            # Create enhanced metadata
            metadata = {
                "source_image": "source_poses.json",
                "target_video": os.path.basename(target_video_path),
                "video_poses": "video_poses.json",
                "frames_generated": len([f for f in os.listdir(output_dir) if f.startswith('frame_')]),
                "video_created": "hugging_sequence.mp4",
                "target_video_frames": len(target_video_poses),
                "timestamp": str(np.datetime64('now')),
                "method": "video_based_hugging_effect"
            }
            
            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"📊 Video metadata saved to {output_dir}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save metadata: {e}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Generate video-based hugging effects using MediaPipe")
    parser.add_argument("source_image", help="Path to source image with people standing")
    parser.add_argument("target_video", help="Path to target hugging video")
    parser.add_argument("output_dir", help="Output directory for generated frames and video")
    parser.add_argument("--frames", type=int, default=48, help="Number of frames to generate (default: 48)")
    parser.add_argument("--easing", default="ease_in_out", 
                       choices=["linear", "ease_in", "ease_out", "ease_in_out", "bounce"],
                       help="Easing function for animation (default: ease_in_out)")
    parser.add_argument("--video-sampling", type=int, default=30, 
                       help="Number of poses to extract from target video (default: 30)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.source_image):
        print(f"❌ Source image not found: {args.source_image}")
        sys.exit(1)
    
    if not os.path.exists(args.target_video):
        print(f"❌ Target video not found: {args.target_video}")
        sys.exit(1)
    
    # Create generator and run
    generator = VideoHuggingEffectGenerator()
    
    success = generator.generate_video_hugging_sequence(
        source_image_path=args.source_image,
        target_video_path=args.target_video,
        output_dir=args.output_dir,
        num_frames=args.frames,
        easing=args.easing,
        video_sampling=args.video_sampling
    )
    
    if success:
        print("\n🎉 Video-based hugging effect generation completed successfully!")
        print(f"📁 Check the output directory: {args.output_dir}")
        print(f"🎥 Video file: {os.path.join(args.output_dir, 'hugging_sequence.mp4')}")
        print(f"📊 Video poses extracted: {args.video_sampling} keyframes")
    else:
        print("\n❌ Video-based hugging effect generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
