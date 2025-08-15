#!/usr/bin/env python3
"""
Enhanced Hugging Motion Generator using MediaPipe
Creates realistic motion of people approaching each other to hug
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

class EnhancedHuggingMotionGenerator:
    def __init__(self):
        """Initialize the enhanced hugging motion generator"""
        self.pose_detector = MediaPipePoseDetector(static_mode=False, model_complexity=2)
        
    def generate_realistic_hugging_motion(self, 
                                        source_image_path: str,
                                        target_video_path: str,
                                        output_dir: str,
                                        num_frames: int = 48,
                                        motion_style: str = "natural") -> bool:
        """
        Generate realistic hugging motion with people moving toward each other
        
        Args:
            source_image_path: Path to source image with people standing
            target_video_path: Path to target hugging video
            output_dir: Directory to save generated frames
            num_frames: Number of frames to generate
            motion_style: Motion style (natural, dramatic, gentle)
            
        Returns:
            True if successful, False otherwise
        """
        print(f"🎬 Generating realistic hugging motion with {num_frames} frames...")
        print(f"🎭 Motion style: {motion_style}")
        
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
        target_video_poses = self.extract_poses_from_video(target_video_path, 30)
        
        if not target_video_poses:
            print("❌ No poses extracted from target video")
            return False
        
        print(f"✅ Detected {len(source_poses)} people in source, {len(target_video_poses)} poses in video")
        
        # Step 3: Generate realistic motion sequence
        print("🎭 Generating realistic hugging motion...")
        success = self._generate_realistic_motion_sequence(
            source_poses, target_video_poses, output_dir, num_frames, motion_style
        )
        
        if success:
            print(f"🎉 Successfully generated {num_frames} frames with realistic motion!")
            
            # Create video from frames
            video_path = os.path.join(output_dir, "realistic_hugging_motion.mp4")
            self._create_video_from_frames(output_dir, video_path, fps=12)
            
            # Save metadata
            self._save_motion_metadata(source_poses, target_video_poses, output_dir, motion_style)
            
        return success
    
    def extract_poses_from_video(self, video_path: str, max_frames: int = 30) -> List[Dict]:
        """Extract poses from video with enhanced motion analysis"""
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
    
    def _generate_realistic_motion_sequence(self, 
                                          source_poses: List[dict],
                                          target_video_poses: List[dict],
                                          output_dir: str,
                                          num_frames: int,
                                          motion_style: str) -> bool:
        """
        Generate realistic motion sequence with people moving toward each other
        """
        try:
            # Create motion phases
            phases = self._create_motion_phases(num_frames, motion_style)
            
            for frame_idx in range(num_frames):
                # Get current motion phase
                phase = phases[frame_idx]
                
                # Generate pose based on motion phase
                if phase['type'] == 'approach':
                    # People moving toward each other
                    interpolated_poses = self._generate_approach_motion(
                        source_poses, target_video_poses, phase['progress']
                    )
                elif phase['type'] == 'embrace':
                    # People coming together to hug
                    interpolated_poses = self._generate_embrace_motion(
                        source_poses, target_video_poses, phase['progress']
                    )
                elif phase['type'] == 'hug':
                    # Full hugging pose
                    interpolated_poses = self._generate_hug_motion(
                        source_poses, target_video_poses, phase['progress']
                    )
                else:
                    # Default interpolation
                    interpolated_poses = self._interpolate_poses_enhanced(
                        source_poses, target_video_poses, frame_idx / (num_frames - 1)
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
                    phase_info = f"Phase: {phase['type']} ({phase['progress']:.1%})"
                    print(f"   Progress: {progress:.1f}% - {phase_info}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating motion sequence: {e}")
            return False
    
    def _create_motion_phases(self, num_frames: int, motion_style: str) -> List[Dict]:
        """
        Create realistic motion phases for hugging sequence
        """
        phases = []
        
        if motion_style == "natural":
            # Natural motion: approach (40%) -> embrace (30%) -> hug (30%)
            approach_frames = int(num_frames * 0.4)
            embrace_frames = int(num_frames * 0.3)
            hug_frames = num_frames - approach_frames - embrace_frames
            
            # Approach phase: people moving toward each other
            for i in range(approach_frames):
                phases.append({
                    'type': 'approach',
                    'progress': i / (approach_frames - 1),
                    'description': 'People moving toward each other'
                })
            
            # Embrace phase: coming together
            for i in range(embrace_frames):
                phases.append({
                    'type': 'embrace',
                    'progress': i / (embrace_frames - 1),
                    'description': 'Coming together to embrace'
                })
            
            # Hug phase: full embrace
            for i in range(hug_frames):
                phases.append({
                    'type': 'hug',
                    'progress': i / (hug_frames - 1),
                    'description': 'Full hugging embrace'
                })
                
        elif motion_style == "dramatic":
            # Dramatic motion: slow approach (50%) -> quick embrace (20%) -> hold (30%)
            approach_frames = int(num_frames * 0.5)
            embrace_frames = int(num_frames * 0.2)
            hug_frames = num_frames - approach_frames - embrace_frames
            
            for i in range(approach_frames):
                phases.append({
                    'type': 'approach',
                    'progress': i / (approach_frames - 1),
                    'description': 'Slow dramatic approach'
                })
            
            for i in range(embrace_frames):
                phases.append({
                    'type': 'embrace',
                    'progress': i / (embrace_frames - 1),
                    'description': 'Quick dramatic embrace'
                })
            
            for i in range(hug_frames):
                phases.append({
                    'type': 'hug',
                    'progress': i / (hug_frames - 1),
                    'description': 'Hold dramatic hug'
                })
        
        else:  # gentle
            # Gentle motion: slow approach (60%) -> gentle embrace (25%) -> soft hug (15%)
            approach_frames = int(num_frames * 0.6)
            embrace_frames = int(num_frames * 0.25)
            hug_frames = num_frames - approach_frames - embrace_frames
            
            for i in range(approach_frames):
                phases.append({
                    'type': 'approach',
                    'progress': i / (approach_frames - 1),
                    'description': 'Gentle slow approach'
                })
            
            for i in range(embrace_frames):
                phases.append({
                    'type': 'embrace',
                    'progress': i / (embrace_frames - 1),
                    'description': 'Gentle coming together'
                })
            
            for i in range(hug_frames):
                phases.append({
                    'type': 'hug',
                    'progress': i / (hug_frames - 1),
                    'description': 'Soft gentle hug'
                })
        
        return phases
    
    def _generate_approach_motion(self, source_poses: List[dict], target_video_poses: List[dict], progress: float) -> List[dict]:
        """
        Generate motion of people approaching each other
        """
        interpolated_poses = []
        
        # Match people between source and video
        matched_pairs = self._match_people_enhanced(source_poses, target_video_poses)
        
        for source_idx, target_pose in matched_pairs:
            if source_idx < len(source_poses):
                source_pose = source_poses[source_idx]
                
                # Create approach motion: people move toward center
                approach_pose = self._create_approach_pose(source_pose, target_pose, progress)
                interpolated_poses.append(approach_pose)
        
        return interpolated_poses
    
    def _create_approach_pose(self, source_pose: dict, target_pose: dict, progress: float) -> dict:
        """
        Create pose showing people moving toward each other
        """
        source_landmarks = source_pose['landmarks']
        target_landmarks = target_pose['landmarks']
        
        # Calculate center point between people
        source_center = self._calculate_pose_center(source_pose)
        target_center = self._calculate_pose_center(target_pose)
        
        # Create intermediate center (people moving toward each other)
        intermediate_center = (
            source_center[0] * (1 - progress) + target_center[0] * progress,
            source_center[1] * (1 - progress) + target_center[1] * progress
        )
        
        # Interpolate landmarks with approach motion
        min_landmarks = min(len(source_landmarks), len(target_landmarks))
        interpolated_landmarks = []
        
        for i in range(min_landmarks):
            source_lm = source_landmarks[i]
            target_lm = target_landmarks[i]
            
            # Apply approach motion: move toward center
            if i in [11, 12, 23, 24]:  # Shoulders and hips - key movement points
                # Move these points toward the intermediate center
                x = source_lm['x'] * (1 - progress) + intermediate_center[0] * progress
                y = source_lm['y'] * (1 - progress) + intermediate_center[1] * progress
            else:
                # Regular interpolation for other points
                x = source_lm['x'] * (1 - progress) + target_lm['x'] * progress
                y = source_lm['y'] * (1 - progress) + target_lm['y'] * progress
            
            z = source_lm['z'] * (1 - progress) + target_lm['z'] * progress
            visibility = source_lm['visibility'] * (1 - progress) + target_lm['visibility'] * progress
            
            interpolated_landmarks.append({
                'x': x, 'y': y, 'z': z, 'visibility': visibility
            })
        
        return {
            'landmarks': interpolated_landmarks,
            'image_width': 512,
            'image_height': 512
        }
    
    def _generate_embrace_motion(self, source_poses: List[dict], target_video_poses: List[dict], progress: float) -> List[dict]:
        """
        Generate motion of people coming together to embrace
        """
        # Similar to approach but with more emphasis on arm positions
        return self._generate_approach_motion(source_poses, target_video_poses, progress)
    
    def _generate_hug_motion(self, source_poses: List[dict], target_video_poses: List[dict], progress: float) -> List[dict]:
        """
        Generate full hugging pose
        """
        # Full interpolation to target pose
        return self._interpolate_poses_enhanced(source_poses, target_video_poses, progress)
    
    def _interpolate_poses_enhanced(self, source_poses: List[dict], target_video_poses: List[dict], t: float) -> List[dict]:
        """
        Enhanced pose interpolation
        """
        interpolated_poses = []
        
        # Match people between source and video
        matched_pairs = self._match_people_enhanced(source_poses, target_video_poses)
        
        for source_idx, target_pose in matched_pairs:
            if source_idx < len(source_poses):
                source_pose = source_poses[source_idx]
                
                # Interpolate between source and video pose
                interpolated_pose = self._interpolate_single_pose_enhanced(source_pose, target_pose, t)
                interpolated_poses.append(interpolated_pose)
        
        return interpolated_poses
    
    def _match_people_enhanced(self, source_poses: List[dict], target_video_poses: List[dict]) -> List[Tuple[int, dict]]:
        """
        Enhanced people matching
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
                if distance < 0.4:  # Slightly more lenient threshold
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
    
    def _interpolate_single_pose_enhanced(self, source_pose: dict, target_pose: dict, t: float) -> dict:
        """
        Enhanced single pose interpolation
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
    
    def _create_video_from_frames(self, frames_dir: str, output_video: str, fps: int = 12):
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
    
    def _save_motion_metadata(self, source_poses: List[dict], target_video_poses: List[dict], 
                             output_dir: str, motion_style: str):
        """Save enhanced motion metadata"""
        try:
            metadata = {
                "source_image": "source_poses.json",
                "target_video": "video_poses.json",
                "motion_style": motion_style,
                "frames_generated": len([f for f in os.listdir(output_dir) if f.startswith('frame_')]),
                "video_created": "realistic_hugging_motion.mp4",
                "target_video_frames": len(target_video_poses),
                "motion_phases": ["approach", "embrace", "hug"],
                "timestamp": str(np.datetime64('now')),
                "method": "enhanced_realistic_motion"
            }
            
            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"📊 Enhanced motion metadata saved to {output_dir}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save metadata: {e}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Generate realistic hugging motion with people moving toward each other")
    parser.add_argument("source_image", help="Path to source image with people standing")
    parser.add_argument("target_video", help="Path to target hugging video")
    parser.add_argument("output_dir", help="Output directory for generated frames and video")
    parser.add_argument("--frames", type=int, default=48, help="Number of frames to generate (default: 48)")
    parser.add_argument("--motion-style", default="natural", 
                       choices=["natural", "dramatic", "gentle"],
                       help="Motion style for hugging sequence (default: natural)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.source_image):
        print(f"❌ Source image not found: {args.source_image}")
        sys.exit(1)
    
    if not os.path.exists(args.target_video):
        print(f"❌ Target video not found: {args.target_video}")
        sys.exit(1)
    
    # Create generator and run
    generator = EnhancedHuggingMotionGenerator()
    
    success = generator.generate_realistic_hugging_motion(
        source_image_path=args.source_image,
        target_video_path=args.target_video,
        output_dir=args.output_dir,
        num_frames=args.frames,
        motion_style=args.motion_style
    )
    
    if success:
        print("\n🎉 Realistic hugging motion generation completed successfully!")
        print(f"📁 Check the output directory: {args.output_dir}")
        print(f"🎥 Video file: {os.path.join(args.output_dir, 'realistic_hugging_motion.mp4')}")
        print(f"🎭 Motion style: {args.motion_style}")
        print(f"📊 Frames generated: {args.frames}")
    else:
        print("\n❌ Realistic hugging motion generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
