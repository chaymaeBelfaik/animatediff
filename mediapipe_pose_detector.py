#!/usr/bin/env python3
"""
MediaPipe Pose Detection and Skeleton Generation
Replaces OpenPose with MediaPipe for easier pose detection and skeleton generation
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import sys
import os
from typing import List, Tuple, Optional

class MediaPipePoseDetector:
    def __init__(self, static_mode=False, model_complexity=1, smooth_landmarks=True):
        """
        Initialize MediaPipe pose detection
        
        Args:
            static_mode: If True, processes single images. If False, processes video streams
            model_complexity: 0, 1, or 2 (higher = more accurate but slower)
            smooth_landmarks: If True, applies smoothing to landmarks
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect_pose_from_image(self, image_path: str) -> List[dict]:
        """
        Detect poses from a single image
        
        Args:
            image_path: Path to the input image
            
        Returns:
            List of detected poses with landmarks
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        poses = []
        if results.pose_landmarks:
            # Convert landmarks to list format
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            poses.append({
                'landmarks': landmarks,
                'image_width': image.shape[1],
                'image_height': image.shape[0]
            })
        
        return poses
    
    def create_skeleton_image(self, poses: List[dict], output_path: str, 
                            image_size: Tuple[int, int] = (512, 512),
                            show_landmarks: bool = True,
                            show_connections: bool = True) -> bool:
        """
        Create a skeleton image from detected poses
        
        Args:
            poses: List of detected poses
            output_path: Path to save the skeleton image
            image_size: Size of output image (width, height)
            show_landmarks: Whether to show landmark points
            show_connections: Whether to show connections between landmarks
            
        Returns:
            True if successful, False otherwise
        """
        if not poses:
            print("No poses detected")
            return False
        
        # Create black background
        img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        
        # MediaPipe pose connections (33 landmarks)
        connections = [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),  # Nose to left ear
            (0, 4), (4, 5), (5, 6), (6, 8),  # Nose to right ear
            (9, 10),  # Mouth
            
            # Upper body
            (11, 12),  # Shoulders
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            
            # Lower body
            (11, 23), (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
            (12, 24), (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
            (23, 24),  # Hips
            
            # Hands
            (15, 17), (17, 19), (19, 21),  # Left hand
            (16, 18), (18, 20), (20, 22),  # Right hand
        ]
        
        # Colors for different people
        colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0)]  # Cyan, Magenta, Yellow
        
        for person_idx, pose in enumerate(poses):
            landmarks = pose['landmarks']
            img_width = pose['image_width']
            img_height = pose['image_height']
            
            color = colors[person_idx % len(colors)]
            
            # Convert landmarks to image coordinates
            points = []
            for landmark in landmarks:
                if landmark['visibility'] > 0.5:  # Only use visible landmarks
                    x = int(landmark['x'] * image_size[0])
                    y = int(landmark['y'] * image_size[1])
                    points.append((x, y))
                else:
                    points.append(None)
            
            # Draw connections
            if show_connections:
                for start_idx, end_idx in connections:
                    if (start_idx < len(points) and end_idx < len(points) and
                        points[start_idx] and points[end_idx]):
                        cv2.line(img, points[start_idx], points[end_idx], 
                                (255, 255, 255), 2)
            
            # Draw landmarks
            if show_landmarks:
                for i, pt in enumerate(points):
                    if pt:
                        # Main landmark
                        cv2.circle(img, pt, 4, color, -1)
                        # White border
                        cv2.circle(img, pt, 6, (255, 255, 255), 1)
        
        # Save the image
        cv2.imwrite(output_path, img)
        print(f"Skeleton image saved to: {output_path}")
        return True
    
    def create_hugging_animation_frames(self, source_image_path: str, 
                                      target_pose_path: str,
                                      output_dir: str,
                                      num_frames: int = 16) -> bool:
        """
        Create a sequence of frames for hugging animation
        
        Args:
            source_image_path: Path to source image with people
            target_pose_path: Path to target hugging pose image
            output_dir: Directory to save animation frames
            num_frames: Number of frames to generate
            
        Returns:
            True if successful, False otherwise
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Detect poses from source and target
        source_poses = self.detect_pose_from_image(source_image_path)
        target_poses = self.detect_pose_from_image(target_pose_path)
        
        if not source_poses or not target_poses:
            print("Could not detect poses in source or target images")
            return False
        
        # Generate intermediate poses for smooth transition
        for frame_idx in range(num_frames):
            # Interpolate between source and target poses
            t = frame_idx / (num_frames - 1)
            
            # Create interpolated pose
            interpolated_poses = self._interpolate_poses(source_poses, target_poses, t)
            
            # Create skeleton image for this frame
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
            self.create_skeleton_image(interpolated_poses, frame_path)
        
        print(f"Generated {num_frames} animation frames in {output_dir}")
        return True
    
    def _interpolate_poses(self, source_poses: List[dict], 
                          target_poses: List[dict], t: float) -> List[dict]:
        """
        Interpolate between source and target poses
        
        Args:
            source_poses: Source pose landmarks
            target_poses: Target pose landmarks
            t: Interpolation factor (0.0 to 1.0)
            
        Returns:
            Interpolated pose landmarks
        """
        interpolated_poses = []
        
        # For simplicity, interpolate the first person from each pose set
        if source_poses and target_poses:
            source_landmarks = source_poses[0]['landmarks']
            target_landmarks = target_poses[0]['landmarks']
            
            # Ensure both have the same number of landmarks
            min_landmarks = min(len(source_landmarks), len(target_landmarks))
            
            interpolated_landmarks = []
            for i in range(min_landmarks):
                source_lm = source_landmarks[i]
                target_lm = target_landmarks[i]
                
                # Interpolate x, y, z coordinates
                x = source_lm['x'] * (1 - t) + target_lm['x'] * t
                y = source_lm['y'] * (1 - t) + target_lm['y'] * t
                z = source_lm['z'] * (1 - t) + target_lm['z'] * t
                
                # Interpolate visibility
                visibility = source_lm['visibility'] * (1 - t) + target_lm['visibility'] * t
                
                interpolated_landmarks.append({
                    'x': x, 'y': y, 'z': z, 'visibility': visibility
                })
            
            interpolated_poses.append({
                'landmarks': interpolated_landmarks,
                'image_width': 512,
                'image_height': 512
            })
        
        return interpolated_poses
    
    def save_poses_to_json(self, poses: List[dict], output_path: str) -> bool:
        """
        Save detected poses to JSON file (for compatibility with existing workflows)
        
        Args:
            poses: List of detected poses
            output_path: Path to save JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(poses, f, indent=2)
            print(f"Poses saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving poses: {e}")
            return False

def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 3:
        print("Usage: python mediapipe_pose_detector.py <input_image> <output_skeleton>")
        print("Example: python mediapipe_pose_detector.py input.jpg skeleton.png")
        sys.exit(1)
    
    input_image = sys.argv[1]
    output_skeleton = sys.argv[2]
    
    # Initialize detector
    detector = MediaPipePoseDetector(static_mode=True)
    
    try:
        # Detect poses
        poses = detector.detect_pose_from_image(input_image)
        
        if poses:
            # Create skeleton image
            success = detector.create_skeleton_image(poses, output_skeleton)
            if success:
                print(f"Successfully created skeleton from {input_image}")
                
                # Also save poses to JSON for compatibility
                json_path = output_skeleton.replace('.png', '.json')
                detector.save_poses_to_json(poses, json_path)
            else:
                print("Failed to create skeleton image")
        else:
            print("No poses detected in the image")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
