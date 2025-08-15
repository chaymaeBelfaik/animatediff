#!/usr/bin/env python3
"""
Fix Input Poses - Create proper 2-person pose visualization
"""

import cv2
import numpy as np
import os
from clean_skeleton_hugging_generator import CleanSkeletonHuggingGenerator

def create_two_person_input_pose(output_path: str, image_size: tuple = (512, 512)) -> bool:
    """
    Create a proper 2-person pose visualization for input reference
    This shows the initial standing pose that will be transformed into hugging
    """
    print("🎭 Creating 2-person input pose visualization...")
    
    # Create black background
    img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    
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
    _draw_person_pose(img, person1_pose, person1_color, joint_color, image_size)
    _draw_person_pose(img, person2_pose, person2_color, joint_color, image_size)
    
    # Add labels
    cv2.putText(img, "Person 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, person1_color, 2)
    cv2.putText(img, "Person 2", (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, person2_color, 2)
    
    cv2.putText(img, "Initial Standing Pose", (150, image_size[1] - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # Save the image
    cv2.imwrite(output_path, img)
    print(f"   ✅ 2-person input pose saved to: {output_path}")
    
    return True

def _draw_person_pose(img, pose, person_color, joint_color, image_size):
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
            start_pos = _normalize_position(pose[start_joint], image_size)
            end_pos = _normalize_position(pose[end_joint], image_size)
            cv2.line(img, start_pos, end_pos, person_color, 2)
    
    # Draw joints
    for joint_name, position in pose.items():
        if position:
            pos = _normalize_position(position, image_size)
            cv2.circle(img, pos, 4, joint_color, -1)

def _normalize_position(pos, image_size):
    """Convert normalized coordinates (0-1) to pixel coordinates"""
    x = int(pos[0] * image_size[0])
    y = int(pos[1] * image_size[1])
    return (x, y)

def fix_workflow_input_poses(workflow_dir: str) -> bool:
    """Fix the input poses in an existing workflow directory"""
    print(f"🔧 Fixing input poses in {workflow_dir}")
    
    if not os.path.exists(workflow_dir):
        print(f"❌ Workflow directory not found: {workflow_dir}")
        return False
    
    # Create correct 2-person input pose
    input_poses_path = os.path.join(workflow_dir, "input_poses.png")
    success = create_two_person_input_pose(input_poses_path)
    
    if success:
        print(f"✅ Fixed input poses - now shows 2 people standing")
        return True
    else:
        print(f"❌ Failed to fix input poses")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix input poses to show 2 people")
    parser.add_argument("workflow_dir", help="Workflow directory to fix")
    
    args = parser.parse_args()
    
    success = fix_workflow_input_poses(args.workflow_dir)
    
    if success:
        print(f"\n🎉 Input poses fixed successfully!")
        print(f"📁 Updated file: {args.workflow_dir}/input_poses.png")
        print(f"👥 Now shows: 2 people in initial standing position")
    else:
        print(f"\n❌ Failed to fix input poses!")

if __name__ == "__main__":
    main()
