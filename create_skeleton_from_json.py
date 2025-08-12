#!/usr/bin/env python3
"""
OpenPose JSON to Skeleton Image Converter
Converts OpenPose keypoint JSON data into visual skeleton images for ComfyUI ControlNet
"""

import cv2
import numpy as np
import json
import sys
import os

def create_stick_figure_skeleton(json_file, output_file):
    """Create a stick figure skeleton image from OpenPose JSON data"""
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if not data['people']:
        print("No people detected in the image")
        return False
    
    # Create black background image
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # OpenPose body connections (COCO format)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Head to right arm
        (1, 5), (5, 6), (6, 7),          # Head to left arm
        (1, 8), (8, 9), (9, 10),         # Torso to right leg
        (1, 11), (11, 12), (12, 13),     # Torso to left leg
        (8, 11),                         # Hip connection
        (0, 14), (0, 15),                # Head to eyes
        (14, 16), (15, 17)               # Eyes to ears
    ]
    
    # Colors for different people
    colors = [(0, 255, 255), (255, 0, 255)]  # Cyan, Magenta
    
    for person_idx, person in enumerate(data['people'][:2]):  # Max 2 people
        keypoints = person['pose_keypoints_2d']
        
        # Convert to (x, y, confidence) format
        points = []
        for i in range(0, len(keypoints), 3):
            x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]
            if conf > 0.1:  # Only use confident keypoints
                # Scale to image size
                x_scaled = int(x * 512 / 1000)  # Assuming original ~1000px width
                y_scaled = int(y * 512 / 1000)
                points.append((x_scaled, y_scaled))
            else:
                points.append(None)
        
        color = colors[person_idx % len(colors)]
        
        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                start_pt = points[start_idx]
                end_pt = points[end_idx]
                
                if start_pt and end_pt:
                    cv2.line(img, start_pt, end_pt, (255, 255, 255), 3)
        
        # Draw keypoints
        for i, pt in enumerate(points):
            if pt:
                cv2.circle(img, pt, 6, (0, 255, 0), -1)  # Green dots
                # Add white border
                cv2.circle(img, pt, 8, (255, 255, 255), 2)
    
    # Save the image
    cv2.imwrite(output_file, img)
    print(f"Skeleton image saved to: {output_file}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 create_skeleton_from_json.py <input.json> <output.png>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(json_file):
        print(f"Error: JSON file '{json_file}' not found")
        sys.exit(1)
    
    success = create_stick_figure_skeleton(json_file, output_file)
    if success:
        print("Skeleton conversion completed successfully!")
    else:
        print("Failed to create skeleton image")
        sys.exit(1)
