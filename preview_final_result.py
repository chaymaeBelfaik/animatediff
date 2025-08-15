#!/usr/bin/env python3
"""
Preview Final Result
Creates a preview of what the final hugging video might look like
by combining your input image style with the skeleton frames
"""

import cv2
import numpy as np
import os
from pathlib import Path

def create_style_preview(input_image_path: str, skeleton_frame_path: str, output_path: str):
    """Create a style preview by blending input image with skeleton"""
    
    # Load input image and skeleton
    input_img = cv2.imread(input_image_path)
    skeleton_img = cv2.imread(skeleton_frame_path)
    
    if input_img is None or skeleton_img is None:
        return False
    
    # Resize both to same size
    height, width = 512, 512
    input_resized = cv2.resize(input_img, (width, height))
    skeleton_resized = cv2.resize(skeleton_img, (width, height))
    
    # Create a blended preview
    # Method 1: Show skeleton over blurred background
    blurred_input = cv2.GaussianBlur(input_resized, (21, 21), 0)
    
    # Extract skeleton (non-black pixels)
    skeleton_gray = cv2.cvtColor(skeleton_resized, cv2.COLOR_BGR2GRAY)
    skeleton_mask = skeleton_gray > 10
    
    # Create preview
    preview = blurred_input.copy()
    preview[skeleton_mask] = skeleton_resized[skeleton_mask]
    
    # Add text overlay
    cv2.putText(preview, "PREVIEW: Input Style + Skeleton Guidance", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(preview, "Final result will be realistic people in this pose", 
                (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imwrite(output_path, preview)
    return True

def create_preview_sequence(workflow_dir: str):
    """Create preview sequence showing style + skeleton combination"""
    
    print("🎨 Creating final result preview...")
    
    input_image = os.path.join(workflow_dir, "input_resized.png")
    skeleton_dir = os.path.join(workflow_dir, "skeleton_frames")
    preview_dir = os.path.join(workflow_dir, "preview_final_result")
    
    if not os.path.exists(input_image):
        print(f"❌ Input image not found: {input_image}")
        return False
    
    if not os.path.exists(skeleton_dir):
        print(f"❌ Skeleton frames not found: {skeleton_dir}")
        return False
    
    # Create preview directory
    os.makedirs(preview_dir, exist_ok=True)
    
    # Get skeleton frames
    skeleton_frames = sorted([f for f in os.listdir(skeleton_dir) 
                             if f.startswith('control_') and f.endswith('.png')])
    
    if not skeleton_frames:
        print(f"❌ No skeleton frames found in {skeleton_dir}")
        return False
    
    print(f"   Processing {len(skeleton_frames)} frames...")
    
    # Create preview for key frames
    key_frames = [0, len(skeleton_frames)//4, len(skeleton_frames)//2, 
                 3*len(skeleton_frames)//4, len(skeleton_frames)-1]
    
    for i in key_frames:
        if i < len(skeleton_frames):
            skeleton_path = os.path.join(skeleton_dir, skeleton_frames[i])
            preview_path = os.path.join(preview_dir, f"preview_{i:03d}.png")
            
            success = create_style_preview(input_image, skeleton_path, preview_path)
            if success:
                progress = i / (len(skeleton_frames) - 1)
                motion_phase = get_motion_phase(progress)
                print(f"   ✅ Frame {i:2d}: {motion_phase}")
            else:
                print(f"   ❌ Failed to create preview for frame {i}")
    
    # Create preview video
    preview_video_path = os.path.join(workflow_dir, "preview_final_result.mp4")
    create_preview_video(preview_dir, preview_video_path)
    
    print(f"✅ Preview sequence created in: {preview_dir}")
    print(f"🎥 Preview video: {preview_video_path}")
    
    return True

def create_preview_video(preview_dir: str, output_video: str):
    """Create video from preview frames"""
    try:
        preview_files = sorted([f for f in os.listdir(preview_dir) 
                               if f.startswith('preview_') and f.endswith('.png')])
        
        if not preview_files:
            print("   ⚠️ No preview frames found for video")
            return False
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(os.path.join(preview_dir, preview_files[0]))
        height, width, layers = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video, fourcc, 4, (width, height))  # Slower 4 fps
        
        # Write frames multiple times for longer duration
        for frame_file in preview_files:
            frame_path = os.path.join(preview_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                # Write each frame multiple times for smoother preview
                for _ in range(4):  # Hold each frame for 1 second at 4 fps
                    video_writer.write(frame)
        
        video_writer.release()
        print(f"   ✅ Preview video created: {output_video}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error creating preview video: {e}")
        return False

def get_motion_phase(progress: float) -> str:
    """Get motion phase description"""
    if progress < 0.3:
        return "Standing apart, ready to walk"
    elif progress < 0.7:
        return "Walking toward each other"
    else:
        return "Embracing in warm hug"

def show_preview_instructions(workflow_dir: str):
    """Show how to use the preview"""
    
    print(f"\n📋 HOW TO INTERPRET THE PREVIEW")
    print("=" * 50)
    print(f"The preview shows:")
    print(f"• Your input image style (blurred background)")
    print(f"• Skeleton poses overlaid (where people will be)")
    print(f"• Text indicating this is just a preview")
    print(f"")
    print(f"🎯 FINAL RESULT WILL BE:")
    print(f"• Realistic people (not skeletons)")
    print(f"• Following the exact same poses")
    print(f"• With the style/lighting from your input image")
    print(f"• Smooth motion between all 16 frames")
    print(f"")
    print(f"📂 Preview files:")
    print(f"• preview_final_result/: Key frame previews")
    print(f"• preview_final_result.mp4: Preview video")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preview final hugging video result")
    parser.add_argument("workflow_dir", help="Workflow directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.workflow_dir):
        print(f"❌ Workflow directory not found: {args.workflow_dir}")
        return
    
    print(f"🎬 Creating preview of final hugging video result...")
    
    success = create_preview_sequence(args.workflow_dir)
    
    if success:
        show_preview_instructions(args.workflow_dir)
        print(f"\n🎉 Preview created successfully!")
        print(f"📁 Check: {args.workflow_dir}/preview_final_result/")
    else:
        print(f"\n❌ Preview creation failed!")

if __name__ == "__main__":
    main()
