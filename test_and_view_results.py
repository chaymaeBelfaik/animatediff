#!/usr/bin/env python3
"""
Test and View Results
Interactive viewer for the hugging video workflow
"""

import cv2
import os
import subprocess
import time
from pathlib import Path

class ResultViewer:
    def __init__(self, workflow_dir: str):
        """Initialize the result viewer"""
        self.workflow_dir = workflow_dir
        self.skeleton_frames_dir = os.path.join(workflow_dir, "skeleton_frames")
        
        print(f"🎬 Result Viewer for: {workflow_dir}")
        
    def view_all_results(self):
        """View all available results"""
        print(f"\n📂 Viewing results from: {self.workflow_dir}")
        print("=" * 60)
        
        # 1. Check input analysis
        self.view_input_analysis()
        
        # 2. View skeleton frames
        self.view_skeleton_frames()
        
        # 3. Play skeleton video
        self.play_skeleton_video()
        
        # 4. View demo results
        self.view_demo_results()
        
        # 5. Show instructions
        self.show_instructions()
        
        # 6. Test individual frames
        self.test_individual_frames()
        
    def view_input_analysis(self):
        """View input image analysis"""
        print(f"\n1. 📷 INPUT IMAGE ANALYSIS")
        print("-" * 30)
        
        # Original input (resized)
        input_resized = os.path.join(self.workflow_dir, "input_resized.png")
        if os.path.exists(input_resized):
            img = cv2.imread(input_resized)
            print(f"✅ Input image: {input_resized}")
            print(f"   Size: {img.shape[1]}x{img.shape[0]} pixels")
            self._show_image_info(input_resized)
        
        # Pose analysis
        input_poses = os.path.join(self.workflow_dir, "input_poses.png")
        if os.path.exists(input_poses):
            print(f"✅ Pose reference: {input_poses}")
            print(f"   Shows: 2 people in initial standing position")
            self._show_image_info(input_poses)
        
    def view_skeleton_frames(self):
        """View skeleton frames information"""
        print(f"\n2. 🎭 SKELETON FRAMES ANALYSIS")
        print("-" * 30)
        
        if not os.path.exists(self.skeleton_frames_dir):
            print(f"❌ Skeleton frames directory not found: {self.skeleton_frames_dir}")
            return
        
        # Count frames
        frame_files = [f for f in os.listdir(self.skeleton_frames_dir) 
                      if f.startswith('control_') and f.endswith('.png')]
        frame_count = len(frame_files)
        
        print(f"✅ Skeleton frames found: {frame_count}")
        print(f"📁 Location: {self.skeleton_frames_dir}")
        
        if frame_count > 0:
            # Show first, middle, and last frame info
            first_frame = os.path.join(self.skeleton_frames_dir, "control_000.png")
            middle_frame = os.path.join(self.skeleton_frames_dir, f"control_{frame_count//2:03d}.png")
            last_frame = os.path.join(self.skeleton_frames_dir, f"control_{frame_count-1:03d}.png")
            
            print(f"   First frame: {first_frame} (Standing apart)")
            self._show_image_info(first_frame)
            
            print(f"   Middle frame: {middle_frame} (Approaching)")
            self._show_image_info(middle_frame)
            
            print(f"   Last frame: {last_frame} (Hugging)")
            self._show_image_info(last_frame)
    
    def play_skeleton_video(self):
        """Play skeleton video if available"""
        print(f"\n3. 🎥 SKELETON VIDEO PLAYBACK")
        print("-" * 30)
        
        skeleton_video = os.path.join(self.workflow_dir, "skeleton_sequence.mp4")
        if os.path.exists(skeleton_video):
            print(f"✅ Skeleton video: {skeleton_video}")
            
            # Get video info
            cap = cv2.VideoCapture(skeleton_video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            print(f"   Duration: {duration:.1f} seconds")
            print(f"   Frames: {frame_count}")
            print(f"   FPS: {fps}")
            
            # Try to play with different methods
            self._try_play_video(skeleton_video, "Skeleton Animation")
        else:
            print(f"⚠️ Skeleton video not found: {skeleton_video}")
    
    def view_demo_results(self):
        """View demo results"""
        print(f"\n4. 🎨 DEMO RESULTS")
        print("-" * 30)
        
        demo_video = os.path.join(self.workflow_dir, "demo_hugging_video.mp4")
        demo_frames_dir = os.path.join(self.workflow_dir, "demo_generated_frames")
        
        if os.path.exists(demo_video):
            print(f"✅ Demo video: {demo_video}")
            self._try_play_video(demo_video, "Demo Hugging Video")
        
        if os.path.exists(demo_frames_dir):
            demo_frames = [f for f in os.listdir(demo_frames_dir) 
                          if f.startswith('demo_generated_') and f.endswith('.png')]
            print(f"✅ Demo frames: {len(demo_frames)} frames in {demo_frames_dir}")
    
    def show_instructions(self):
        """Show processing instructions"""
        print(f"\n5. 📋 PROCESSING INSTRUCTIONS")
        print("-" * 30)
        
        instructions_file = os.path.join(self.workflow_dir, "INSTRUCTIONS.md")
        if os.path.exists(instructions_file):
            print(f"✅ Instructions: {instructions_file}")
            print(f"📖 Content preview:")
            
            with open(instructions_file, 'r') as f:
                lines = f.readlines()[:10]  # First 10 lines
                for i, line in enumerate(lines, 1):
                    print(f"   {i:2d}: {line.rstrip()}")
            
            if len(lines) == 10:
                print(f"   ... (see full file for complete instructions)")
        else:
            print(f"⚠️ Instructions file not found: {instructions_file}")
    
    def test_individual_frames(self):
        """Test individual skeleton frames"""
        print(f"\n6. 🔍 INDIVIDUAL FRAME TESTING")
        print("-" * 30)
        
        if not os.path.exists(self.skeleton_frames_dir):
            print(f"❌ No skeleton frames to test")
            return
        
        frame_files = sorted([f for f in os.listdir(self.skeleton_frames_dir) 
                             if f.startswith('control_') and f.endswith('.png')])
        
        if not frame_files:
            print(f"❌ No control frames found")
            return
        
        print(f"✅ Testing {len(frame_files)} skeleton frames...")
        
        # Test a few key frames
        test_frames = [0, len(frame_files)//4, len(frame_files)//2, 
                      3*len(frame_files)//4, len(frame_files)-1]
        
        for i in test_frames:
            if i < len(frame_files):
                frame_path = os.path.join(self.skeleton_frames_dir, frame_files[i])
                img = cv2.imread(frame_path)
                if img is not None:
                    motion_phase = self._get_motion_phase(i / (len(frame_files) - 1))
                    print(f"   Frame {i:2d}: {frame_files[i]} - {motion_phase}")
                    print(f"            Size: {img.shape[1]}x{img.shape[0]}, Non-black pixels: {self._count_non_black_pixels(img)}")
    
    def _show_image_info(self, image_path: str):
        """Show basic image information"""
        try:
            img = cv2.imread(image_path)
            if img is not None:
                file_size = os.path.getsize(image_path) / 1024  # KB
                non_black = self._count_non_black_pixels(img)
                print(f"            Size: {img.shape[1]}x{img.shape[0]}, File: {file_size:.1f}KB, Content: {non_black} pixels")
            else:
                print(f"            ❌ Could not load image")
        except Exception as e:
            print(f"            ❌ Error reading image: {e}")
    
    def _count_non_black_pixels(self, img):
        """Count non-black pixels in image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.countNonZero(gray)
    
    def _get_motion_phase(self, progress: float) -> str:
        """Get motion phase description"""
        if progress < 0.3:
            return "Walking toward each other"
        elif progress < 0.7:
            return "Approaching, arms extending"
        else:
            return "Embracing in warm hug"
    
    def _try_play_video(self, video_path: str, title: str):
        """Try to play video with different methods"""
        print(f"   🎬 Playing: {title}")
        
        # Method 1: Try opening with system default
        try:
            if os.name == 'nt':  # Windows
                os.startfile(video_path)
                print(f"   ✅ Opened with system default player")
                return
        except:
            pass
        
        # Method 2: Try common video players
        players = ['vlc', 'mpv', 'mplayer', 'xdg-open']
        for player in players:
            try:
                subprocess.run([player, video_path], check=True, timeout=1)
                print(f"   ✅ Opened with {player}")
                return
            except:
                continue
        
        # Method 3: Show video info instead
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                print(f"   📊 Video info: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames, "
                     f"{cap.get(cv2.CAP_PROP_FPS):.1f} fps")
                cap.release()
                print(f"   💡 To view: Download {video_path} and open with your video player")
            else:
                print(f"   ❌ Could not open video file")
        except Exception as e:
            print(f"   ❌ Error reading video: {e}")
    
    def create_test_summary(self):
        """Create a comprehensive test summary"""
        print(f"\n📊 TEST SUMMARY")
        print("=" * 60)
        
        # Check all expected files
        expected_files = {
            'input_resized.png': 'Input image (512x512)',
            'input_poses.png': '2-person pose reference',
            'skeleton_sequence.mp4': 'Skeleton animation video',
            'demo_hugging_video.mp4': 'Demo result video',
            'INSTRUCTIONS.md': 'Processing instructions',
            'skeleton_frames/': 'Directory with 16 control frames'
        }
        
        print(f"✅ READY FILES:")
        for file_name, description in expected_files.items():
            file_path = os.path.join(self.workflow_dir, file_name)
            if os.path.exists(file_path):
                print(f"   ✅ {file_name}: {description}")
            else:
                print(f"   ❌ {file_name}: MISSING - {description}")
        
        # Next steps
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Review skeleton frames in: {self.skeleton_frames_dir}")
        print(f"   2. Follow instructions in: INSTRUCTIONS.md")
        print(f"   3. Set up ComfyUI with ControlNet OpenPose model")
        print(f"   4. Process frames to generate realistic video")
        print(f"   5. Combine generated frames into final video")
        
        # ComfyUI readiness check
        print(f"\n🎯 COMFYUI READINESS:")
        skeleton_frames = [f for f in os.listdir(self.skeleton_frames_dir) 
                          if f.startswith('control_') and f.endswith('.png')]
        if len(skeleton_frames) == 16:
            print(f"   ✅ All 16 skeleton frames ready for ControlNet")
            print(f"   ✅ Input style image ready")
            print(f"   ✅ Processing instructions available")
            print(f"   🎬 READY FOR COMFYUI PROCESSING!")
        else:
            print(f"   ❌ Expected 16 frames, found {len(skeleton_frames)}")

def main():
    """Main function for testing results"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test and view hugging video workflow results")
    parser.add_argument("workflow_dir", help="Workflow directory to test")
    parser.add_argument("--summary", action="store_true", help="Show only summary")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.workflow_dir):
        print(f"❌ Workflow directory not found: {args.workflow_dir}")
        return
    
    viewer = ResultViewer(args.workflow_dir)
    
    if args.summary:
        viewer.create_test_summary()
    else:
        viewer.view_all_results()
        viewer.create_test_summary()

if __name__ == "__main__":
    main()
