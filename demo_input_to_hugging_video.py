#!/usr/bin/env python3
"""
Demo script for Input to Hugging Video Generator
Shows how to use the complete workflow
"""

import os
import sys
from input_to_hugging_video_generator import InputToHuggingVideoGenerator

def main():
    """Demo the complete input to hugging video workflow"""
    
    print("🎬 Input to Hugging Video Generator Demo")
    print("=" * 50)
    
    # Setup paths
    input_image = "input/hug-img.jpg"  # User's input image
    output_dir = "output/input_to_hugging_video_demo"
    
    # Check if input image exists
    if not os.path.exists(input_image):
        print(f"⚠️ Demo input image not found: {input_image}")
        print("📝 Please place your input image at the specified path or update the path below:")
        print(f"   Current path: {input_image}")
        
        # Try alternative paths
        alternative_paths = [
            "input/test_image.jpg",
            "input/demo.jpg", 
            "input/sample.png"
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"✅ Found alternative image: {alt_path}")
                input_image = alt_path
                break
        else:
            print("❌ No input image found. Please provide an image to process.")
            return False
    
    # Create generator
    print(f"\n🚀 Initializing generator...")
    generator = InputToHuggingVideoGenerator()
    
    # Custom settings for demo
    demo_settings = {
        "prompt": "two people embracing in a warm hug, romantic scene, soft lighting, high quality, detailed, cinematic",
        "negative_prompt": "text, watermark, blurry, low quality, distorted, ugly, deformed, extra limbs",
        "controlnet_strength": 1.2,  # Stronger pose control
        "steps": 30,  # More steps for quality
        "cfg": 8.0,   # Higher CFG for prompt adherence
        "seed": 123   # Fixed seed for reproducibility
    }
    
    print(f"📁 Input image: {input_image}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎭 Settings: {demo_settings}")
    
    # Generate the hugging video
    print(f"\n🎬 Starting hugging video generation...")
    success = generator.generate_hugging_video_from_input(
        input_image_path=input_image,
        output_dir=output_dir,
        **demo_settings
    )
    
    if success:
        print(f"\n🎉 Demo completed successfully!")
        print(f"\n📂 Generated files:")
        print(f"   📁 Full output: {output_dir}/")
        print(f"   🎥 Final video: {output_dir}/hugging_video.mp4")
        print(f"   🎭 Skeleton frames: {output_dir}/skeleton_frames/")
        print(f"   🎯 ControlNet inputs: {output_dir}/processed_frames/")
        print(f"   📋 Instructions: {output_dir}/processing_instructions.md")
        print(f"   🐍 ComfyUI script: {output_dir}/comfyui_process.py")
        
        print(f"\n🔧 Next steps:")
        print(f"   1. Review the processing instructions")
        print(f"   2. Set up ComfyUI with required models")
        print(f"   3. Run the generated ComfyUI script")
        print(f"   4. Check the final video output")
        
        return True
    else:
        print(f"\n❌ Demo failed!")
        return False

def show_workflow_overview():
    """Show an overview of the complete workflow"""
    print("\n📋 Workflow Overview:")
    print("=" * 30)
    print("1. 📷 Input Image Analysis")
    print("   • Load and resize user's image")
    print("   • Detect existing poses (optional)")
    print("   • Prepare for style transfer")
    print()
    print("2. 🎭 Skeleton Generation")
    print("   • Generate 16 walking/hugging skeleton frames")
    print("   • Create clean pose sequences")
    print("   • MediaPipe-based motion")
    print()
    print("3. 🎯 ControlNet Preparation")
    print("   • Convert skeletons to ControlNet format")
    print("   • Prepare pose guidance frames")
    print("   • Set up batch processing")
    print()
    print("4. 🎨 Frame Generation")
    print("   • Apply ControlNet pose guidance")
    print("   • Generate realistic frames")
    print("   • Maintain consistent style")
    print()
    print("5. 🎥 Video Assembly")
    print("   • Combine generated frames")
    print("   • Create smooth video")
    print("   • Apply AnimateDiff (optional)")
    print()

if __name__ == "__main__":
    print("🎬 Input to Hugging Video Generator")
    print("Complete workflow: Image → Hugging Video")
    
    # Show workflow overview
    show_workflow_overview()
    
    # Run demo
    if len(sys.argv) > 1 and sys.argv[1] == "--overview":
        print("📋 Workflow overview displayed above.")
    else:
        main()
