#!/usr/bin/env python3
"""
Demo script for MediaPipe Hugging Effect Generator
Shows how to create hugging animations from static images
"""

import os
import sys
from hugging_effect_generator import HuggingEffectGenerator

def create_sample_hugging_effect():
    """
    Create a sample hugging effect using MediaPipe
    """
    print("🎬 MediaPipe Hugging Effect Demo")
    print("=" * 40)
    
    # Initialize the generator
    generator = HuggingEffectGenerator()
    
    # Example usage - you'll need to provide your own images
    print("\n📋 To use this demo, you need:")
    print("1. A source image with 2 people standing (e.g., 'people_standing.jpg')")
    print("2. A target image with hugging pose (e.g., 'hugging_pose.jpg')")
    print("3. An output directory for results")
    
    # Check if sample images exist
    source_image = "input/people_standing.jpg"
    target_image = "input/hugging_pose.jpg"
    output_dir = "output/hugging_effect"
    
    print(f"\n🔍 Looking for sample images...")
    
    if os.path.exists(source_image) and os.path.exists(target_image):
        print(f"✅ Found sample images!")
        print(f"   Source: {source_image}")
        print(f"   Target: {target_image}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🎭 Generating hugging effect...")
        success = generator.generate_hugging_sequence(
            source_image_path=source_image,
            target_hug_pose_path=target_image,
            output_dir=output_dir,
            num_frames=16,  # 16 frames for smooth animation
            easing="ease_in_out"
        )
        
        if success:
            print(f"\n🎉 Demo completed successfully!")
            print(f"📁 Check results in: {output_dir}")
            print(f"🎥 Video file: {os.path.join(output_dir, 'hugging_sequence.mp4')}")
        else:
            print(f"\n❌ Demo failed!")
            
    else:
        print(f"❌ Sample images not found!")
        print(f"   Please place your images in the input/ directory:")
        print(f"   - {source_image}")
        print(f"   - {target_image}")
        
        # Show how to use with custom images
        print(f"\n💡 To use with your own images:")
        print(f"   python hugging_effect_generator.py your_source.jpg your_target.jpg output/")

def show_mediapipe_vs_openpose_comparison():
    """
    Show the advantages of MediaPipe over OpenPose
    """
    print("\n🔍 MediaPipe vs OpenPose Comparison")
    print("=" * 40)
    
    comparison = {
        "Installation": {
            "MediaPipe": "pip install mediapipe (5 seconds)",
            "OpenPose": "Complex compilation (30+ minutes)"
        },
        "Python Integration": {
            "MediaPipe": "Native Python API",
            "OpenPose": "Command-line binaries + JSON parsing"
        },
        "Performance": {
            "MediaPipe": "Optimized, real-time capable",
            "OpenPose": "Heavy, slower processing"
        },
        "Accuracy": {
            "MediaPipe": "33 landmarks, better pose estimation",
            "OpenPose": "25 landmarks, older model"
        },
        "Maintenance": {
            "MediaPipe": "Google actively maintains",
            "OpenPose": "Community maintained, less frequent updates"
        }
    }
    
    for category, details in comparison.items():
        print(f"\n📊 {category}:")
        print(f"   ✅ MediaPipe: {details['MediaPipe']}")
        print(f"   ❌ OpenPose: {details['OpenPose']}")

def show_usage_examples():
    """
    Show various usage examples
    """
    print("\n📚 Usage Examples")
    print("=" * 40)
    
    examples = [
        {
            "title": "Basic Hugging Effect",
            "command": "python hugging_effect_generator.py source.jpg target.jpg output/",
            "description": "Generate 24 frames with smooth easing"
        },
        {
            "title": "Custom Frame Count",
            "command": "python hugging_effect_generator.py source.jpg target.jpg output/ --frames 32",
            "description": "Generate 32 frames for smoother animation"
        },
        {
            "title": "Different Easing",
            "command": "python hugging_effect_generator.py source.jpg target.jpg output/ --easing bounce",
            "description": "Use bounce easing for playful animation"
        },
        {
            "title": "Programmatic Usage",
            "command": "from hugging_effect_generator import HuggingEffectGenerator\n# ... code example",
            "description": "Use in your own Python scripts"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   Command: {example['command']}")
        print(f"   Description: {example['description']}")

def main():
    """Main demo function"""
    print("🎬 MediaPipe Hugging Effect Generator Demo")
    print("=" * 50)
    
    # Show comparison
    show_mediapipe_vs_openpose_comparison()
    
    # Show usage examples
    show_usage_examples()
    
    # Try to run demo
    print("\n" + "=" * 50)
    create_sample_hugging_effect()
    
    print("\n🎯 Next Steps:")
    print("1. Install MediaPipe: pip install mediapipe")
    print("2. Prepare your source and target images")
    print("3. Run: python hugging_effect_generator.py source.jpg target.jpg output/")
    print("4. Use the generated frames with AnimateDiff for final video generation")

if __name__ == "__main__":
    main()
