# ComfyUI Setup Guide - Hugging Video Generation

## 🎯 WHAT TO PUT IN COMFYUI

### 📁 **Files to Copy to ComfyUI**

#### 1. **Skeleton Control Frames** (ESSENTIAL)
**Location:** `output/hug_img_simple_workflow/skeleton_frames/`
**Files:** `control_000.png` to `control_015.png` (16 files)
**Copy to:** `ComfyUI/input/` or your input folder

```
control_000.png  → Frame 1: Two people standing apart
control_001.png  → Frame 2: Starting to walk
control_002.png  → Frame 3: Walking motion
...
control_015.png  → Frame 16: Final hug embrace
```

#### 2. **Input Style Image** (ESSENTIAL)
**File:** `output/hug_img_simple_workflow/input_resized.png`
**Purpose:** Style reference for clothing, lighting, background
**Copy to:** `ComfyUI/input/`

#### 3. **Pose Reference** (OPTIONAL - for reference only)
**File:** `output/hug_img_simple_workflow/input_poses.png`
**Purpose:** Shows initial 2-person setup (for your reference)

---

## 🛠️ **ComfyUI Node Setup**

### **Required Models (Download These First)**

1. **Stable Diffusion Model**
   - File: `stable-diffusion-v1-5.ckpt` or similar
   - Location: `ComfyUI/models/checkpoints/`

2. **ControlNet OpenPose Model**
   - File: `control_v11p_sd15_openpose.pth`
   - Location: `ComfyUI/models/controlnet/`
   - Download: https://huggingface.co/lllyasviel/ControlNet-v1-1

---

## 🎬 **ComfyUI Workflow Nodes**

### **Method 1: Process All 16 Frames (Recommended)**

#### **Node Setup:**
```
1. LoadImage (×16) → Load each control frame
   ├── control_000.png
   ├── control_001.png
   ├── ...
   └── control_015.png

2. BatchImages → Combine all 16 frames into batch

3. ControlNetLoader → Load OpenPose model
   └── control_v11p_sd15_openpose.pth

4. ControlNetApply → Apply skeleton control
   ├── Strength: 1.0-1.2
   ├── Start: 0.0
   └── End: 1.0

5. CLIPTextEncode (Positive) → Prompt
   └── "two people hugging, warm embrace, emotional moment, 
        high quality, detailed, realistic, beautiful lighting"

6. CLIPTextEncode (Negative) → Negative prompt
   └── "text, watermark, blurry, low quality, distorted, 
        ugly, deformed, extra limbs"

7. CheckpointLoaderSimple → Load SD model
   └── stable-diffusion-v1-5.ckpt

8. EmptyLatentImage → Set dimensions
   ├── Width: 512
   ├── Height: 512
   └── Batch: 16

9. KSampler → Generate images
   ├── Steps: 25-30
   ├── CFG: 7.5-8.0
   ├── Sampler: euler_ancestral
   ├── Scheduler: normal
   └── Seed: 42 (keep consistent)

10. SaveImage → Save results
    ├── Prefix: "hugging_"
    └── Output: generated frames

11. VideoCombine (Optional) → Create video
    ├── FPS: 8
    └── Output: hugging_video.mp4
```

### **Method 2: Process One Frame at a Time**

**For each frame (000 to 015):**
```
LoadImage → control_XXX.png
    ↓
ControlNetLoader → OpenPose model
    ↓
ControlNetApply → Strength: 1.0
    ↓
CLIPTextEncode → Your prompt
    ↓
CheckpointLoaderSimple → SD model
    ↓
EmptyLatentImage → 512×512, batch: 1
    ↓
KSampler → Generate
    ↓
SaveImage → hugging_XXX.png
```

**Repeat 16 times, changing:**
- Input frame: `control_000.png` → `control_001.png` → ... → `control_015.png`
- Output name: `hugging_000.png` → `hugging_001.png` → ... → `hugging_015.png`
- Seed: Keep same seed (42) for consistent style

---

## ⚙️ **Recommended Settings**

### **ControlNet Settings:**
```
Strength: 1.0-1.2     (Higher = stricter pose following)
Start: 0.0            (Apply from beginning)
End: 1.0              (Apply to end)
```

### **Generation Settings:**
```
Steps: 25-30          (Higher = better quality)
CFG Scale: 7.5-8.0    (Higher = more prompt adherence)
Sampler: euler_ancestral
Scheduler: normal
Seed: 42              (Keep same for all frames)
Width: 512
Height: 512
```

### **Prompt Settings:**
```
Positive: "two people hugging, warm embrace, emotional moment, 
          high quality, detailed, realistic, beautiful lighting, 
          natural skin, proper anatomy"

Negative: "text, watermark, blurry, low quality, distorted, 
          ugly, deformed, extra limbs, bad anatomy, 
          disconnected limbs, floating limbs"
```

---

## 🎨 **Style Customization**

### **Based on Your Input Image:**
If your `input_resized.png` shows:
- **Formal wear** → Add "business attire, formal clothing" to prompt
- **Casual clothes** → Add "casual clothing, everyday wear" to prompt
- **Outdoor setting** → Add "outdoor, natural lighting" to prompt
- **Indoor setting** → Add "indoor, soft lighting" to prompt

### **Prompt Examples:**
```
Romantic: "two people in elegant attire embracing at sunset, 
          romantic atmosphere, golden hour lighting, cinematic"

Family: "family members hugging, warm reunion, emotional moment, 
        natural lighting, heartwarming"

Friends: "friends embracing, joyful reunion, casual clothing, 
         bright lighting, happy moment"
```

---

## 🎥 **Video Creation**

### **After Generating All Frames:**

#### **Method 1: ComfyUI VideoCombine Node**
```
Generated Images → VideoCombine
├── FPS: 8
├── Format: MP4
└── Output: final_hugging_video.mp4
```

#### **Method 2: External FFmpeg**
```bash
ffmpeg -framerate 8 -i hugging_%03d.png -c:v libx264 -pix_fmt yuv420p final_hugging_video.mp4
```

---

## 🔧 **Troubleshooting**

### **Common Issues:**

**1. Poses don't match skeletons**
- Increase ControlNet strength to 1.2-1.5
- Check that control frames are properly loaded

**2. Inconsistent style across frames**
- Use the same seed for all frames
- Keep all settings identical

**3. Poor quality results**
- Increase steps to 30-40
- Increase CFG to 8.0-9.0
- Use a better SD model

**4. Wrong number of people**
- Check that control frames show 2 people
- Add "two people" explicitly in prompt
- Use negative prompt: "single person, one person"

**5. Unnatural motion**
- Lower ControlNet strength to 0.8-1.0
- Add "natural motion, smooth movement" to prompt

---

## 📊 **Expected Results**

**Input:** 16 skeleton control frames + your style image
**Output:** 16 realistic images showing two people hugging
**Final:** MP4 video (2 seconds at 8 fps) of smooth hugging motion

**Quality indicators:**
- People follow skeleton poses exactly
- Consistent clothing/style across frames
- Smooth motion progression
- Natural anatomy and proportions

---

## 🚀 **Quick Start Checklist**

- [ ] Copy 16 control frames to ComfyUI input folder
- [ ] Download ControlNet OpenPose model
- [ ] Set up ComfyUI workflow nodes
- [ ] Configure settings (strength: 1.0, steps: 25, cfg: 7.5)
- [ ] Process all 16 frames
- [ ] Combine into video
- [ ] Review and adjust if needed

**Your skeleton frames are perfectly ready for ComfyUI processing!** 🎬✨
