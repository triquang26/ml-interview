import random
import os
import glob
import torch
from PIL import Image
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler

DEVICE = "cuda"
ADAPTER_PATH = "/mnt/data/sftp/data/quangpt3/difgit/ml-interview/outputs/agibot_t2i_adapter/adapter_step_2500"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DATA_DIR = "/mnt/data/sftp/data/quangpt3/difgit/ml-interview/pose-generator/AgiBotWorld2026/extracted_data"
OUTPUT_DIR = "outputs/inference_contexts"

def get_sample_poses(num=10):
    """Finds multiple pose images from the extracted dataset."""
    pose_files = glob.glob(os.path.join(DATA_DIR, "extracted_poses_ep_*", "pose", "*.jpg"))
    if not pose_files:
        pose_files = glob.glob(os.path.join(DATA_DIR, "extracted_poses_ep_*", "pose", "*.png"))
    
    if not pose_files:
        raise FileNotFoundError("Could not find any pose images in the dataset directory.")
    
    # Randomize to get diverse poses
    random.shuffle(pose_files)
    return pose_files[:num]

def main():
    
    print("Loading specifically trained T2I Adapter...")
    adapter = T2IAdapter.from_pretrained(
        ADAPTER_PATH,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    
    print("Loading SDXL Adapter Pipeline...")
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        BASE_MODEL,
        adapter=adapter,
        torch_dtype=torch.bfloat16,
        variant="fp16",
    ).to(DEVICE)
    
    # Use Euler Ancestral which works great for SDXL
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("Xformers enabled.")
    except Exception as e:
        print(f"Xformers not available: {e}")
        pass
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 10 Context Prompts defined
    contexts = [
        "first-person view of double robotic arms",
        "first-person view of double robotic arms in a modern clean kitchen",
        "first-person view of double robotic arms in an industrial automotive factory",
        "first-person view of double robotic arms in a futuristic sci-fi laboratory",
        "first-person view of double robotic arms operating on a wooden desk",
        "first-person view of double robotic arms outdoors in a sunny green park",
        "first-person view of double robotic arms inside a busy logistics warehouse",
        "first-person view of double robotic arms underwater in a research submarine",
        "first-person view of double robotic arms in a bright hospital operating room",
        "first-person view of double robotic arms under neon cyberpunk city lights"
    ]
        
    pose_img_paths = get_sample_poses(num=len(contexts))
    
    for i, (prompt, pose_img_path) in enumerate(zip(contexts, pose_img_paths)):
        print(f"\n[{i+1}/{len(contexts)}] Loading pose from: {pose_img_path}")
        pose_image = Image.open(pose_img_path).convert("RGB")
        
        # SDXL natively trains at 1024x1024
        pose_image = pose_image.resize((1024, 1024), Image.Resampling.BILINEAR)
    
        print(f"Generating image with context prompt: '{prompt}'")
        
        image = pipe(
            prompt=prompt,
            image=pose_image,
            num_inference_steps=50,
            adapter_conditioning_scale=1.0,
            adapter_conditioning_factor=1.0,
            guidance_scale=7.5
        ).images[0]
        
        out_path = os.path.join(OUTPUT_DIR, f"context_render_{i+1:02d}.jpg")
        image.save(out_path)
        print(f"-> Saved context render to {out_path}")
        
    print("\nBatch generation complete!")

if __name__ == "__main__":
    main()
