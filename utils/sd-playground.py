import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler,StableDiffusionXLPipeline



INPUT_IMAGE = "/mnt/data/sftp/data/quangpt3/difgit/ml-interview/pose-generator/AgiBotWorld2026/extracted_data/extracted_poses_ep_000000/frame/frame_0586.jpg"
PROMPT = "first-person view of robotic arms reaching forward"
NUM_STEPS = 30
GUIDANCE_SCALE = 7.5
SEED = 42

def run_pure_sdxl():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading SDXL Base Model (this might take a moment)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(device)

    # Euler Ancestral is highly recommended for SDXL (fast and detailed)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    print(f"\nGenerating image...")
    print(f"Prompt: '{PROMPT}'")
    
    # Using a manual seed allows you to reproduce the exact same image later
    generator = torch.Generator(device=device).manual_seed(SEED)

    image = pipe(
        prompt=PROMPT,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator
    ).images[0]

    output_filename = "playground_pure_sdxl.png"
    image.save(output_filename)
    print(f"✅ Done! Image saved successfully as '{output_filename}'")

if __name__ == "__main__":
    run_pure_sdxl()