import torch
from PIL import Image
from diffusers import StableDiffusionAdapterPipeline, T2IAdapter

# ==========================================
# 🎛️ PLAYGROUND SETTINGS
# ==========================================
# Your pre-extracted pose map
POSE_IMAGE_PATH = "/mnt/data/sftp/data/quangpt3/difgit/ml-interview/pose-generator/AgiBotWorld2026/extracted_data/extracted_poses_ep_000000/pose/frame_0103.jpg"

# Standard prompt (no custom 'sks' tokens since we are using base models)
PROMPT = "first-person view of double robotic arms reaching forward"

NUM_STEPS = 30
GUIDANCE_SCALE = 7.5
ADAPTER_SCALE = 1.0 # How strictly the model follows your pose (0.0 to 1.0)
# ==========================================

def run_pretrained_playground():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading your extracted pose image...")
    try:
        pose_image = Image.open(POSE_IMAGE_PATH).convert("RGB")
        # Ensure it matches the SD 1.5 training resolution
        pose_image = pose_image.resize((512, 512), Image.Resampling.LANCZOS)
    except FileNotFoundError:
        print(f"❌ Error: Could not find pose image at {POSE_IMAGE_PATH}")
        return

    print("Downloading/Loading Hugging Face T2I-Adapter...")
    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2iadapter_openpose_sd14v1", 
        torch_dtype=torch.float16
    ).to(device)

    print("Downloading/Loading Base SD 1.5...")
    pipe = StableDiffusionAdapterPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        adapter=adapter,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)

    print(f"\nGenerating image with prompt: '{PROMPT}'")
    
    generated_image = pipe(
        prompt=PROMPT,
        image=pose_image,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        adapter_conditioning_scale=ADAPTER_SCALE,
    ).images[0]

    output_filename = "playground_pretrained_adapter.png"
    generated_image.save(output_filename)
    
    print(f"✅ Inference complete! Saved as '{output_filename}'")

if __name__ == "__main__":
    run_pretrained_playground()