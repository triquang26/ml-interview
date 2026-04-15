import os
import torch
from PIL import Image
from diffusers import T2IAdapter, StableDiffusionAdapterPipeline, UniPCMultistepScheduler, UNet2DConditionModel
from transformers import CLIPTextModel
from controlnet_aux import OpenposeDetector
from pathlib import Path

DEVICE = "cuda"
REPO_ROOT = Path().absolute()
MODEL_SAVE_DIR = "/mnt/data/sftp/data/quangpt3/difgit/ml-interview/outputs/saved_models/experiment4/unitree"
OUTPUT_DIR = str(REPO_ROOT / "outputs" / "benchmark-report" / "experiment4")
UNIQUE_TOKEN = "sks"
CLASS_NAME = "humanoid robot"
POSE_PATHS = [
    str(REPO_ROOT / "datasets" / "posing" / "posing-01.jpg"),
    str(REPO_ROOT / "datasets" / "posing" / "posing-04.jpg")
]


txt_path = REPO_ROOT / "datasets" / "dataset" / "prompts_and_classes.txt"
lines = txt_path.read_text().splitlines()

PROMPT_LIST = []
start_idx = lines.index("Object Prompts")
for line in lines[start_idx:]:
    if line.startswith("'"):
        prompt = line.split("'")[1]
        PROMPT_LIST.append(prompt)
    elif "]" in line:
        break

os.makedirs(OUTPUT_DIR, exist_ok=True)

detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
trained_unet = UNet2DConditionModel.from_pretrained(f"{MODEL_SAVE_DIR}/unet", torch_dtype=torch.float16)
trained_text_encoder = CLIPTextModel.from_pretrained(f"{MODEL_SAVE_DIR}/text_encoder", torch_dtype=torch.float16)
adapter = T2IAdapter.from_pretrained("TencentARC/t2iadapter_openpose_sd14v1", torch_dtype=torch.float16)

pipe = StableDiffusionAdapterPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    adapter=adapter,
    unet=trained_unet,
    text_encoder=trained_text_encoder,
    torch_dtype=torch.float16,
    safety_checker=None
).to(DEVICE)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

for pose_idx, pose_path in enumerate(POSE_PATHS, start=1):
    pose_dir = os.path.join(OUTPUT_DIR, f"pose{pose_idx}")
    os.makedirs(pose_dir, exist_ok=True)
    
    original_image = Image.open(pose_path).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
    pose_map = detector(original_image)
    pose_map.save(os.path.join(pose_dir, "input_pose_visualized.png"))
    
    for prompt_idx, base_prompt in enumerate(PROMPT_LIST):
        prompt = base_prompt.format(UNIQUE_TOKEN, CLASS_NAME)
        
        generator = torch.Generator(DEVICE).manual_seed(42)
        
        images = pipe(
            prompt=prompt,
            image=pose_map,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=512,
            width=512,
            adapter_conditioning_scale=0.8,
            num_images_per_prompt=4,
            generator=generator
        ).images
        
        safe_name = base_prompt.replace(" ", "_").replace("{0}", "").replace("{1}", "")[:30].strip("_")
        for img_idx, image in enumerate(images):
            image.save(os.path.join(pose_dir, f"{prompt_idx:02d}_{safe_name}_{img_idx}.png"))