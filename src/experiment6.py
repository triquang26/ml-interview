import os
import sys
import glob
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, T2IAdapter, DDPMScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

MAX_VRAM_GB = 35
DATA_BASE_DIR = "/mnt/data/sftp/data/quangpt3/difgit/ml-interview/pose-generator/AgiBotWorld2026/extracted_data"
OUTPUT_DIR = "outputs/agibot_t2i_adapter"
STATIC_PROMPT = "first-person view of double robotic arms"
RESOLUTION = 1024

BATCH_SIZE = 2
ACCUMULATION_STEPS = 1

LEARNING_RATE = 3e-5
MAX_TRAIN_STEPS = 2500
SAVE_EVERY = 1250
DEVICE = "cuda"
WEIGHT_DTYPE = torch.bfloat16

class AgiBotPoseDataset(Dataset):
    def __init__(self, base_dir, resolution):
        self.pairs = []
        episode_dirs = glob.glob(os.path.join(base_dir, "extracted_poses_ep_*"))
        for ep_dir in episode_dirs:
            frames_dir = os.path.join(ep_dir, "frame")
            poses_dir = os.path.join(ep_dir, "pose")
            if not os.path.exists(frames_dir) or not os.path.exists(poses_dir): continue
            frame_files = sorted(os.listdir(frames_dir))
            for f_name in frame_files:
                if f_name.endswith((".jpg", ".png")):
                    frame_path = os.path.join(frames_dir, f_name)
                    pose_path = os.path.join(poses_dir, f_name)
                    if os.path.exists(pose_path):
                        self.pairs.append((frame_path, pose_path))
        self.image_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.pose_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        frame_img = Image.open(self.pairs[idx][0]).convert("RGB")
        pose_img = Image.open(self.pairs[idx][1]).convert("RGB")
        return self.image_transforms(frame_img), self.pose_transforms(pose_img)

def encode_prompt(prompt, tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2):
    with torch.no_grad():
        text_inputs_1 = tokenizer_1(prompt, padding="max_length", max_length=tokenizer_1.model_max_length, truncation=True, return_tensors="pt").input_ids.to(DEVICE)
        text_inputs_2 = tokenizer_2(prompt, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids.to(DEVICE)
        prompt_embeds_1 = text_encoder_1(text_inputs_1, output_hidden_states=True)
        prompt_embeds_2 = text_encoder_2(text_inputs_2, output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds_2[0]
        prompt_embeds = torch.cat([prompt_embeds_1.hidden_states[-2], prompt_embeds_2.hidden_states[-2]], dim=-1)
    return prompt_embeds, pooled_prompt_embeds

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading Text Encoders...")
    tokenizer_1 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2")
    text_encoder_1 = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", torch_dtype=WEIGHT_DTYPE, variant="fp16").to(DEVICE)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2", torch_dtype=WEIGHT_DTYPE, variant="fp16").to(DEVICE)
    
    print("Encoding static prompt...")
    prompt_embeds, pooled_prompt_embeds = encode_prompt(STATIC_PROMPT, tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2)
    
    print("Freeing Text Encoders from memory...")
    del tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Loading VAE, UNet, and Adapter...")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=WEIGHT_DTYPE).to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", torch_dtype=WEIGHT_DTYPE, variant="fp16").to(DEVICE)
    noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
    adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-openpose-sdxl-1.0", torch_dtype=torch.float32).to(DEVICE)
    
    vae.eval()
    unet.eval()
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    vae.enable_slicing()
    
    adapter.requires_grad_(True)
    adapter.train()
    
    unet.enable_gradient_checkpointing()
    try:
        unet.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(f"Xformers not enabled: {e}")
        pass
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    dataset = AgiBotPoseDataset(DATA_BASE_DIR, RESOLUTION)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    
    add_time_ids = torch.tensor([[RESOLUTION, RESOLUTION, 0, 0, RESOLUTION, RESOLUTION]], dtype=WEIGHT_DTYPE).to(DEVICE)
    
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    
    for epoch in range(10): 
        for step, (pixels, poses) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            pixels = pixels.to(DEVICE, dtype=WEIGHT_DTYPE)
            poses = poses.to(DEVICE, dtype=torch.float32) 
            
            with torch.no_grad():
                latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
            
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=DEVICE).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            with torch.autocast("cuda", dtype=WEIGHT_DTYPE):
                adapter_state = adapter(poses)
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds.expand(bsz, -1, -1),
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds.expand(bsz, -1),
                        "time_ids": add_time_ids.expand(bsz, -1)
                    },
                    down_intrablock_additional_residuals=[state for state in adapter_state]
                ).sample
                
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                loss = loss / ACCUMULATION_STEPS
                
            loss.backward()
            
            if (step + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            
                if global_step % 100 == 0:
                    print(f"Step: {global_step}/{MAX_TRAIN_STEPS} | Loss: {(loss.item() * ACCUMULATION_STEPS):.4f}")
                
                if global_step % SAVE_EVERY == 0 or global_step == MAX_TRAIN_STEPS:
                    del latents, noise, noise_pred, noisy_latents, adapter_state
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    print(f"Saving checkpoint at step {global_step}...")
                    save_path = os.path.join(OUTPUT_DIR, f"adapter_step_{global_step}")
                    
                    adapter.to("cpu")
                    adapter.save_pretrained(save_path, safe_serialization=True)
                    adapter.to(DEVICE)
                    
                if global_step >= MAX_TRAIN_STEPS:
                    print("Training complete!")
                    sys.exit(0)

            del pixels, poses, loss
            if 'latents' in locals(): del latents, noise, noisy_latents, adapter_state, noise_pred
            

            if step % 20 == 0:
                gc.collect()

if __name__ == "__main__":
    main()