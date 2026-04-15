import os
import gc
import numpy as np
import torch
import torch.nn.functional as F
import itertools
from itertools import cycle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, T2IAdapter
from transformers import Sam3Processor, Sam3Model
from controlnet_aux import OpenposeDetector

REPO_ROOT = Path().absolute()

DEVICE = "cuda"
BATCH_SIZE = 1
UNIQUE_TOKEN = "sks"
SAM_PROMPT = "first-person view of double robotic arms"

LAMBDA = 1.0                   
NUM_TRAIN = 400                 
LR = 2e-6                       
OPENPOSE_WEIGHT = 1.0           


INSTANCE_BASE_DIR = str(REPO_ROOT / "dataset-dreambooth-agibot" / "instance")
PRIOR_BASE_DIR = str(REPO_ROOT / "datasets" / "dataset_prior")
MODEL_SAVE_DIR = str(REPO_ROOT / "outputs" / "saved_models" / "experiment5-agibot")

CLASSES_DICT = {
    'first-person view of double robotic arms': ['agibot']
}

image_transforms = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

cond_transforms = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])

mask_transforms = transforms.Compose([
    transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

class SimpleImageDataset(Dataset):
    def __init__(self, data_dir, det_op=None):
        os.makedirs(data_dir, exist_ok=True)
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
    def __len__(self): 
        return max(1, len(self.image_paths))
        
    def __getitem__(self, idx): 
        if not self.image_paths:
            img = Image.new("RGB", (512, 512), (0, 0, 0))
            return image_transforms(img), cond_transforms(img)
            
        img = Image.open(self.image_paths[idx]).convert("RGB")
        blank_pose = Image.new("RGB", (512, 512), (0, 0, 0))
        return image_transforms(img), cond_transforms(blank_pose)

class InstancePoseMaskDataset(Dataset):
    def __init__(self, data_dir, det_op=None):
        images_dir = os.path.join(data_dir, "images")
        poses_dir = os.path.join(data_dir, "pose")
        masks_dir = os.path.join(data_dir, "mask")
        
        self.image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.data = []
        for img_path in self.image_paths:
            img_name = os.path.basename(img_path)
            img = Image.open(img_path).convert("RGB")
            
            pose_path = os.path.join(poses_dir, img_name)
            mask_path = os.path.join(masks_dir, img_name)
            
            if os.path.exists(pose_path) and os.path.exists(mask_path):
                pose_img = Image.open(pose_path).convert("RGB")
                mask_img = Image.open(mask_path).convert("L")
                
                self.data.append((
                    image_transforms(img),
                    cond_transforms(pose_img),
                    mask_transforms(mask_img)
                ))
        
    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, idx): 
        return self.data[idx]

def class_collate_fn(examples):
    return torch.stack([e[0] for e in examples]), torch.stack([e[1] for e in examples])

def instance_collate_fn(examples):
    return torch.stack([e[0] for e in examples]), torch.stack([e[1] for e in examples]), torch.stack([e[2] for e in examples])

def prepare_sam_masks(instance_dir):
    pass

def train_and_save(class_name, instance):
    instance_dir = INSTANCE_BASE_DIR 
    class_dir = os.path.join(PRIOR_BASE_DIR, instance) 
    save_path = os.path.join(MODEL_SAVE_DIR, instance)
    os.makedirs(save_path, exist_ok=True)
    
    instance_prompt = f"a {UNIQUE_TOKEN} {class_name}"
    class_prompt = f"a {class_name}"
    
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32, safety_checker=None).to(DEVICE)
    vae, text_encoder, tokenizer, unet, noise_scheduler = pipeline.vae, pipeline.text_encoder, pipeline.tokenizer, pipeline.unet, pipeline.scheduler
    
    unet.enable_gradient_checkpointing()
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except:
        pass
    
    adapter = T2IAdapter.from_pretrained("TencentARC/t2iadapter_openpose_sd14v1", torch_dtype=torch.float32).to(DEVICE)
    detector_openpose = None
    
    vae.requires_grad_(False).eval()
    adapter.requires_grad_(True).train()
    text_encoder.requires_grad_(True).train()
    unet.requires_grad_(True).train()
    
    optimizer = torch.optim.AdamW(
        itertools.chain(unet.parameters(), text_encoder.parameters(), adapter.parameters()), 
        lr=LR, weight_decay=1e-2
    )
    
    inst_loader = DataLoader(InstancePoseMaskDataset(instance_dir, detector_openpose), batch_size=BATCH_SIZE, shuffle=True, collate_fn=instance_collate_fn)
    class_loader = DataLoader(SimpleImageDataset(class_dir, detector_openpose), batch_size=BATCH_SIZE, shuffle=True, collate_fn=class_collate_fn)
    inst_iter, class_iter = cycle(inst_loader), cycle(class_loader)

    def tokenize(prompt): 
        return tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(DEVICE)
    
    inst_tokens, class_tokens = tokenize(instance_prompt), tokenize(class_prompt)

    for step in tqdm(range(1, NUM_TRAIN + 1), desc=f"Training {instance}"):
        optimizer.zero_grad()
        
        inst_pixels, inst_poses, inst_masks = next(inst_iter)
        class_pixels, class_poses = next(class_iter)
        
        inst_poses, inst_masks = inst_poses.to(DEVICE), inst_masks.to(DEVICE)
        class_poses = class_poses.to(DEVICE)
        
        with torch.no_grad():
            latents = vae.encode(torch.cat([inst_pixels.to(DEVICE), class_pixels.to(DEVICE)])).latent_dist.sample() * vae.config.scaling_factor
            
        adapter_cond_pose = torch.cat([inst_poses, class_poses], dim=0)
        adapter_states = adapter(adapter_cond_pose)
        if not isinstance(adapter_states, (list, tuple)): 
            adapter_states = adapter_states.to_tuple()
        adapter_states = [OPENPOSE_WEIGHT * state for state in adapter_states]

        prompt_embeds = torch.cat([text_encoder(inst_tokens)[0], text_encoder(class_tokens)[0]])
        
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=DEVICE).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        noise_pred = unet(
            noisy_latents, timesteps, encoder_hidden_states=prompt_embeds,
            down_intrablock_additional_residuals=[sample.to(dtype=torch.float32) for sample in adapter_states]
        ).sample
        
        noise_pred_inst, noise_pred_prior = noise_pred.chunk(2, dim=0)
        target_inst, target_prior = noise.chunk(2, dim=0)
        
        loss_inst_unreduced = F.mse_loss(noise_pred_inst, target_inst, reduction="none")
        weighted_mask = (inst_masks * 2.0) + 0.05 * (1.0 - inst_masks)
        weighted_mask = weighted_mask.clamp(0.1, 2.0)
        loss_inst = (loss_inst_unreduced * weighted_mask).mean()
        
        loss_prior = F.mse_loss(noise_pred_prior, target_prior, reduction="mean")
        loss = loss_inst + LAMBDA * loss_prior
        
        loss.backward()
        optimizer.step()

    # --- CRITICAL FIX: MEMORY CLEANUP BEFORE SAVING ---
    # 1. Delete optimizer and clear VRAM immediately to free ~8GB+ memory
    del optimizer
    torch.cuda.empty_cache()
    gc.collect()

    # 2. Move models to CPU to prevent RAM/VRAM overlap spikes during serialization
    pipeline.to("cpu")
    adapter.to("cpu")
    torch.cuda.empty_cache()
    
    # 3. Save the models using safe_serialization
    pipeline.save_pretrained(save_path, safe_serialization=True)
    adapter.save_pretrained(os.path.join(save_path, "adapter"), safe_serialization=True)
    
    # 4. Final cleanup
    del inst_loader, class_loader, inst_iter, class_iter
    del pipeline, adapter
    gc.collect()

if __name__ == "__main__":
    for class_name, instances in CLASSES_DICT.items():
        for instance in instances:
            train_and_save(class_name, instance)