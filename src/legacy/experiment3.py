import os
import gc
import re
import numpy as np
import torch
import torch.nn.functional as F
from itertools import cycle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, T2IAdapter
from transformers import Sam3Processor, Sam3Model
from controlnet_aux import OpenposeDetector
from peft import LoraConfig, get_peft_model

REPO_ROOT = Path().absolute()

DEVICE = "cuda"
LAMBDA = 0.1
NUM_TRAIN = 500
LR = 5e-6
BATCH_SIZE = 1
UNIQUE_TOKEN = "sks"
OPENPOSE_WEIGHT = 0.8
SAM_PROMPT = "a humanoid robot"

INSTANCE_BASE_DIR = str(REPO_ROOT / "datasets" / "dataset")
PRIOR_BASE_DIR = str(REPO_ROOT / "datasets" / "dataset_prior")
MODEL_SAVE_DIR = str(REPO_ROOT / "outputs" / "saved_models" / "our_controlnet_blora")

CLASSES_DICT = {
    'humanoid robot': ['unitree']
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
    def __init__(self, data_dir):
        os.makedirs(data_dir, exist_ok=True)
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
    def __len__(self): 
        return max(1, len(self.image_paths))
        
    def __getitem__(self, idx): 
        if not self.image_paths:
            return Image.new("RGB", (512, 512), (0, 0, 0))
        return Image.open(self.image_paths[idx]).convert("RGB")

class InstancePoseMaskDataset(Dataset):
    def __init__(self, data_dir, det_op):
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.endswith("_mask.png")]
        self.data = []
        for img_path in self.image_paths:
            img = Image.open(img_path).convert("RGB")
            mask_path = img_path.rsplit(".", 1)[0] + "_mask.png"
            mask_img = Image.open(mask_path).convert("L") if os.path.exists(mask_path) else Image.new("L", img.size, 255)
            self.data.append((
                image_transforms(img),
                cond_transforms(det_op(img)),
                mask_transforms(mask_img)
            ))
        
    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, idx): 
        return self.data[idx]

def class_collate_fn(examples):
    return torch.stack([image_transforms(img) if isinstance(img, Image.Image) else img for img in examples])

def instance_collate_fn(examples):
    return (torch.stack([e[0] for e in examples]), torch.stack([e[1] for e in examples]), torch.stack([e[2] for e in examples]))

def prepare_sam_masks(instance_dir):
    missing_masks = [f for f in os.listdir(instance_dir) if f.lower().endswith((".jpg", ".png")) and not f.endswith("_mask.png") and not os.path.exists(os.path.join(instance_dir, f.rsplit(".", 1)[0] + "_mask.png"))]
    if not missing_masks: return

    model = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    for img_name in tqdm(missing_masks):
        img_path = os.path.join(instance_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, text=SAM_PROMPT, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        results = processor.post_process_instance_segmentation(outputs, threshold=0.5, mask_threshold=0.5, target_sizes=inputs.get("original_sizes", torch.tensor([[image.height, image.width]])).tolist())[0]
        
        if len(results["masks"]) > 0:
            best_mask = results["masks"][0].detach().cpu().numpy() if isinstance(results["masks"][0], torch.Tensor) else results["masks"][0]
            mask_np = (np.squeeze(best_mask) > 0).astype(np.uint8) * 255
            Image.fromarray(mask_np).save(os.path.join(instance_dir, img_name.rsplit(".", 1)[0] + "_mask.png"))

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

def train_and_save(class_name, instance):
    instance_dir, class_dir = os.path.join(INSTANCE_BASE_DIR, instance), os.path.join(PRIOR_BASE_DIR, instance)
    save_path = os.path.join(MODEL_SAVE_DIR, instance)
    os.makedirs(save_path, exist_ok=True)
    
    prepare_sam_masks(instance_dir)
    
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32, safety_checker=None).to(DEVICE)
    vae, text_encoder, tokenizer, unet, noise_scheduler = pipeline.vae, pipeline.text_encoder, pipeline.tokenizer, pipeline.unet, pipeline.scheduler
    
    adapter = T2IAdapter.from_pretrained("TencentARC/t2iadapter_openpose_sd14v1", torch_dtype=torch.float32).to(DEVICE).eval()
    detector_openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    
    vae.requires_grad_(False).eval()
    adapter.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()
    unet.requires_grad_(False)
    
    target_modules = r".*up_blocks\.[123].*\.to_(q|k|v|out\.0)"
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
    )
    
    unet = get_peft_model(unet, lora_config)
    unet.train()
    
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LR, weight_decay=1e-4)
    
    inst_loader = DataLoader(InstancePoseMaskDataset(instance_dir, detector_openpose), batch_size=BATCH_SIZE, shuffle=True, collate_fn=instance_collate_fn)
    class_loader = DataLoader(SimpleImageDataset(class_dir), batch_size=BATCH_SIZE, shuffle=True, collate_fn=class_collate_fn)
    inst_iter, class_iter = cycle(inst_loader), cycle(class_loader)

    def tokenize(prompt): return tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(DEVICE)
    inst_tokens, class_tokens = tokenize(f"a {UNIQUE_TOKEN} {class_name}"), tokenize(f"a {class_name}")

    for step in tqdm(range(1, NUM_TRAIN + 1)):
        optimizer.zero_grad()
        
        inst_pixels, inst_poses, inst_masks = next(inst_iter)
        class_pixels = next(class_iter)
        inst_poses, inst_masks = inst_poses.to(DEVICE), inst_masks.to(DEVICE)
        
        with torch.no_grad():
            latents = vae.encode(torch.cat([inst_pixels.to(DEVICE), class_pixels.to(DEVICE)])).latent_dist.sample() * vae.config.scaling_factor
            adapter_states = adapter(torch.cat([inst_poses, torch.zeros_like(inst_poses)], dim=0))
            adapter_states = [OPENPOSE_WEIGHT * state for state in (adapter_states.to_tuple() if not isinstance(adapter_states, (list, tuple)) else adapter_states)]
            prompt_embeds = torch.cat([
                text_encoder(inst_tokens)[0].repeat(inst_pixels.shape[0], 1, 1),
                text_encoder(class_tokens)[0].repeat(class_pixels.shape[0], 1, 1)
            ])

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=DEVICE).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds, down_intrablock_additional_residuals=[s.to(dtype=torch.float32) for s in adapter_states]).sample
        
        noise_pred_inst, noise_pred_prior = noise_pred.chunk(2, dim=0)
        target_inst, target_prior = noise.chunk(2, dim=0)
        
        loss_inst_unreduced = F.mse_loss(noise_pred_inst, target_inst, reduction="none")
        loss_inst = (loss_inst_unreduced * inst_masks).mean()
        
        loss_prior = F.mse_loss(noise_pred_prior, target_prior, reduction="mean")
        
        loss = loss_inst + LAMBDA * loss_prior
        loss.backward()
        optimizer.step()

    unet = unet.merge_and_unload()
    pipeline.unet = unet
    
    pipeline.save_pretrained(save_path)
    
    del optimizer, inst_loader, class_loader, pipeline, adapter, detector_openpose, unet
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    for class_name, instances in CLASSES_DICT.items():
        for instance in instances:
            train_and_save(class_name, instance)