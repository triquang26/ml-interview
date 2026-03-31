import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTModel, CLIPProcessor, CLIPModel
from huggingface_hub import login
from tqdm import tqdm
from pathlib import Path

login(token="hf_")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UNIQUE_TOKEN = "sks"
CLASS_NAME = "humanoid robot"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REAL_IMG_DIR = BASE_DIR / "dataset" / "dreambooth" / "dataset" / "unitree"
BENCHMARK_REPORT_DIR = BASE_DIR / "ml-interview" / "outputs" / "benchmark-report"
GEN_BASE_DIR = BENCHMARK_REPORT_DIR / "our_methods_loss_mask_controlnet_08"
POSE_FOLDERS = ["pose1", "pose2"]

PROMPT_LIST = [
    'a {0} {1} in the jungle', 'a {0} {1} in the snow', 'a {0} {1} on the beach',
    'a {0} {1} on a cobblestone street', 'a {0} {1} on top of pink fabric',
    'a {0} {1} on top of a wooden floor', 'a {0} {1} with a city in the background',
    'a {0} {1} with a mountain in the background', 'a {0} {1} with a blue house in the background',
    'a {0} {1} on top of a purple rug in a forest', 'a {0} {1} with a wheat field in the background',
    'a {0} {1} with a tree and autumn leaves in the background', 'a {0} {1} with the Eiffel Tower in the background',
    'a {0} {1} floating on top of water', 'a {0} {1} floating in an ocean of milk',
    'a {0} {1} on top of green grass with sunflowers around it', 'a {0} {1} on top of a mirror',
    'a {0} {1} on top of the sidewalk in a crowded street', 'a {0} {1} on top of a dirt road',
    'a {0} {1} on top of a white rug', 'a red {0} {1}', 'a purple {0} {1}',
    'a shiny {0} {1}', 'a wet {0} {1}', 'a cube shaped {0} {1}'
]

dino_processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
dino_model = ViTModel.from_pretrained('facebook/dino-vits16').to(DEVICE).eval()

clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(DEVICE).eval()

@torch.no_grad()
def get_dino_embeddings(images):
    inputs = dino_processor(images=images, return_tensors="pt").to(DEVICE)
    outputs = dino_model(**inputs)
    return F.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=-1)

@torch.no_grad()
def get_clip_image_embeddings(images):
    inputs = clip_processor(images=images, return_tensors="pt").to(DEVICE)
    vision_outputs = clip_model.vision_model(pixel_values=inputs.pixel_values)
    image_embeds = vision_outputs.pooler_output if hasattr(vision_outputs, "pooler_output") else vision_outputs[1]
    return F.normalize(clip_model.visual_projection(image_embeds), p=2, dim=-1)

@torch.no_grad()
def get_clip_text_embeddings(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
    text_outputs = clip_model.text_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    text_embeds = text_outputs.pooler_output if hasattr(text_outputs, "pooler_output") else text_outputs[1]
    return F.normalize(clip_model.text_projection(text_embeds), p=2, dim=-1)

def evaluate_poses():
    real_paths = [p for p in REAL_IMG_DIR.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
    real_imgs = [Image.open(p).convert('RGB') for p in real_paths]
    
    real_dino_embs = get_dino_embeddings(real_imgs)
    real_clip_embs = get_clip_image_embeddings(real_imgs)

    for pose in POSE_FOLDERS:
        gen_dir = GEN_BASE_DIR / pose
        if not gen_dir.exists():
            continue
            
        dino_scores, clip_i_scores, clip_t_scores = [], [], []

        for i, base_prompt in enumerate(tqdm(PROMPT_LIST, desc=pose)):
            prompt = base_prompt.format(UNIQUE_TOKEN, CLASS_NAME)
            txt_emb = get_clip_text_embeddings(prompt)
            safe_name = base_prompt.replace(" ", "_").replace("{0}", "").replace("{1}", "")[:30].strip("_")
            
            for j in range(4):
                img_path = gen_dir / f"{i:02d}_{safe_name}_{j}.png"
                if not img_path.exists():
                    continue
                    
                gen_img = Image.open(img_path).convert('RGB')
                
                gen_dino_emb = get_dino_embeddings([gen_img])
                gen_clip_emb = get_clip_image_embeddings([gen_img])

                dino_sim = F.cosine_similarity(gen_dino_emb, real_dino_embs, dim=-1).mean().item()
                clip_i_sim = F.cosine_similarity(gen_clip_emb, real_clip_embs, dim=-1).mean().item()
                clip_t_sim = F.cosine_similarity(gen_clip_emb, txt_emb, dim=-1).item()

                dino_scores.append(dino_sim)
                clip_i_scores.append(clip_i_sim)
                clip_t_scores.append(clip_t_sim)

        if dino_scores:
            print(f"\n--- {pose.upper()} ---")
            print(f"DINO: {np.mean(dino_scores):.4f}")
            print(f"CLIP-I: {np.mean(clip_i_scores):.4f}")
            print(f"CLIP-T: {np.mean(clip_t_scores):.4f}")
            print("-" * 20)

if __name__ == "__main__":
    evaluate_poses()