import os
import argparse
import glob
import torch
import numpy as np
from PIL import Image
from transformers import Sam3Model, Sam3Processor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = "/mnt/data/sftp/data/quangpt3/difgit/ml-interview/pose-generator/AgiBotWorld2026/extracted_data"
TEXT_PROMPT = "first-person view of double robotic arms"
MASK_THRESHOLD = 0.7

class FrameDataset(Dataset):
    def __init__(self, frame_paths, mask_paths):
        self.frame_paths = frame_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(frame_path).convert("RGB")
        return image, frame_path, mask_path

def custom_collate(batch):
    return batch[0]

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    model.eval()

    frame_paths = []
    mask_paths = []
    
    for ep_id in range(args.start, args.end + 1):
        ep_folder = os.path.join(DATA_DIR, f"extracted_poses_ep_{ep_id:06d}")
        if not os.path.isdir(ep_folder):
            continue
            
        frame_dir = os.path.join(ep_folder, "frame")
        mask_dir = os.path.join(ep_folder, "mask")
        os.makedirs(mask_dir, exist_ok=True)

        frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
        for p in frames:
            name = os.path.basename(p)
            mp = os.path.join(mask_dir, name)
            if not os.path.exists(mp):
                frame_paths.append(p)
                mask_paths.append(mp)
                
    dataset = FrameDataset(frame_paths, mask_paths)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, collate_fn=custom_collate)

    with ThreadPoolExecutor(max_workers=8) as executor:
        for image, frame_path, mask_path in tqdm(dataloader, total=len(dataset)):
            try:
                inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                w, h = image.width, image.height
                results = processor.post_process_instance_segmentation(
                    outputs, threshold=0.5, mask_threshold=MASK_THRESHOLD, target_sizes=[(h, w)]
                )[0]
    
                if len(results["masks"]) > 0:
                    mask = results["masks"][0].cpu().numpy().squeeze()
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                else:
                    mask_img = Image.new('L', (w, h), color=0)
                    
                # Giao việc lưu ảnh cho background thread
                executor.submit(mask_img.save, mask_path)
                
            except Exception as e:
                print(f"Error processing {frame_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()
    main(args)
