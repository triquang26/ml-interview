import os
import glob
import random
import shutil
import numpy as np
from PIL import Image

INPUT_DIR = "/mnt/data/sftp/data/quangpt3/difgit/ml-interview/pose-generator/AgiBotWorld2026/extracted_data"
OUTPUT_BASE = "/mnt/data/sftp/data/quangpt3/difgit/ml-interview/dataset-dreambooth-agibot/instance"
NUM_SAMPLES = 50

def parse_all_triplets():
    triplets = []
    # Find all episode directories
    eps = glob.glob(os.path.join(INPUT_DIR, "extracted_poses_ep_*"))
    for ep in eps:
        frames = glob.glob(os.path.join(ep, "frame", "*.jpg"))
        for frame_path in frames:
            frame_name = os.path.basename(frame_path)
            mask_path = os.path.join(ep, "mask", frame_name)
            pose_path = os.path.join(ep, "pose", frame_name)
            
            # Check if all parts of the triplet exist
            if os.path.exists(mask_path) and os.path.exists(pose_path):
                triplets.append((frame_path, pose_path, mask_path))
                
    return triplets

def is_valid_mask(mask_path):
    try:
        # Open mask and convert to grayscale just in case
        img = Image.open(mask_path).convert('L')
        # extrema returns (min, max) value. If max is 0, it's completely black.
        return img.getextrema()[1] > 0
    except:
        return False

def main():
    triplets = parse_all_triplets()
    print(f"Found {len(triplets)} valid triplets (image, pose, mask).")
    
    if len(triplets) == 0:
        print("No complete triplets found! Make sure frames, poses, and masks exist.")
        return
        
    # Xáo trộn mảng và đi tìm đủ 50 mask hợp lệ
    random.shuffle(triplets)
    valid_sampled = []
    
    print("Filtering out black masks...")
    for triplet in triplets:
        if is_valid_mask(triplet[2]): # Check path của mask
            valid_sampled.append(triplet)
        
        # Đã gom đủ 50 ảnh đẹp
        if len(valid_sampled) == NUM_SAMPLES:
            break
            
    if len(valid_sampled) < NUM_SAMPLES:
        print(f"Warning: Only found {len(valid_sampled)} valid mask triplets.")
        
    img_out = os.path.join(OUTPUT_BASE, "images")
    pose_out = os.path.join(OUTPUT_BASE, "pose")
    mask_out = os.path.join(OUTPUT_BASE, "mask")
    
    # Xoá folder cũ đi cho an toàn nếu bạn sample lại
    shutil.rmtree(img_out, ignore_errors=True)
    shutil.rmtree(pose_out, ignore_errors=True)
    shutil.rmtree(mask_out, ignore_errors=True)
    
    # Create directories if they don't exist
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(pose_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)
    
    # Copy files and rename them from 1 to 50
    for i, (f_path, p_path, m_path) in enumerate(valid_sampled, start=1):
        filename = f"{i:02d}.jpg" # Pad with zero for correct sorting e.g. 01.jpg, 02.jpg
        
        shutil.copy2(f_path, os.path.join(img_out, filename))
        shutil.copy2(p_path, os.path.join(pose_out, filename))
        shutil.copy2(m_path, os.path.join(mask_out, filename))
        
    print(f"✅ Successfully copied and renamed {len(valid_sampled)} VALID triplets to '{OUTPUT_BASE}'")

if __name__ == "__main__":
    main()
