import torch
import numpy as np
from PIL import Image
from transformers import Sam3Model, Sam3Processor
import matplotlib.pyplot as plt

# ==========================================
# 🎛️ PLAYGROUND SETTINGS
# ==========================================
IMAGE_PATH = "/mnt/data/sftp/data/quangpt3/difgit/ml-interview/pose-generator/AgiBotWorld2026/extracted_data/extracted_poses_ep_000000/frame/frame_0586.jpg" # Replace with your ego-centric image
# Test literal, physical descriptions here:
TEXT_PROMPT = "first-person view of double robotic arms" 
MASK_THRESHOLD = 0.7
# ==========================================

def run_sam_playground():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading SAM 3...")
    
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)

    # Load and process image
    image = Image.open(IMAGE_PATH).convert("RGB")
    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)

    print(f"Segmenting targeting: '{TEXT_PROMPT}'...")
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract mask
    results = processor.post_process_instance_segmentation(
        outputs, 
        threshold=0.5, 
        mask_threshold=MASK_THRESHOLD,
        target_sizes=[(image.height, image.width)]
    )[0]

    # Visualization
    if len(results["masks"]) > 0:
        mask = results["masks"][0].cpu().numpy().squeeze()
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(image)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f"SAM Mask: '{TEXT_PROMPT}'")
        plt.imshow(image)
        plt.imshow(mask, cmap="jet", alpha=0.5) # Overlay mask in semi-transparent color
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("sam_output.png", bbox_inches='tight')
        print("Saved SAM output to 'sam_output.png'")
        # plt.show()
    else:
        print(f"⚠️ SAM could not find anything matching the prompt: '{TEXT_PROMPT}'")

if __name__ == "__main__":
    run_sam_playground()