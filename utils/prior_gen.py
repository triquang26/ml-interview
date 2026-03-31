import shutil
import torch
from pathlib import Path
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

BASE_DIR = Path(__file__).resolve().parent.parent
TXT_PATH = BASE_DIR / "datasets" / "dataset" / "prompts_and_classes.txt"
BASE_SAVE_DIR = BASE_DIR / "datasets" / "dataset_prior"

NUM_TRAIN = 1000
TARGET_CLASSES = ["dog", "toy"]
MODEL_ID = "runwayml/stable-diffusion-v1-5"
BATCH_SIZE = 8
NUM_INFERENCE_STEPS = 25

def load_class_to_subjects(txt_path):
    class_to_subjects = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == "subject_name,class":
            start_idx = i + 1
            break

    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split(",")
        if len(parts) != 2:
            continue
            
        subject, class_name = parts
        if subject == "subject_name":
            continue
            
        if class_name not in class_to_subjects:
            class_to_subjects[class_name] = []
            
        if subject not in class_to_subjects[class_name]:
            class_to_subjects[class_name].append(subject)
            
    for c, s in class_to_subjects.items():
        print(f"Class: '{c}', Subjects: {s}")
            
    return class_to_subjects

def setup_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16
    ).to("cuda")
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
        
    pipe.set_progress_bar_config(disable=True)
    return pipe

def main():
    class_to_subjects = load_class_to_subjects(TXT_PATH)
    pipe = setup_pipeline()

    for class_name, subjects in class_to_subjects.items():
        if class_name not in TARGET_CLASSES:
            continue
            
        print(f"\nGenerating {NUM_TRAIN} images for class: {class_name}")
        prompt = f"a {class_name}"
        subject_dirs = []
        
        for subject in subjects:
            subject_dir = BASE_SAVE_DIR / subject
            subject_dir.mkdir(parents=True, exist_ok=True)
            subject_dirs.append(subject_dir)
            
        num_batches = (NUM_TRAIN + BATCH_SIZE - 1) // BATCH_SIZE
        saved_count = 0
        
        for _ in tqdm(range(num_batches), desc=f"Generating {class_name} (Batch={BATCH_SIZE})"):
            current_batch_size = min(BATCH_SIZE, NUM_TRAIN - saved_count)
            prompts = [prompt] * current_batch_size
            
            images = pipe(prompts, num_inference_steps=NUM_INFERENCE_STEPS).images
            
            first_subject_dir = subject_dirs[0]
            for image in images:
                first_save_path = first_subject_dir / f"{class_name}_{saved_count}.png"
                image.save(first_save_path)
                
                for subject_dir in subject_dirs[1:]:
                    other_save_path = subject_dir / f"{class_name}_{saved_count}.png"
                    shutil.copy(first_save_path, other_save_path)
                    
                saved_count += 1

    print("\nPrior dataset generation completed!")

if __name__ == "__main__":
    main()
