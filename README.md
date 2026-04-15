# DreamBooth & Pose Conditioning

## Environment Setup

This project requires **Python 3.12.3**. Follow the instructions below to set up your environment.

### Option 1: Using Conda (Recommended)

1. **Create a virtual environment:**
   ```bash
   conda create -n ml-interview python=3.12.3 -y
   ```

2. **Activate the environment:**
   ```bash
   conda activate ml-interview
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision diffusers accelerate transformers xformers ipykernel controlnet-aux huggingface_hub
   ```

### Option 2: Using `venv`

1. **Ensure Python 3.12.3 is installed on your system.**

2. **Create a virtual environment:**
   ```bash
   python3.12 -m venv venv
   ```

3. **Activate the environment:**
   - On **Linux/macOS**:
     ```bash
     source venv/bin/activate
     ```
   - On **Windows**:
     ```bash
     venv\Scripts\activate
     ```

4. **Install dependencies:**
   ```bash
   pip install torch torchvision diffusers accelerate transformers xformers ipykernel controlnet-aux huggingface_hub
   ```

---

## Usage Guide

Follow these steps to generate data, train models, and evaluate the outputs.

### 1. Generate the Prior Dataset
First, you need to generate the prior dataset required for training.
```bash
python utils/prior_gen.py
```

### 2. Train the Model
Navigate to the [`notebooks`](./notebooks) directory and run the relevant Jupyter notebooks to train your models.

### 3. Generate Images
Once training is complete, the saved models will be located in the [`outputs/saved_models`](./outputs/saved_models) directory. 

Navigate to [`outputs/saved_models`](./outputs/saved_models) and execute the `generate.ipynb` notebook to generate images using your trained models.

### 4. Evaluate the Results
After running `generate.ipynb`, the generated output images will be saved in [`outputs/benchmark-report`](./outputs/benchmark-report).

- **Dreambooth Standard Evaluation**: We use `score_credit.py` to calculate the evaluation metrics following the Dreambooth paper standard. It uses input classes and instances, which can be found in [`datasets/dataset/prompts_and_classes.txt`](./datasets/dataset/prompts_and_classes.txt).

### 5. Generate Scoring Report
Run the evaluation script to compare the target instances located in [`datasets/dataset/unitree`](./datasets/dataset/unitree) with the generated output images.
```bash
python utils/score_report.py
```

---

## Quick Start: AgiBot World Pipeline

### 1. Install Additional Dependencies
```bash
pip install pinocchio imageio opencv-python
```

### 2. Download Dataset
Go to [AgiBotWorld2026 on HuggingFace](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026) and download the dataset. 
Place the data into the `pose-generator/AgiBotWorld2026/` directory matching this exact structure:
- `data/data`
- `data/meta`
- `data/videos`

### 3. Data Preprocessing
Navigate to the AgiBotWorld folder:
```bash
cd pose-generator/AgiBotWorld2026/
```

- **Extract skeleton poses & original frames:**
```bash
python data_preprocess.py
```
**[!] Path configuration note for `data_preprocess.py`:**
- `START_EPISODE` and `END_EPISODE` variables (lines 9, 10): Modify these to limit the dataset generation.
- `BASE_DIR` variable (line 11): Currently hardcoded to the absolute path `/mnt/data/sftp/data/quangpt3/difgit/ml-interview/pose-generator/AgiBotWorld2026`. If you encounter a path error, update this to your correct dataset directory.

- **Generate segmentation masks:**
```bash
python mask_preprocess.py --start <episode_start> --end <episode_end>
```
*(Example: `python mask_preprocess.py --start 0 --end 10`)*

- **Test 3D skeleton to 2D projection:**
```bash
python skeleton.py
```
**[!] Path configuration note for `skeleton.py`:**
- `EPISODE = 31` and `FRAME_IDX = 150` variables (lines 8, 9): Modify these to test a specific frame and episode.
- `BASE_DIR`: Similar to `data_preprocess.py`, ensure this points directly to the `AgiBotWorld2026` folder. The script will output a file named `EXTENDED_DIRECTION.jpg` into the current directory.

### 4. Training (Experiments)
Return to the main project directory:
```bash
cd ../../
```

**[!] Important note on paths for Training scripts:**
All 3 experiment files below contain hardcoded path variables at the beginning of the file. You must review and modify them before training:
- **`src/experiment4.py`**: The `INSTANCE_BASE_DIR` and `PRIOR_BASE_DIR` variables are resolved via a `REPO_ROOT` path. Note that the output directory at `MODEL_SAVE_DIR` is set to `experiment4`.
- **`src/experiment5.py`**: The dataset points to the `dataset-dreambooth-agibot/instance` directory. Output is saved to `experiment5-agibot`.
- **`src/experiment6.py`**: The `DATA_BASE_DIR` path is hardcoded as an absolute string `/mnt/data/sftp/...AgiBotWorld2026/extracted_data` (line 15). Be sure to update this if your folder location changes. The output is hardcoded to the root directory at `outputs/agibot_t2i_adapter`.

- **Experiment 4** (Train resolve followup question 1):
```bash
python src/experiment4.py
```

- **Experiment 5** (Train Dreambooth + T2I-Adapter base):
```bash
python src/experiment5.py
```

- **Experiment 6** (Train T2I-Adapter with new dataset & SDXL freeze):
```bash
python src/experiment6.py
```

### 5. Inference
**[!] Inference Scripts:** 
Before running inference, you **MUST** configure 3 important parameters inside the inference scripts:
1. `POSE_PATHS`: An array containing file paths to the pose images you want to use as inputs (modify it with the images you want to render).
2. `MODEL_SAVE_DIR` / `ADAPTER_PATH`: The directory to load the pretrained model and T2I-Adapter checkpoints from (e.g., `experiment4/unitree`, `agibot_t2i_adapter/adapter_step_2500`...).
3. `OUTPUT_DIR`: The directory to save the generated output images.

- **Inference with AgiBot Model:**
```bash
python src/inference-agibot.py
```

- **Inference with T2I-Adapter for SDXL:**
```bash
python src/inference-t2i-adapter-sdxl.py
```

- **General Inference (ControlNet/T2I-Adapter base):**
```bash
python src/inference.py
```