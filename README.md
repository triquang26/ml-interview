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