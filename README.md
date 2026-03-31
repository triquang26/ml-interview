# Cài đặt môi trường

Dự án này yêu cầu Python phiên bản 3.12.3. Dưới đây là hướng dẫn cài đặt môi trường.

## Sử dụng Conda (Khuyến nghị)

1. **Tạo môi trường ảo với Conda:**
   ```bash
   conda create -n ml-interview python=3.12.3 -y
   ```

2. **Kích hoạt môi trường vừa tạo:**
   ```bash
   conda activate ml-interview
   ```

3. **Cài đặt các thư viện phụ thuộc:**
   ```bash
   pip install torch torchvision diffusers accelerate transformers xformers ipykernel controlnet-aux huggingface_hub
   ```

## Sử dụng venv

1. **Đảm bảo bạn đã cài đặt Python 3.12.3 trên máy.**

2. **Tạo môi trường ảo:**
   ```bash
   python3.12 -m venv venv
   ```

3. **Kích hoạt môi trường:**
   - Trên Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
   - Trên Windows:
     ```bash
     venv\Scripts\activate
     ```

4. **Cài đặt các thư viện phụ thuộc:**
   ```bash
   pip install torch torchvision diffusers accelerate transformers xformers ipykernel controlnet-aux huggingface_hub
   ```
# Cách sử dụng repository này
## 1. Tạo dataset prior
```bash
python utils/prior_gen.py
```
## 2. Vào folder [notebooks](./notebooks), chạy các notebook để train model
## 3. Vào folder [outputs/saved_models](./outputs/saved_models), chạy notebook `generate.ipynb` để sinh ra ảnh từ model đã train. Model train xong ở [outputs/saved_models](./outputs/saved_models)
## 4. Chạy `generate.ipynb` thì output sẽ ra ở [outputs/benchmark-report](./outputs/benchmark-report)
- `score_credit.py` là để đánh giá chuẩn theo Dreambooth paper, input vào class và instance, có thể xem trong [datasets/dataset/prompts_and_classes.txt](./datasets/dataset/prompts_and_classes.txt)
## 5. Chạy [utils/score_report.py](./utils/score_report.py) để so sánh instance trong [datasets/dataset/unitree](./datasets/dataset/unitree) với output được generate ra 