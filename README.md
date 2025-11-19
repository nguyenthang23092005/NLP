# Vietnamese Text Summarization with mT5

Dự án fine-tune mô hình mT5 cho tóm tắt văn bản tiếng Việt với LoRA và CPO.

## 📋 Tổng quan

- **Mô hình base**: google/mt5-small
- **Kỹ thuật**: LoRA (Parameter-Efficient Fine-Tuning) + CPO (Contrastive Preference Optimization)
- **Dataset**: Vietnamese news articles (80k+ samples)
- **Framework**: HuggingFace Transformers, PEFT, TRL

## 🚀 Cài đặt

```bash
# Clone repository
git clone <repo-url>
cd NLP

# Cài đặt dependencies
pip install -r requirements.txt

# Kiểm tra GPU
python check_gpu.py
```

## 📊 Chuẩn bị dữ liệu

```bash
# Xử lý raw data thành processed_dataset.json
python data_processing.py

# Tạo train/test/validation splits
python recreate_splits.py
```

Dữ liệu sau khi xử lý:
- `data/processed_dataset.json`: Dữ liệu đã clean
- `data/splits/`: Train/test/validation splits (80/10/10)
- `data/cpo_splits/`: Preference pairs cho CPO training

## 🎯 Training

### 1. Training LoRA (SFT)

```bash
# Full dataset
python train_lora.py

# Quick test với 1000 samples
python train_lora.py --max_samples 1000 --num_train_epochs 3 --output_dir ./models/mt5-lora-1k
```

### 2. Training CPO

```bash
# Train CPO trên model đã fine-tune LoRA
python train_cpo.py --model_path ./models/mt5-lora-full
```

### 3. Two-Stage Training (SFT + CPO)

```bash
# Tự động train 2 giai đoạn
python train_two_stage.py --max_samples 5000 --sft_num_epochs 5 --cpo_num_epochs 3
```

### Tham số quan trọng

| Tham số | SFT | CPO | Mô tả |
|---------|-----|-----|-------|
| `--max_samples` | ✓ | ✓ | Số samples train (None = full) |
| `--num_train_epochs` | ✓ | - | Số epochs cho SFT |
| `--learning_rate` | ✓ | ✓ | Learning rate (1e-4 cho SFT, 5e-5 cho CPO) |
| `--batch_size` | ✓ | ✓ | Batch size per device |
| `--output_dir` | ✓ | ✓ | Thư mục lưu model |

## 📈 Đánh giá

```bash
# Evaluate model trên test set
python evaluate_model.py \
    --model_path ./models/mt5-lora-full/checkpoint-7728 \
    --data_path data/splits \
    --split test \
    --output_file predictions_lora.jsonl

# So sánh nhiều models
python compare_models.py
```

Metrics: ROUGE-1, ROUGE-2, ROUGE-L

## 🌐 Web Interface

```bash
# Chạy Streamlit app
streamlit run app.py
```

Features:
- Upload file (PDF, DOC, DOCX) hoặc nhập text
- Chọn model (LoRA SFT, CPO, DPO, Base mT5)
- Xem và tải kết quả tóm tắt
- Hiển thị thống kê độ dài

## 📁 Cấu trúc thư mục

```
NLP/
├── data/
│   ├── processed_dataset.json      # Dữ liệu đã xử lý
│   ├── splits/                     # Train/test/val splits
│   └── cpo_splits/                 # CPO preference pairs
├── models/
│   ├── mt5-lora-full/             # Model LoRA full dataset
│   ├── mt5-lora-1k/               # Model LoRA 1k samples
│   └── mt5-lora-cpo/              # Model sau CPO
├── train_lora.py                   # Training script LoRA
├── train_cpo.py                    # Training script CPO
├── train_two_stage.py              # 2-stage training
├── evaluate_model.py               # Đánh giá model
├── app.py                          # Streamlit web app
└── data_processing.py              # Xử lý raw data
```

## 🔧 Scripts hỗ trợ

- `check_gpu.py`: Kiểm tra GPU availability và VRAM
- `check_data.py`: Xem thống kê dataset
- `recreate_splits.py`: Tạo lại train/test/val splits
- `visualization.ipynb`: Visualize training metrics

## 💡 Tips

**Để train nhanh với ít dữ liệu:**
```bash
python train_lora.py --max_samples 1000 --num_train_epochs 3 --output_dir ./models/test
```

**Để train full với best performance:**
```bash
python train_two_stage.py --sft_num_epochs 5 --cpo_num_epochs 3
```

**Nếu thiếu VRAM:**
- Giảm `--per_device_train_batch_size` xuống 2 hoặc 1
- Tăng `--gradient_accumulation_steps` lên 8 hoặc 16

## 📝 Tài liệu thêm

- [TRAIN_EXPLAINED.md](TRAIN_EXPLAINED.md): Chi tiết về training process
- [CPO_EXPLAINED.md](CPO_EXPLAINED.md): Giải thích CPO algorithm
- [DPO_EXPLAINED.md](DPO_EXPLAINED.md): Giải thích DPO algorithm
- [QUICKSTART.md](QUICKSTART.md): Hướng dẫn nhanh
- [APP_GUIDE.md](APP_GUIDE.md): Hướng dẫn sử dụng web app

## 📊 Kết quả

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| Base mT5 | - | - | - |
| LoRA SFT | - | - | - |
| LoRA + CPO | - | - | - |

*(Chạy evaluate_model.py để cập nhật)*

## 🤝 Contributing

Pull requests welcome! Hãy đảm bảo code của bạn:
- Follow PEP 8 style guide
- Có docstrings cho functions
- Test trước khi commit

## 📄 License

MIT License