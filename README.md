# Bài tập lớn NLP: Xây dựng mô hình ngôn ngữ tóm tắt văn bản tiếng Việt

## 📌 Môn học
**Xử lý Ngôn ngữ Tự nhiên (NLP)**  
**Nhóm:** 25  
**Giảng viên:** PGS.TS. Phạm Tiến Lâm

**Đề tài:**  
**Xây dựng mô hình ngôn ngữ tóm tắt văn bản tiếng Việt bằng mT5 + LoRA + CPO**

---

## 👥 Thành viên nhóm
| STT | Họ và tên | MSSV | Vai trò |
|-----|-----------|-------|---------|
| 1 | **Nguyễn Văn Thăng** (Nhóm trưởng) | 23010572 | Xử lý dữ liệu, nghiên cứu mô hình, fine-tune, đánh giá, triển khai |
| 2 | **Phạm Văn Sự** | 23010523 | Khảo sát phương pháp, xây dựng mô hình, đánh giá & viết báo cáo |
| 3 | **Đặng Anh Tuyền** | 23010912 | Thu thập dữ liệu, xây dựng mô hình, đánh giá |
| 4 | **Nguyễn Thị Nhung** | 23010607 | Thu thập dữ liệu, xử lý, xây dựng mô hình, đánh giá |

---

# 🚀 1. Giới thiệu đề tài
Mục tiêu đề tài:
- Xây dựng hệ thống **tóm tắt bài báo tiếng Việt** tự động.  
- Sử dụng mô hình **mT5-small**.  
- Tối ưu chi phí train bằng **LoRA**.  
- Tăng chất lượng sinh văn bản bằng **CPO**.  
- Đánh giá bằng **ROUGE, BLEU, METEOR**.  
- Xây dựng giao diện demo.

---

# 🧠 2. Pipeline tổng quan
```
RAW DATA
   │
   ├── 1. Thu thập dữ liệu từ 3 bộ:
   │       VietNews, VNONews, NewsDatasetVN
   │
   ├── 2. Tiền xử lý:
   │       - Loại ký tự nhiễu  
   │       - Chuẩn hóa Unicode  
   │       - Lowercase  
   │       - Chuẩn hóa khoảng trắng  
   │       - Loại mẫu trống
   │
   ├── 3. Chia dữ liệu:
   │       train (80%) - val (10%) - test (10%)
   │
   ├── 4. Giai đoạn 1:
   │       Fine-tune mT5 + LoRA
   │       - ROUGE evaluation
   │
   ├── 5. Giai đoạn 2:
   │       Huấn luyện CPO
   │       - ROUGE evaluation
   │
   ├── 6. Đánh giá:
   │       ROUGE-1/2/L, BLEU, METEOR
   │
   └── 7. Triển khai:
           - checkpoint stage1 / stage2
           - giao diện web
```

---

# 📂 3. Cấu trúc thư mục dự án
```
📦 NLP-Summarization
.
├── data
│   ├── cpo_splits/
│   ├── splits/
│   └── processed_dataset.json
├── data_raw
│   ├── News_Dataset_Vietnamese.json
│   ├── Vietnamese_Online_News_Dataset.json
│   └── vietnews.json
├── metrics
│   ├── predictions_lora_cpo_metrics.json
│   ├── predictions_lora_metrics.json
│   └── predictions_mt5small_metrics.json
├── models
│   ├── mt5-cpo/
│   ├── mt5-cpo-full/
│   ├── mt5-lora-full/
│   ├── mt5-lora-v2/
│   └── mt5-small/
│
├── pred
│   ├── predictions_lora_cpo.jsonl
│   ├── predictions_lora.jsonl
│   └── predictions_mt5small.jsonl
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── app.py
├── check_data.py
├── check_gpu.py
├── data_processing.py
├── data_visualization.ipynb
├── evaluate_model.py
├── load_model.py
├── train_cpo.py
├── train_lora.py
└── training_visualization.ipynb

```

---

# 🛠 4. Hướng dẫn cài đặt
### 1️⃣ Cài môi trường
```
pip install -r requirements.txt
```

---


# ✨ 5. Chạy mô hình tóm tắt
```
python streamlit run app.py
```

---

# 📊 6. Kết quả mô hình
| Metric | Base | LoRA | LoRA + CPO |
|--------|-------|-------|------|
| ROUGE-1 | 5.54 | 51.95 | 52.97 |
| ROUGE-2 | 1.62 | 25.11 | 25.85 |
| ROUGE-L | 5.54 | 51.92 | 52.94 |
| ROUGE-Lsum | 5.54 | 51.92 | 52.94 |


