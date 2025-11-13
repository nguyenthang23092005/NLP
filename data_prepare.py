import json
import os
import re

input_files = [
    "data_raw/News_Dataset_Vietnamese.json",
    "data_raw/Vietnamese_Online_News_Dataset.json",
    "data_raw/vietnews.json"
]

output_file = "data/processed_dataset.json"
processed_data = []

content_keys = ["Contents", "content", "Article", "article"]
summary_keys = ["Summary", "summary", "Abstract", "abstract"]

def clean_text(text):
    if not text:
        return ""
    text = str(text)
    text = re.sub(r"[\xa0_\-\\/*]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

for input_file in input_files:
    if not os.path.exists(input_file):
        print(f"File {input_file} không tồn tại, bỏ qua.")
        continue

    with open(input_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f) 
        except json.JSONDecodeError:
            print(f"Không đọc được file {input_file}")
            continue

        for item in data:
            content = next((item[k] for k in content_keys if k in item), None)
            summary = next((item[k] for k in summary_keys if k in item), None)

            if content and summary:
                processed_data.append({
                    "content": clean_text(content),
                    "summary": clean_text(summary)
                })

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)

print(f"Đã tạo file {output_file} với {len(processed_data)} mục.")
