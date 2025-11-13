import os
import json

input_file = "data/processed_dataset.json"
output_file = "data/processed_dataset_cleaned.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)
    
print(len(data))

cleaned_data = []
empty_indices = []
for i, sample in enumerate(data):
    content = sample.get("content", "").strip()
    summary = sample.get("summary", "").strip()
    if content and summary:
        cleaned_data.append(sample)
    else:
        empty_indices.append(i)
        print(data[i])

print(f"Tổng số sample ban đầu: {len(data)}")
print(f"Số sample trống bị loại bỏ: {len(empty_indices)}")
print(f"Số sample còn lại: {len(cleaned_data)}")

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
