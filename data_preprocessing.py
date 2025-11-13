import os
import json

input_file = "data/processed_dataset.json"
output_file = "data/processed_dataset_cleaned.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)
    
print(data[52])
# for sample in data[:5]:  # chỉ lấy 5 mẫu đầu
#     print(sample)
