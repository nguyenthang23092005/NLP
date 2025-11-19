import os
import re
import json
from multiprocessing import Pool, cpu_count

# ====== CONFIG ======
input_files = [
    "data_raw/News_Dataset_Vietnamese.json",
    "data_raw/Vietnamese_Online_News_Dataset.json",
    "data_raw/vietnews.json"
]

output_file = "data/processed_dataset.json"

content_keys = ["Contents", "content", "Article", "article"]
summary_keys = ["Summary", "summary", "Abstract", "abstract"]

# ====== REGEX ======
RE_STRIP = re.compile(r"[^\sa-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ.,!?-]")
RE_MULTI_SPACE = re.compile(r"\s+")


# ====== CLEAN RAW TEXT ======
def clean_text(text):
    if not text:
        return ""

    text = str(text).lower()
    text = RE_STRIP.sub(" ", text)
    text = RE_MULTI_SPACE.sub(" ", text).strip()
    return text


# ====== PROCESS SAMPLE ======
def process_sample(item):
    # Lấy field
    raw_content = next((item[k] for k in content_keys if k in item), "")
    raw_summary = next((item[k] for k in summary_keys if k in item), "")

    content = clean_text(raw_content)
    summary = clean_text(raw_summary)

    if not content or not summary:
        return None

    return {
        "content": content,
        "summary": summary
    }


# ====== PROCESS CHUNK ======
def process_chunk(chunk):
    return [res for res in map(process_sample, chunk) if res]


# ====== CHUNKIFY ======
def chunkify(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i + 1)*k + min(i + 1, m)] for i in range(n)]


def main():
    merged_raw = []

    # ===== LOAD ALL RAW FILES =====
    for fname in input_files:
        if not os.path.exists(fname):
            print(f"File {fname} không tồn tại, bỏ qua.")
            continue

        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Đã load {fname}: {len(data)} mẫu")
            merged_raw.extend(data)
        except:
            print(f"Lỗi đọc {fname}")

    print(f"Tổng raw sample: {len(merged_raw)}")

    # ===== MULTIPROCESSING =====
    n_cpu = cpu_count()
    chunks = chunkify(merged_raw, n_cpu)

    cleaned = []
    with Pool(n_cpu) as pool:
        for result in pool.map(process_chunk, chunks):
            cleaned.extend(result)

    print(f"Số sample hợp lệ: {len(cleaned)}")

    # ===== SAVE FINAL JSON =====
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, sample in enumerate(cleaned):
            f.write("  {\n")
            f.write(f"    \"content\": {json.dumps(sample['content'], ensure_ascii=False)},\n")
            f.write(f"    \"summary\": {json.dumps(sample['summary'], ensure_ascii=False)}\n")
            f.write("  }")
            if i < len(cleaned) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("]\n")

    print("File lưu tại:", output_file)


if __name__ == "__main__":
    main()
