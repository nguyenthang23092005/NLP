import json

print("="*60)
print("KIỂM TRA DỮ LIỆU CƠ BẢN")
print("="*60)

# Load data
print("\n1. Loading dataset...")
try:
    with open("data/processed_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"   ✓ Loaded {len(data)} samples")
except Exception as e:
    print(f"   ✗ Error loading data: {e}")
    exit(1)

# Check first 10 samples
print("\n2. Kiểm tra 10 mẫu đầu tiên:")
print("-"*60)

issues = []
for i in range(min(10, len(data))):
    sample = data[i]
    content = sample.get("content", "")
    summary = sample.get("summary", "")
    
    print(f"\nMẫu {i+1}:")
    print(f"  Content: {len(content)} chars, {len(content.split())} words")
    print(f"  Summary: {len(summary)} chars, {len(summary.split())} words")
    
    # Check for issues
    if len(content) == 0:
        issues.append(f"Mẫu {i+1}: Content rỗng!")
        print("  ❌ Content rỗng!")
    if len(summary) == 0:
        issues.append(f"Mẫu {i+1}: Summary rỗng!")
        print("  ❌ Summary rỗng!")
    if len(summary.split()) < 3:
        issues.append(f"Mẫu {i+1}: Summary quá ngắn ({len(summary.split())} words)!")
        print(f"  ⚠ Summary quá ngắn!")
    
    # Print sample
    print(f"  Content (first 100 chars): {content[:100]}...")
    print(f"  Summary (first 100 chars): {summary[:100]}...")

# Statistics on all data
print("\n3. Thống kê toàn bộ dataset:")
print("-"*60)

empty_content = 0
empty_summary = 0
short_summary = 0
very_short_summary = 0

content_lengths = []
summary_lengths = []

for sample in data:
    content = sample.get("content", "")
    summary = sample.get("summary", "")
    
    content_words = len(content.split())
    summary_words = len(summary.split())
    
    content_lengths.append(content_words)
    summary_lengths.append(summary_words)
    
    if len(content) == 0:
        empty_content += 1
    if len(summary) == 0:
        empty_summary += 1
    if summary_words < 5:
        short_summary += 1
    if summary_words < 3:
        very_short_summary += 1

print(f"  Total samples: {len(data)}")
print(f"  Empty content: {empty_content}")
print(f"  Empty summary: {empty_summary}")
print(f"  Short summary (<5 words): {short_summary}")
print(f"  Very short summary (<3 words): {very_short_summary}")

if content_lengths:
    print(f"\n  Content length stats:")
    print(f"    Min: {min(content_lengths)} words")
    print(f"    Max: {max(content_lengths)} words")
    print(f"    Avg: {sum(content_lengths)/len(content_lengths):.1f} words")

if summary_lengths:
    print(f"\n  Summary length stats:")
    print(f"    Min: {min(summary_lengths)} words")
    print(f"    Max: {max(summary_lengths)} words")
    print(f"    Avg: {sum(summary_lengths)/len(summary_lengths):.1f} words")

# Report
print("\n4. Tổng kết:")
print("="*60)

if empty_content > 0 or empty_summary > 0:
    print(f"❌ CRITICAL: Dataset có vấn đề!")
    print(f"   - Content rỗng: {empty_content}")
    print(f"   - Summary rỗng: {empty_summary}")
    print(f"\n   CẦN LỌC LẠI DATASET!")
elif very_short_summary > 100:
    print(f"⚠ WARNING: Có {very_short_summary} summary rất ngắn (<3 words)")
    print(f"   Tỉ lệ: {very_short_summary/len(data)*100:.1f}%")
    print(f"\n   Điều này CÓ THỂ gây ra loss=0.0, grad_norm=nan")
    print(f"   Khuyến nghị: Lọc bỏ các summary quá ngắn")
elif short_summary > len(data) * 0.1:
    print(f"⚠ WARNING: Có {short_summary} summary ngắn (<5 words)")
    print(f"   Tỉ lệ: {short_summary/len(data)*100:.1f}%")
else:
    print(f"✓ Dataset trông OK!")
    print(f"   - Không có content/summary rỗng")
    print(f"   - Summary có độ dài hợp lý")

print("\n" + "="*60)
