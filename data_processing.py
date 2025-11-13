import re
import vietokenizer

# Danh sách stopwords tiếng Việt (có thể mở rộng)
stopwords = set([
    "và", "của", "là", "có", "cho", "được", "trong", "một", "những", "này", 
    "mình", "bạn", "các", "như", "khi", "với", "theo"
])

def clean_tokenize_vietnamese(text, remove_stopwords=True):
    if not text:
        return []
    
    # 1. Chuyển về chữ thường
    text = text.lower()
    
    # 2. Xóa tất cả ký tự đặc biệt (chỉ giữ chữ và số, khoảng trắng)
    text = re.sub(r"[^a-z0-9\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", " ", text)
    
    # 3. Xóa khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()
    
    # 4. Tokenization bằng VietTokenizer
    tokens = vietokenizer.VietTokenizer(text).split()
    
    # 5. Loại bỏ stopwords nếu cần
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    
    return tokens

# Ví dụ
text = "Hôm nay, trời đẹp quá! Mình muốn đi dạo, nhưng không có ai đi cùng."
print(clean_tokenize_vietnamese(text))
# Output: ['hôm_nay', 'trời', 'đẹp', 'quá', 'muốn', 'đi', 'dạo', 'nhưng', 'không', 'ai', 'đi', 'cùng']
