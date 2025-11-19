import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import os
from pathlib import Path
import tempfile

# Import libraries for file processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from docx2txt import process as docx2txt_process
    DOC_AVAILABLE = True
except ImportError:
    DOC_AVAILABLE = False


# ============== Configuration ==============
st.set_page_config(
    page_title="Vietnamese Text Summarization",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
    }
    .summary-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    .stats-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


# ============== File Processing Functions ==============
def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    if not PDF_AVAILABLE:
        st.error("PyPDF2 chưa được cài đặt. Chạy: pip install PyPDF2")
        return None
    
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Lỗi khi đọc PDF: {str(e)}")
        return None


def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    if not DOCX_AVAILABLE:
        st.error("python-docx chưa được cài đặt. Chạy: pip install python-docx")
        return None
    
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Lỗi khi đọc DOCX: {str(e)}")
        return None


def extract_text_from_doc(file):
    """Extract text from DOC file"""
    if not DOC_AVAILABLE:
        st.error("docx2txt chưa được cài đặt. Chạy: pip install docx2txt")
        return None
    
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        # Extract text
        text = docx2txt_process(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        return text.strip() if text else None
    except Exception as e:
        st.error(f"Lỗi khi đọc DOC: {str(e)}")
        return None


# ============== Model Loading ==============
@st.cache_resource
def load_model(model_path, base_model="google/mt5-small"):
    """Load model with caching"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        with st.spinner(f"Đang tải model từ {model_path}..."):
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            
            # Check if it's a LoRA model
            if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                # Load LoRA model
                base = AutoModelForSeq2SeqLM.from_pretrained(base_model)
                model = PeftModel.from_pretrained(base, model_path)
                model = model.merge_and_unload()
            else:
                # Load full fine-tuned model
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            
            model.to(device)
            model.eval()
            
            return model, tokenizer, device
    
    except Exception as e:
        st.error(f"Lỗi khi tải model: {str(e)}")
        return None, None, None


# ============== Summarization Function ==============
def generate_summary(
    text,
    model,
    tokenizer,
    device,
    max_source_length=4096,
    max_target_length=512,
    num_beams=5,
    temperature=0.8,
    top_p=0.92,
    do_sample=True,
    repetition_penalty=1.2
):
    """Generate summary for input text"""
    
    if not text or not text.strip():
        return None
    
    # CRITICAL FIX: Dùng prefix đúng cho mT5
    input_text = f"summarize: {text}"
    
    inputs = tokenizer(
        input_text,
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
        padding=False
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_target_length,
            min_length=60,  
            num_beams=num_beams,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            do_sample=do_sample,
            early_stopping=True,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=3,
            length_penalty=1.2,  # Tăng từ 1.0 -> 1.2 để ưu tiên output dài hơn
            # CRITICAL: Force decoder to start properly (not with pad)
            decoder_start_token_id=tokenizer.pad_token_id,  # mT5 uses pad as decoder start
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode và làm sạch output
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # DEBUG: Phân tích chi tiết tokens
    print(f"\n{'='*60}")
    print(f"DEBUG - Token Analysis:")
    print(f"{'='*60}")
    print(f"Total tokens: {len(outputs[0])}")
    print(f"Output tokens: {outputs[0].tolist()}")
    
    # Decode từng token để xem
    print(f"\nToken breakdown:")
    for i, token_id in enumerate(outputs[0].tolist()):
        token_text = tokenizer.decode([token_id])
        token_name = tokenizer.convert_ids_to_tokens([token_id])[0]
        print(f"  [{i}] ID={token_id:6d} | Token='{token_name:15s}' | Text='{token_text}'")
    
    # Decode với skip_special_tokens
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nDecoded (skip_special_tokens=True): '{summary}'")
    
    # Decode không skip để so sánh
    summary_with_special = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"Decoded (skip_special_tokens=False): '{summary_with_special}'")
    print(f"{'='*60}\n")
    
    # Post-processing: Loại bỏ các token lạ còn sót lại
    summary = summary.replace("<extra_id_0>", "").replace("<extra_id_1>", "")
    summary = summary.replace("<extra_id_2>", "").replace("<extra_id_3>", "")
    summary = summary.replace("<pad>", "").replace("</s>", "").replace("<s>", "")
    summary = summary.strip()
    
    # Kiểm tra nếu summary không hợp lệ
    if not summary or len(summary) < 15:
        print(f"⚠️ WARNING: Summary too short or invalid: '{summary}'")
        return "⚠️ Model chưa được train đúng. Vui lòng train lại model hoặc chọn model khác."
    
    # Kiểm tra nếu summary chỉ là copy từ input (dấu hiệu model chưa học)
    if summary in text:
        print(f"⚠️ WARNING: Summary is just a substring from input")
        return f"⚠️ Model đang copy văn bản gốc thay vì tóm tắt. Kết quả: '{summary}'"
    
    return summary


# ============== Main App ==============
def main():
    # Header
    st.markdown('<div class="main-header">📝 Tóm Tắt Văn Bản Tiếng Việt</div>', unsafe_allow_html=True)
    
    # Sidebar - Model Selection
    st.sidebar.header("⚙️ Cấu hình")
    
    # Model selection
    model_options = {
        "LoRA + CPO": "./models/mt5-cpo-full/checkpoint-1500",
        "LoRA v1": "./models/mt5-lora-full/checkpoint-7728",
        "LoRA v2": "./models/mt5-lora-v2/checkpoint-5000",
        "Base mT5": "google/mt5-small"
    }
    
    # Find available models
    available_models = {}
    for name, path in model_options.items():
        if name == "Base mT5" or os.path.exists(path):
            available_models[name] = path
    
    if not available_models:
        st.error("Không tìm thấy model nào! Vui lòng train model trước.")
        return
    
    # Default to LoRA SFT model if available
    default_index = 0
    if "LoRA SFT (Khuyên dùng)" in available_models:
        default_index = list(available_models.keys()).index("LoRA SFT (Khuyên dùng)")
    
    selected_model_name = st.sidebar.selectbox(
        "Chọn model:",
        options=list(available_models.keys()),
        index=default_index,
        help="Chọn model để sử dụng cho tóm tắt"
    )
    
    model_path = available_models[selected_model_name]
    
    # Fixed parameters (không hiển thị trên UI) - Tăng độ dài cho tóm tắt phong phú hơn
    max_source_length = 2048  # Tăng từ 512 -> 1024 để nhận văn bản dài hơn
    max_target_length = 512   # Tăng từ 128 -> 256 để tóm tắt chi tiết hơn
    num_beams = 5             # Tăng từ 4 -> 5 để tìm kiếm tốt hơn
    repetition_penalty = 1.3  # Tăng từ 1.2 -> 1.3 để giảm lặp từ
    do_sample = True          # Bật sampling để đa dạng hơn
    temperature = 0.8         # Nhiệt độ 0.8 cân bằng giữa đa dạng và chất lượng
    top_p = 0.92              # Top-p cao hơn cho nhiều lựa chọn từ
    
    # Device info
    device_type = "🎮 GPU" if torch.cuda.is_available() else "💻 CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.sidebar.info(f"{device_type}: {gpu_name}")
    else:
        st.sidebar.warning(f"{device_type} (chậm hơn)")
    
    # Load model
    model, tokenizer, device = load_model(model_path)
    
    if model is None:
        st.error("Không thể tải model!")
        return
    
    st.sidebar.success(f"✅ Model đã sẵn sàng: {selected_model_name}")
    
    # Main content
    st.markdown("---")
    
    # Input method selection
    input_method = st.radio(
        "Chọn phương thức nhập:",
        options=["📝 Nhập văn bản trực tiếp", "📄 Upload file (PDF, DOC, DOCX)"],
        horizontal=True
    )
    
    input_text = None
    
    if input_method == "📝 Nhập văn bản trực tiếp":
        # Text input
        input_text = st.text_area(
            "Nhập văn bản cần tóm tắt:",
            height=250,
            placeholder="Nhập hoặc dán văn bản tiếng Việt vào đây...",
            help="Nhập văn bản bạn muốn tóm tắt",
            key="input_text_area"
        )
        
        # Kiểm tra nếu input thay đổi thì xóa kết quả cũ
        if 'last_input' not in st.session_state:
            st.session_state.last_input = ""
        
        if input_text != st.session_state.last_input:
            st.session_state.last_input = input_text
            if 'summary_result' in st.session_state:
                st.session_state.summary_result = None
        
        # Output area - hiển thị kết quả tóm tắt
        if 'summary_result' in st.session_state and st.session_state.summary_result:
            st.markdown("#### 📝 Kết quả tóm tắt:")
            st.text_area(
                "Tóm tắt:",
                value=st.session_state.summary_result,
                height=150,
                disabled=False,
                key="output_text_area",
                help="Bạn có thể chỉnh sửa kết quả tóm tắt tại đây"
            )
    
    else:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload file:",
            type=['pdf', 'doc', 'docx'],
            help="Hỗ trợ các định dạng: PDF, DOC, DOCX"
        )
        
        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            with st.spinner(f"Đang đọc file {uploaded_file.name}..."):
                if file_type == 'pdf':
                    input_text = extract_text_from_pdf(uploaded_file)
                elif file_type == 'docx':
                    input_text = extract_text_from_docx(uploaded_file)
                elif file_type == 'doc':
                    input_text = extract_text_from_doc(uploaded_file)
                else:
                    st.error(f"Định dạng file không được hỗ trợ: {file_type}")
            
            if input_text:
                st.success(f"✅ Đã đọc thành công file: {uploaded_file.name}")
                
                # Show extracted text in expander
                with st.expander("📄 Xem nội dung đã trích xuất"):
                    st.text_area("Văn bản từ file:", input_text, height=200, disabled=True)
        
        # Output area - hiển thị kết quả tóm tắt cho file upload
        if 'summary_result' in st.session_state and st.session_state.summary_result:
            st.markdown("#### 📝 Kết quả tóm tắt:")
            st.text_area(
                "Tóm tắt:",
                value=st.session_state.summary_result,
                height=150,
                disabled=False,
                key="output_text_area_file",
                help="Bạn có thể chỉnh sửa kết quả tóm tắt tại đây"
            )
    
    # Summarize button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        summarize_button = st.button(
            "🚀 TÓM TẮT NGAY",
            type="primary",
            use_container_width=True
        )
    
    # Generate summary
    if summarize_button:
        if not input_text or not input_text.strip():
            st.warning("⚠️ Vui lòng nhập văn bản hoặc upload file!")
        else:
            # Xóa kết quả cũ trước khi tóm tắt mới
            if 'summary_result' in st.session_state:
                st.session_state.summary_result = None
            
            # Show input statistics
            input_words = len(input_text.split())
            input_chars = len(input_text)
            
            st.markdown("### 📊 Thông tin văn bản đầu vào")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Số từ", f"{input_words:,}")
            with col2:
                st.metric("Số ký tự", f"{input_chars:,}")
            with col3:
                estimated_time = max(1, input_words // 100)
                st.metric("Thời gian ước tính", f"~{estimated_time}s")
            
            # Generate summary
            with st.spinner("⏳ Đang tóm tắt văn bản..."):
                summary = generate_summary(
                    text=input_text,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty
                )
                
                # Lưu kết quả vào session state để hiển thị ở text area
                if summary and len(summary.strip()) > 0:
                    st.session_state.summary_result = summary
                    # Cập nhật last_input để tránh xóa kết quả mới
                    st.session_state.last_input = input_text
            
            # Rerun để cập nhật UI với kết quả trong text area
            if summary and len(summary.strip()) > 0:
                st.rerun()
            
            # Show statistics and download options below the text areas
            if 'summary_result' in st.session_state and st.session_state.summary_result:
                summary = st.session_state.summary_result
                
                # Summary statistics
                summary_words = len(summary.split())
                summary_chars = len(summary)
                compression_ratio = (1 - summary_words / input_words) * 100 if input_words > 0 else 0
                
                st.markdown("### 📈 Thống kê tóm tắt")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Số từ", f"{summary_words}")
                with col2:
                    st.metric("Số ký tự", f"{summary_chars}")
                with col3:
                    st.metric("Tỷ lệ nén", f"{compression_ratio:.1f}%")
                with col4:
                    st.metric("Model", selected_model_name.split('(')[0].strip())
                
                # Download options
                st.markdown("---")
                st.markdown("### 💾 Tải xuống kết quả")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download as text
                    result_text = f"VĂN BẢN GỐC:\n{input_text}\n\n{'='*50}\n\nTÓM TẮT:\n{summary}\n\n{'='*50}\n\nTHỐNG KÊ:\n- Văn bản gốc: {input_words} từ, {input_chars} ký tự\n- Tóm tắt: {summary_words} từ, {summary_chars} ký tự\n- Tỷ lệ nén: {compression_ratio:.1f}%\n- Model: {selected_model_name}"
                    
                    st.download_button(
                        label="📄 Tải xuống (.txt)",
                        data=result_text,
                        file_name="tom_tat.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    # Copy to clipboard button (visual only)
                    st.button("📋 Sao chép tóm tắt", help="Sao chép tóm tắt vào clipboard")
            else:
                st.error("❌ Không thể tạo tóm tắt. Vui lòng thử lại!")


if __name__ == "__main__":
    main()
