import streamlit as st
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import numpy as np
import io
import os
import pandas as pd
from pathlib import Path
import time
import zipfile
import tempfile
import shutil
import base64
import hashlib
import datetime
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

def pil_image_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    base64_str = base64.b64encode(byte_im).decode()
    return base64_str

# Các biến cấu hình
top_k_results = 10  # Số lượng kết quả top dự đoán muốn hiển thị

# Helper function to load and encode the image
def get_image_as_base64(image_path):
    if not os.path.exists(image_path):
        st.error(f"Logo image not found at: {image_path}")
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Phân tích Mã độc",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS tùy chỉnh ---
st.markdown("""
<style>
    /* Background color for entire app */
    .stApp {
        background-color: #F6FBFF;
        margin-top: 2rem;
    }
    
    /* Main Styles */
    .main-header {
        font-size: 2rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
        padding: 0.5rem 0;
        background: radial-gradient(circle, #BBDEFB 0%, #E3F2FD 70%, #F5FBFF 100%);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Model upload styling */
    .model-upload-section {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border: 2px dashed #dee2e6;
    }
    
    .model-info-card {
        background-color: #e8f5e8;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #4caf50;
    }
    
    /* Add equal height columns */
    .equal-height-columns .stColumn {
        display: flex;
        flex-direction: column;
    }
    
    .equal-height-columns .stColumn > div {
        flex-grow: 1;
        display: flex;
        flex-direction: column;
    }
    
    .equal-height-columns .stColumn > div > div {
        flex-grow: 1;
        display: flex;
        flex-direction: column;
    }
    
    /* File info card styling */
    .file-info {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
     /* Result cards styling */
    .malware-card, .benign-card, .uncertain-card {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        height: 100%;
    }
    
    .malware-card {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    
    .benign-card {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    
    .uncertain-card {
        background-color: #fff8e1;
        border-left: 5px solid #ffc107;
    }
    
    /* Logo styling */
    .logo-container {
        margin-right: 5px;
        margin-left: -100px;
    }
    
    .logo-container img {
        height: 75px;
        width: auto;
        transition: transform 0.15s ease;
    }
    
    /* Title styling */
    .title-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    .title-line {
        display: block;
        text-align: center;
        line-height: 1.2;
    }
    
    .logo-container img:hover {
        transform: scale(1.05);
    }
    
    /* Add this to reduce the top margin of the main container */
    .block-container {
        padding-top: 1rem !important;
        margin-top: 0 !important;
    }
    
    /* Reduce spacing at the top of the app */
    .stApp {
        margin-top: 2rem;
    }
    
    /* Set sidebar width to exactly 324px when expanded */
    [data-testid="stSidebar"][aria-expanded="true"] {
        width: 324px !important;
        min-width: 324px !important;
        max-width: 324px !important;
    }
    
    /* Make sure sidebar can be collapsed */
    [data-testid="stSidebar"][aria-expanded="false"] {
        width: auto !important;
        margin-left: -20px;
    }
    
    /* Fix for sidebar collapse button */
    button[kind="headerNoPadding"] {
        display: block !important;
    }
    
    /* Sidebar styling based on the image */
    .st-emotion-cache-vmpjyt.e1dbuyne0 {
        position: relative;
        top: 1.8rem;
        background-color: rgb(240, 242, 246);
        z-index: 99999;
        min-width: 244px;
        max-width: 550px;
        transform: none;
        transition: transform 300ms, min-width 300ms, max-width 300ms;
    }
    
    /* Hide sidebar header with collapse button */
    [data-testid="stSidebarHeader"] {
        display: none !important;
    }
    
    /* Hide the arrow button at the bottom */
    button[kind="secondary"] {
        display: none !important;
    }
    
    /* Make info box text bold and centered */
    .stAlert {
        text-align: center !important;
        font-weight: bold !important;
    }
    
    /* Improve tab styling with more padding */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 10px 0px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #E3F2FD;
        border-radius: 4px 4px 0 0;
        padding: 12px 20px;
        border: 1px solid #BBDEFB;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1976D2 !important;
        color: white !important;
    }
    
    /* Giảm khoảng cách giữa các label và input */
    .css-ocqkz7, .css-1qrvfrg {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Redesigned sub-header for better appearance */
    .sub-header {
        font-size: 1.4rem;
        color: #0D47A1;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 1rem;
        padding: 0.4rem 4rem;
        background: linear-gradient(90deg, #E3F2FD, #BBDEFB);
        border-radius: 10000px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Center tabs */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }
    
    /* Info text styling */
    .info-text {
        text-align: center;
        padding: 0px;
        margin: 10px 0;
        border-radius: 8px;
        background-color: #f8f9fa;
    }
    
    /* Map header styling */
    .map-header {
        font-size: 1.5rem;
        color: #1E88E5;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        text-align: center;
    }
    
    /* Map container styling */
    iframe {
        width: 100%;
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Tiêu đề ---
st.markdown('''
<div class="main-header">
    <a href="https://dhkthc.bocongan.gov.vn/TrangChu/" target="_blank" class="logo-container">
        <img src="https://raw.githubusercontent.com/HICKER-WH/PhatHienPhanMemDocHai/main/Logot07.png" alt="Logo" style="height: 60px;">
    </a>
    <div class="title-container">
        <span class="title-line">🔍 Phát Hiện Phần Mềm Độc Hại</span>
        <span class="title-line">Trên Nền Tảng Windows</span>
    </div>
</div>
''', unsafe_allow_html=True)
st.markdown('<div class="info-text">💡 Sử dụng mô hình Inception v3 để phân tích và phát hiện mã độc trong các file thực thi .exe hoặc .dll</div>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="sub-header">⚙️ <span>Cấu hình</span></div>', unsafe_allow_html=True)
    
    # # THÊM PHẦN UPLOAD MODEL
    # st.markdown("### 🤖 Tải mô hình")
    
    # Chọn cách tải model
    model_source = st.radio(
        "🤖 Chọn nguồn mô hình:",
        ["Đường dẫn cục bộ", "Upload từ máy tính"],
        help="Chọn cách để tải mô hình AI"
    )
    
    MODEL_PATH = None
    uploaded_model = None
    
    if model_source == "Đường dẫn cục bộ":
        # Đường dẫn đến model (cũ)
        MODEL_PATH = st.text_input("Đường dẫn mô hình", value='Best_Inception_Version3.pth')
        
    elif model_source == "Upload từ máy tính":
        # Upload model từ máy tính (mới)
        # st.markdown('<div class="model-upload-section">', unsafe_allow_html=True)
        uploaded_model = st.file_uploader(
            "Chọn file mô hình (.pth, .pt)",
            type=['pth', 'pt'],
            help="Upload file mô hình PyTorch đã được huấn luyện"
        )
        if uploaded_model is not None:
            # Hiển thị thông tin file model
            model_size_mb = uploaded_model.size / (1024 * 1024)
            st.markdown(f"""
            <div class="model-info-card">
                <strong>📄 Tên file:</strong> {uploaded_model.name}<br>
                <strong>📊 Kích thước:</strong> {model_size_mb:.2f} MB<br>
                <strong>🔧 Loại:</strong> {uploaded_model.type}<br>
                <strong>✅ Trạng thái:</strong> Đã tải lên thành công
            </div>
            """, unsafe_allow_html=True)
    
    # Kích thước ảnh
    col1, col2 = st.columns(2)
    with col1:
        img_size_1 = st.number_input("Size ảnh model", value=299, min_value=64, max_value=512)
    with col2:
        img_size_2 = st.number_input("Size ảnh hiển thị", value=224, min_value=64, max_value=512)    
    IMAGE_SIZE = (int(img_size_1), int(img_size_1))
    IMAGE_SIZE2 = (int(img_size_2), int(img_size_2))
    
    # Ngưỡng xác suất
    threshold = st.slider("Ngưỡng xác suất phát hiện mã độc", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    
    # Thêm tham số mới
    max_file_size = st.number_input("Kích thước file tối đa (MB)", value=200, max_value=500) 
    
    # Danh sách lớp lành tính
    benign_input = st.text_area("Nhập tên các lớp lành tính (mỗi tên một dòng)", "benign\nclean\nnormal\nlegitimate")
    benign_classes = [cls.strip() for cls in benign_input.split('\n') if cls.strip()]
    
    # Kiểm tra thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"Đang sử dụng: {device}")

# --- Hàm trợ giúp ---
def binary_to_image(file_bytes, size=(224, 224)):
    """Chuyển đổi bytes của file thành ảnh PIL Grayscale, sau đó sang RGB."""
    try:
        byte_array = np.frombuffer(file_bytes, dtype=np.uint8)
        required_pixels = size[0] * size[1]

        # Padding hoặc Truncating để có đúng số lượng pixel
        if byte_array.size < required_pixels:
            # Pad bằng 0 nếu file nhỏ hơn kích thước ảnh
            padded_array = np.pad(byte_array, (0, required_pixels - byte_array.size), 'constant')
        else:
            # Truncate nếu file lớn hơn kích thước ảnh
            padded_array = byte_array[:required_pixels]

        # Reshape thành ảnh grayscale
        image_array = padded_array.reshape(size)

        # Chuyển đổi sang ảnh PIL Grayscale
        img = Image.fromarray(image_array, 'L')

        # Chuyển đổi Grayscale sang RGB (vì InceptionV3 cần 3 kênh)
        img_rgb = img.convert('RGB')
        return img_rgb
    except Exception as e:
        st.error(f"Lỗi khi chuyển đổi file sang ảnh: {e}")
        return None

# THÊM HÀM LOAD MODEL TỪ UPLOADED FILE
@st.cache_resource
def load_pytorch_model_from_upload(uploaded_file, device):
    """Tải mô hình Inception v3 từ uploaded file."""
    try:
        # Đọc file từ uploaded_file
        file_bytes = uploaded_file.getvalue()
        
        # Tải checkpoint từ bytes
        checkpoint = torch.load(io.BytesIO(file_bytes), map_location=device)
        
        # Lấy thông tin về lớp từ checkpoint
        if 'classes' in checkpoint:
            class_names = checkpoint['classes']
        else:
            st.warning("Không tìm thấy thông tin lớp trong file model, sử dụng tên lớp mặc định")
            class_names = [f'Loại_{i+1}' for i in range(60)]  # Mặc định 60 lớp
            
        num_classes = len(class_names)
        
        # Khởi tạo kiến trúc mô hình
        model = torchvision.models.inception_v3(weights=None, aux_logits=True)

        # Điều chỉnh lớp cuối cho phù hợp với số lớp
        num_ftrs_fc = model.fc.in_features
        model.fc = nn.Linear(num_ftrs_fc, num_classes)
        if model.aux_logits:
             num_ftrs_aux = model.AuxLogits.fc.in_features
             model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)

        # Tải trọng số đã huấn luyện
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint # Giả sử file chỉ chứa state_dict

        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval() # Chuyển sang chế độ đánh giá
        
        st.success(f"✅ Đã tải thành công mô hình từ file: {uploaded_file.name}")
        return model, class_names
        
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình từ file upload: {e}")
        return None, None

# Cache việc tải model để tăng tốc độ (cũ - từ đường dẫn)
@st.cache_resource
def load_pytorch_model(model_path, device):
    """Tải mô hình Inception v3 đã huấn luyện từ đường dẫn."""
    try:
        if not os.path.exists(model_path):
            st.error(f"Lỗi: Không tìm thấy file mô hình tại '{model_path}'")
            return None, None

        # Tải checkpoint để lấy thông tin về lớp
        checkpoint = torch.load(model_path, map_location=device)
        
        # Lấy thông tin về lớp từ checkpoint
        if 'classes' in checkpoint:
            class_names = checkpoint['classes']
        else:
            st.warning("Không tìm thấy thông tin lớp trong file model, sử dụng tên lớp mặc định")
            class_names = [f'Loại_{i+1}' for i in range(60)]  # Mặc định 60 lớp
            
        num_classes = len(class_names)
        
        # Khởi tạo kiến trúc mô hình
        model = torchvision.models.inception_v3(weights=None, aux_logits=True)

        # Điều chỉnh lớp cuối cho phù hợp với số lớp
        num_ftrs_fc = model.fc.in_features
        model.fc = nn.Linear(num_ftrs_fc, num_classes)
        if model.aux_logits:
             num_ftrs_aux = model.AuxLogits.fc.in_features
             model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)

        # Tải trọng số đã huấn luyện
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint # Giả sử file chỉ chứa state_dict

        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval() # Chuyển sang chế độ đánh giá
        return model, class_names
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        return None, None

def predict(model, image, device, class_names, top_k=10):
    """Thực hiện dự đoán trên ảnh đầu vào."""
    if model is None or image is None:
        return None

    try:
        # Định nghĩa các phép biến đổi ảnh
        preprocess = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0) # Tạo batch dimension
        input_batch = input_batch.to(device)

        with torch.no_grad():
            output = model(input_batch)
            # Xử lý output từ Inception v3 (có thể là tuple nếu aux_logits=True trong eval)
            if isinstance(output, tuple):
                output = output[0] # Chỉ lấy output chính

            probabilities = torch.softmax(output, dim=1)

        # Lấy top K dự đoán
        top_p, top_class_indices = torch.topk(probabilities, top_k, dim=1)

        # Chuyển kết quả sang CPU và numpy
        top_p = top_p.squeeze().cpu().numpy()
        top_class_indices = top_class_indices.squeeze().cpu().numpy()

        # Tạo danh sách kết quả (class_name, probability)
        results = []
        for i in range(top_k):
            class_idx = top_class_indices[i]
            if class_idx < len(class_names):
                class_name = class_names[class_idx]
                probability = float(top_p[i])
                results.append({"Lớp": class_name, "Xác suất": probability})
            else:
                 results.append({"Lớp": f"Index_{class_idx}_Ngoài_Phạm_Vi", "Xác suất": float(top_p[i])})

        return results

    except Exception as e:
        st.error(f"Lỗi trong quá trình dự đoán: {e}")
        return None

def is_malware(predictions, benign_classes=None, threshold=0.7):
    """
    Xác định xem file có phải là mã độc hay không dựa trên dự đoán.
    
    Args:
        predictions: Danh sách các dự đoán
        benign_classes: Danh sách các lớp được coi là lành tính (không phải mã độc)
        threshold: Ngưỡng xác suất để xác định kết quả
    
    Returns:
        (bool, str, float, int): (Có phải mã độc không, Lý do, Xác suất, Loại kết quả)
        Loại kết quả: 0 = lành tính, 1 = mã độc, 2 = không chắc chắn
    """
    if not predictions or len(predictions) == 0:
        return False, "Không có kết quả dự đoán", 0.0, 2
    
    # Nếu không có danh sách lớp lành tính, chỉ coi "benign" là lành tính
    if benign_classes is None:
        benign_classes = ["benign"]
    
    # Lấy lớp có xác suất cao nhất
    top_prediction = predictions[0]
    top_class = top_prediction["Lớp"]
    top_prob = top_prediction["Xác suất"]
    
    # Kiểm tra xem lớp có xác suất cao nhất có phải là lành tính không
    is_benign_class = any(benign_name.lower() in top_class.lower() for benign_name in benign_classes)
    
    # Phân loại dựa trên lớp và ngưỡng xác suất
    if is_benign_class:
        if top_prob >= threshold:
            return False, f"Lành tính ({top_class})", top_prob, 0
        else:
            return False, f"Có thể lành tính ({top_class}), nhưng xác suất thấp", top_prob, 2
    else:
        # Nếu không phải lớp lành tính, thì là mã độc
        if top_prob >= threshold:
            return True, f"Mã độc ({top_class})", top_prob, 1
        else:
            # Khi xác suất thấp, vẫn coi là mã độc nhưng với mức độ tin cậy thấp
            return True, f"Có thể là mã độc ({top_class}), xác suất thấp", top_prob, 2

def extract_zip_to_temp(zip_file):
    """Giải nén file zip vào thư mục tạm và trả về đường dẫn"""
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir

def scan_directory(directory_path, model, class_names, device, benign_classes, threshold, min_size_kb=0, max_size_mb=100, analysis_depth="Cân bằng"):
    """Quét thư mục để tìm và phân tích các file .exe và .dll"""
    results = []
    file_paths = []
    
    # Chuyển đổi kích thước thành bytes
    min_size_bytes = min_size_kb * 1024
    max_size_bytes = max_size_mb * 1024 * 1024
    
    # Tìm tất cả các file .exe và .dll trong thư mục
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.exe', '.dll')):
                file_path = os.path.join(root, file)
                # Kiểm tra kích thước file
                file_size = os.path.getsize(file_path)
                if min_size_bytes <= file_size <= max_size_bytes:
                    file_paths.append(file_path)
    
    if not file_paths:
        return [], 0, 0, 0
    
    # Hiển thị thanh tiến trình
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Thêm container cho thông tin thời gian
    time_info = st.empty()

    # Điều chỉnh độ sâu phân tích
    if analysis_depth == "Nhanh":
        sample_rate = 0.5  # Chỉ phân tích 50% file nếu có quá nhiều
        if len(file_paths) > 100:
            file_paths = file_paths[:int(len(file_paths) * sample_rate)]
    elif analysis_depth == "Sâu":
        # Phân tích tất cả file với cài đặt chi tiết hơn
        pass
    
    # Phân tích từng file
    malware_count = 0
    uncertain_count = 0
    total_files = len(file_paths)
    
    # Thêm biến theo dõi thời gian
    start_time = time.time()
    file_times = []  # Lưu thời gian xử lý mỗi file

    for i, file_path in enumerate(file_paths):
        status_text.text(f"Đang phân tích: {os.path.basename(file_path)} ({i+1}/{total_files})")
        file_start_time = time.time()
        status_text.text(f"Đang phân tích: {os.path.basename(file_path)} ({i+1}/{total_files})")
        try:
            # Đọc file
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            
            # Chuyển đổi thành ảnh
            image = binary_to_image(file_bytes, size=IMAGE_SIZE2)
            
            if image:
                # Dự đoán
                predictions = predict(model, image, device, class_names, top_k=top_k_results)
                
                if predictions:
                    # Kiểm tra có phải mã độc không
                    is_mal, reason, prob, result_type = is_malware(predictions, benign_classes, threshold)
                    
                    # Thêm thông tin
                    # Thêm thông tin chi tiết hơn về file
                    file_info = {
                        "Tên file": os.path.basename(file_path),
                        "Đường dẫn": file_path,
                        "Kích thước": len(file_bytes),
                        "Kích thước (KB)": round(len(file_bytes) / 1024, 2),
                        "Loại": reason,
                        "Xác suất": prob,
                        "Là mã độc": is_mal,
                        "Kết quả": result_type,
                        "Top dự đoán": predictions[:top_k_results],
                        "Thời gian phân tích": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Entropy": calculate_entropy(file_bytes) if analysis_depth == "Sâu" else None
                    }
                    
                    # Lưu kết quả
                    results.append(file_info)
                    
                    if is_mal:
                        malware_count += 1
                    if result_type == 2:  # Không chắc chắn
                        uncertain_count += 1
        
        except Exception as e:
            st.error(f"Lỗi khi phân tích file {file_path}: {e}")
        
         # Lưu thời gian xử lý file
        file_times.append(time.time() - file_start_time)

        # Cập nhật thanh tiến trình
        progress_percent = (i + 1) / total_files
        progress_bar.progress(progress_percent)

        # Tính toán và hiển thị thông tin thời gian
        if i > 0:  # Cần ít nhất 1 file để ước tính
            elapsed_time = time.time() - start_time
            avg_time_per_file = sum(file_times) / len(file_times)
            remaining_files = total_files - (i + 1)
            estimated_remaining_time = avg_time_per_file * remaining_files
            
            # Chuyển đổi thời gian còn lại thành ngày/giờ/phút
            days, remainder = divmod(estimated_remaining_time, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # Chuyển đổi thời gian đã trôi qua
            elapsed_days, remainder = divmod(elapsed_time, 86400)
            elapsed_hours, remainder = divmod(remainder, 3600)
            elapsed_minutes, remainder = divmod(remainder, 60)
            elapsed_seconds = int(remainder)
            
            # Hiển thị thông tin thời gian
            time_info.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <b>Tiến độ quét:</b> {progress_percent:.1%} ({i+1}/{total_files} files) <br>
                <b>Thời gian đã quét:</b> {int(elapsed_days)} ngày {int(elapsed_hours)} giờ {int(elapsed_minutes)} phút {elapsed_seconds} giây <br>
                <b>Thời gian còn lại (ước tính):</b> {int(days)} ngày {int(hours)} giờ {int(minutes)} phút {int(seconds)} giây <br>
                <b>Tốc độ trung bình:</b> {1/avg_time_per_file:.2f} files/giây
            </div>
            """, unsafe_allow_html=True)
        
    # Hiển thị thông tin tổng thời gian quét
    total_time = time.time() - start_time
    days, remainder = divmod(total_time, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    status_text.markdown(f"""
    <div style="background-color: #e6f3ff; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <b>✅ Quét hoàn tất!</b><br>
        <b>Tổng thời gian quét:</b> {int(days)} ngày {int(hours)} giờ {int(minutes)} phút {int(seconds)} giây<br>
        <b>Số file đã quét:</b> {len(results)}/{total_files}<br>
        <b>Số file mã độc phát hiện được:</b> {malware_count}
    </div>
    """, unsafe_allow_html=True)
    
    status_text.empty()
    return results, malware_count, uncertain_count, total_files

def calculate_entropy(data):
    """Tính entropy của dữ liệu binary"""
    if not data:
        return 0
    
    entropy = 0
    for x in range(256):
        p_x = data.count(x) / len(data)
        if p_x > 0:
            entropy += -p_x * np.log2(p_x)
    return entropy

# --- PHẦN TẢI MÔ HÌNH (CẬP NHẬT) ---
model = None
class_names = None

# Tải mô hình dựa trên lựa chọn của người dùng
if model_source == "Đường dẫn cục bộ" and MODEL_PATH:
    model, class_names = load_pytorch_model(MODEL_PATH, device)
elif model_source == "Upload từ máy tính" and uploaded_model is not None:
    model, class_names = load_pytorch_model_from_upload(uploaded_model, device)

# --- Tab chính ---
<<<<<<< HEAD
tab1, tab2, tab3, tab4 = st.tabs(["📄 Quét chương trình đơn lẻ", "📁 Quét nhanh chương trình trong thư mục", "💻 SOC VIỆT NAM", "ℹ️ Thông tin chung"])
=======
tab1, tab2, tab3, tab4 = st.tabs(["📄 Quét chương trình đơn lẻ", "📁 Quét nhanh chương trình trong thư mục", "💻 SOC Việt Nam", "ℹ️ Thông tin chung"])

# --- Tab phân tích file đơn lẻ ---
with tab1:
    st.markdown('<div class="sub-header">📄 Tải lên file để phân tích</div>', unsafe_allow_html=True)
    
    # Kiểm tra xem mô hình đã được tải chưa
    if model is None:
        st.warning("⚠️ Vui lòng tải mô hình trước khi phân tích file!")
        st.info("💡 Hãy chọn nguồn mô hình trong sidebar và tải mô hình lên.")
    else:
        st.markdown("""
        <style>
        /* Hiệu ứng icon khi hover */
        div[data-testid="stFileUploader"] svg {
            transition: transform 0.3s cubic-bezier(.22,1,.36,1.01);
        }
        div[data-testid="stFileUploader"]:hover svg {
            transform: scale(1) rotate(-720deg);
            filter: drop-shadow(0 2px 10px #1976d233);
            color: #ec0567  ;
        }
        
        </style>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Chọn file (.exe hoặc .dll)", type=['exe', 'dll'])

        if uploaded_file is not None:
            st.markdown('<div class="sub-header">📊 Kết quả phân tích</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Tên file:** `{uploaded_file.name}`")
                st.markdown(f"**Loại file:** `{uploaded_file.type}`")
                st.markdown(f"**Kích thước:** `{uploaded_file.size:,}` bytes (`{uploaded_file.size/1024:.2f}` KB)")
                st.markdown(f"**Thời gian phân tích:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`")

                # Đọc nội dung file
                file_bytes = uploaded_file.getvalue()

                with st.spinner("Đang chuyển đổi file sang ảnh..."):
                    pil_image = binary_to_image(file_bytes, size=IMAGE_SIZE2)

                if pil_image:
                    st.image(pil_image, caption="Biểu diễn hình ảnh của file", width=200)
                else:
                    st.error("Không thể tạo ảnh từ file.")

            with col2:
                if pil_image:
                    with st.spinner("Đang phân tích bằng mô hình..."):
                        predictions = predict(model, pil_image, device, class_names, top_k=top_k_results)

                    if predictions:
                        # Tạo tabs cho tất cả các thông tin
                        result_tabs = st.tabs(["Kết luận", "Top dự đoán", "Chi tiết"])
                        
                        with result_tabs[0]:
                            # Thêm kết luận về mã độc
                            is_malware_result, reason, prob, result_type = is_malware(predictions, benign_classes, threshold)
                            
                            # Hiển thị kết luận với màu sắc tương ứng
                            if is_malware_result and result_type == 1:  # Mã độc với xác suất cao
                                st.markdown('<div class="malware-card">', unsafe_allow_html=True)
                                st.markdown(f"### ⚠️ KẾT LUẬN: File là mã độc")
                                st.markdown(f"**Loại:** {reason}")
                                st.markdown(f"**Xác suất:** {prob:.2%}")
                                st.markdown(f"**Mức độ tin cậy:** Cao")
                                st.markdown('</div>', unsafe_allow_html=True)
                            elif not is_malware_result and result_type == 0:  # Lành tính với xác suất cao
                                st.markdown('<div class="benign-card">', unsafe_allow_html=True)
                                st.markdown(f"### ✅ KẾT LUẬN: File không phải mã độc")
                                st.markdown(f"**Loại:** {reason}")
                                st.markdown(f"**Xác suất:** {prob:.2%}")
                                st.markdown(f"**Mức độ tin cậy:** Cao")
                                st.markdown('</div>', unsafe_allow_html=True)
                            else:  # Không chắc chắn
                                st.markdown('<div class="uncertain-card">', unsafe_allow_html=True)
                                st.markdown(f"### ⚠️ KẾT LUẬN: {reason}")
                                st.markdown(f"**Xác suất:** {prob:.2%}")
                                st.markdown(f"**Lưu ý:** Xác suất dưới ngưỡng {threshold:.2%}, kết quả có thể không chính xác")
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        with result_tabs[1]:
                            # Hiển thị kết quả dự đoán
                            df_predictions = pd.DataFrame([{"Lớp": p["Lớp"], "Xác suất": f"{p['Xác suất']:.4f}"} for p in predictions])
                            st.dataframe(df_predictions, use_container_width=True, height=300)
                            
                        with result_tabs[2]:
                            # Thêm phân tích chi tiết
                            with st.expander("Xem phân tích chi tiết"):
                                # Tạo tabs cho phân tích chi tiết
                                detail_tabs = st.tabs(["Tổng quan", "Phân tích tĩnh", "Phân bố byte", "Chuỗi đáng chú ý"])
                            
                                with detail_tabs[0]:
                                    # Tab tổng quan
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("#### Thông tin cơ bản")
                                        st.markdown(f"**MD5:** `{hashlib.md5(file_bytes).hexdigest()}`")
                                        st.markdown(f"**SHA-1:** `{hashlib.sha1(file_bytes).hexdigest()}`")
                                        st.markdown(f"**SHA-256:** `{hashlib.sha256(file_bytes).hexdigest()}`")
                                        st.markdown(f"**Kích thước:** `{len(file_bytes):,}` bytes")
                                        # Tính entropy
                                        entropy = calculate_entropy(file_bytes)
                                        st.markdown(f"**Entropy:** `{entropy:.4f}/8.0`")
                                        # Đánh giá entropy
                                        if entropy < 6.0:
                                            entropy_eval = "Thấp (file thông thường)"
                                        elif entropy < 7.0:
                                            entropy_eval = "Trung bình (có thể nén/mã hóa một phần)"
                                        else:
                                            entropy_eval = "Cao (có thể được nén/mã hóa/đóng gói)"
                                        st.markdown(f"**Đánh giá entropy:** {entropy_eval}")
                                    with col2:
                                        st.markdown("#### Đánh giá mối đe dọa")
                                        # Tạo thang điểm đe dọa dựa trên xác suất và entropy
                                        threat_score = int((prob * 0.7 + min(entropy/8.0, 1.0) * 0.3) * 10)
                                        # Hiển thị thang điểm đe dọa
                                        threat_color = "red" if threat_score >= 7 else "orange" if threat_score >= 4 else "green"
                                        st.markdown(f"""
                                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
                                            <h5>Điểm đe dọa: <span style="color: {threat_color};">{threat_score}/10</span></h5>
                                            <div style="background-color: #e9ecef; height: 20px; border-radius: 10px; overflow: hidden;">
                                                <div style="width: {threat_score*10}%; background-color: {threat_color}; height: 100%;"></div>
                                            </div>
                                            <p style="margin-top: 10px; font-size: 0.9em;">Dựa trên xác suất phát hiện và các đặc điểm tĩnh</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Hiển thị các cảnh báo
                                        if is_malware_result:
                                            if entropy > 7.0:
                                                st.warning("⚠️ Entropy cao kết hợp với dự đoán mã độc là dấu hiệu đáng ngờ!")
                                            if prob > 0.9:
                                                st.error("🔴 Xác suất phát hiện mã độc rất cao!")
                                
                                with detail_tabs[1]:
                                    # Tab phân tích tĩnh
                                    st.markdown("#### Phân tích tĩnh")
                                    
                                    # Phân tích header PE nếu là file PE
                                    try:
                                        import pefile
                                        pe = pefile.PE(data=file_bytes)
                                        
                                        # Thông tin cơ bản về PE
                                        st.markdown("##### Thông tin PE Header")
                                        
                                        # Hiển thị thông tin Machine, TimeDateStamp, Characteristics
                                        timestamp = datetime.datetime.fromtimestamp(pe.FILE_HEADER.TimeDateStamp)
                                        
                                        pe_info = {
                                            "Machine": f"0x{pe.FILE_HEADER.Machine:04X} ({pefile.MACHINE_TYPE.get(pe.FILE_HEADER.Machine, 'Unknown')})",
                                            "Số section": pe.FILE_HEADER.NumberOfSections,
                                            "Thời gian biên dịch": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                            "Characteristics": f"0x{pe.FILE_HEADER.Characteristics:04X}"
                                        }
                                        
                                        for key, value in pe_info.items():
                                            st.markdown(f"**{key}:** `{value}`")
                                        
                                        # Hiển thị thông tin về sections
                                        st.markdown("##### Sections")
                                        sections_data = []
                                        
                                        for section in pe.sections:
                                            section_name = section.Name.decode('utf-8', 'ignore').strip('\x00')
                                            section_entropy = section.get_entropy()
                                            sections_data.append({
                                                "Tên": section_name,
                                                "Virtual Size": f"0x{section.Misc_VirtualSize:08X}",
                                                "Virtual Address": f"0x{section.VirtualAddress:08X}",
                                                "Raw Size": f"0x{section.SizeOfRawData:08X}",
                                                "Entropy": f"{section_entropy:.4f}",
                                                "Đánh giá": "Có thể đóng gói/mã hóa" if section_entropy > 7.0 else "Bình thường"
                                            })
                                        
                                        st.table(pd.DataFrame(sections_data))
                                        
                                        # Hiển thị thông tin về imports
                                        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                                            st.markdown("##### Imports")
                                            imports_data = []
                                            
                                            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                                                dll_name = entry.dll.decode('utf-8', 'ignore')
                                                for imp in entry.imports[:10]:  # Giới hạn 10 imports mỗi DLL
                                                    if imp.name:
                                                        imp_name = imp.name.decode('utf-8', 'ignore')
                                                    else:
                                                        imp_name = f"Ordinal {imp.ordinal}"
                                                    imports_data.append({
                                                        "DLL": dll_name,
                                                        "Function": imp_name,
                                                        "Address": f"0x{imp.address:08X}"
                                                    })
                                            
                                            # Hiển thị 20 imports đầu tiên
                                            st.dataframe(pd.DataFrame(imports_data[:20]))
                                            
                                            if len(imports_data) > 20:
                                                st.info(f"Hiển thị 20/{len(imports_data)} imports. Mở rộng để xem tất cả.")
                                                with st.expander("Xem tất cả imports"):
                                                    st.dataframe(pd.DataFrame(imports_data))
                                        
                                    except Exception as e:
                                        st.warning(f"Không thể phân tích file PE: {str(e)}")
                                        
                                        # Hiển thị thông tin hex dump nếu không phân tích được PE
                                        st.markdown("##### Hex Dump (64 bytes đầu tiên)")
                                        hex_dump = ' '.join([f"{b:02X}" for b in file_bytes[:64]])
                                        ascii_dump = ''.join([chr(b) if 32 <= b <= 126 else '.' for b in file_bytes[:64]])
                                        
                                        st.code(f"Hex: {hex_dump}\nASCII: {ascii_dump}")
                                
                                with detail_tabs[2]:
                                    # Tab phân bố byte
                                    st.markdown("#### Phân bố byte")
                                    byte_counts = np.bincount(np.frombuffer(file_bytes, dtype=np.uint8), minlength=256)
                                    byte_df = pd.DataFrame({
                                        'Byte': range(256),
                                        'Tần suất': byte_counts
                                    })
                                    
                                    # Thêm cột ASCII để hiển thị ký tự tương ứng
                                    byte_df['ASCII'] = byte_df['Byte'].apply(lambda x: chr(x) if 32 <= x <= 126 else '.')
                                    
                                    # Hiển thị biểu đồ phân bố byte
                                    st.bar_chart(byte_df.set_index('Byte')['Tần suất'])
                                    
                                    # Hiển thị thống kê về byte phổ biến nhất
                                    st.markdown("##### Byte phổ biến nhất")
                                    top_bytes = byte_df.sort_values('Tần suất', ascending=False).head(10)
                                    top_bytes['Phần trăm'] = top_bytes['Tần suất'] / len(file_bytes) * 100
                                    st.table(top_bytes[['Byte', 'ASCII', 'Tần suất', 'Phần trăm']])
                                    
                                with detail_tabs[3]:
                                    # Tab chuỗi đáng chú ý
                                    st.markdown("#### Chuỗi đáng chú ý")
                                    
                                    # Trích xuất chuỗi ASCII từ file
                                    import re
                                    ascii_strings = re.findall(b'[ -~]{5,}', file_bytes)
                                    ascii_strings = [s.decode('ascii', errors='ignore') for s in ascii_strings]
                                    
                                    # Danh sách các từ khóa đáng ngờ
                                    suspicious_keywords = [
                                        'http://', 'https://', 'cmd.exe', 'powershell', 'registry', 'RegCreateKey',
                                        'CreateProcess', 'VirtualAlloc', 'AES', 'RC4', 'XOR', 'URLDownload',
                                        'WinExec', 'ShellExecute', 'WriteProcessMemory', 'CreateRemoteThread',
                                        'SetWindowsHook', 'GetProcAddress', 'LoadLibrary', 'WSASocket',
                                        'InternetOpen', 'InternetConnect', 'InternetReadFile', 'InternetWriteFile'
                                    ]
                                    
                                    # Lọc chuỗi đáng ngờ
                                    suspicious_strings = []
                                    for string in ascii_strings:
                                        for keyword in suspicious_keywords:
                                            if keyword.lower() in string.lower():
                                                suspicious_strings.append({
                                                    'Chuỗi': string,
                                                    'Từ khóa': keyword
                                                })
                                                break
                                    
                                    # Hiển thị chuỗi đáng ngờ
                                    if suspicious_strings:
                                        st.markdown("##### Chuỗi đáng ngờ")
                                        st.dataframe(pd.DataFrame(suspicious_strings))
                                        
                                        if any('http://' in s['Chuỗi'] or 'https://' in s['Chuỗi'] for s in suspicious_strings):
                                            st.warning("⚠️ Phát hiện URL - có thể liên quan đến C&C hoặc tải xuống")
                                        
                                        if any('cmd.exe' in s['Chuỗi'] or 'powershell' in s['Chuỗi'] for s in suspicious_strings):
                                            st.warning("⚠️ Phát hiện lệnh shell - có thể thực thi mã độc")
                                    else:
                                        st.info("Không phát hiện chuỗi đáng ngờ")
                                    
                                    # Thay thế expander bằng checkbox để hiển thị tất cả chuỗi
                                    show_all_strings = st.checkbox("Hiển thị tất cả chuỗi", key="show_all_strings")
                                    
                                    if show_all_strings:
                                        # Giới hạn số lượng chuỗi hiển thị
                                        max_strings = 100
                                        limited_strings = ascii_strings[:max_strings]
                                        
                                        # Tạo container để hiển thị chuỗi
                                        string_container = st.container()
                                        with string_container:
                                            for i, string in enumerate(limited_strings):
                                                st.text(f"{i+1}. {string}")
                                        
                                        if len(ascii_strings) > max_strings:
                                            st.info(f"Hiển thị {max_strings}/{len(ascii_strings)} chuỗi")

            st.markdown(
                """
                <style>
                    .info-text {
                        color: #888888; /* Màu xám nhạt */
                        font-size: 16px;
                        opacity: 0.7;   /* Chìm nhẹ */
                    }
                </style>
                <div class="info-text">⚠️ Lưu ý: Hệ thống sử dụng các thuật toán và mô hình trí tuệ nhân tạo để phát hiện mã độc, các kết quả phân tích không thể đảm bảo chính xác tuyệt đối trong mọi trường hợp!</div>
                """,
                unsafe_allow_html=True
            )


# --- Tab quét thư mục ---
with tab2:
    st.markdown('<div class="sub-header">📁 Quét thư mục chứa file .exe và .dll</div>', unsafe_allow_html=True)
    
    # Kiểm tra xem mô hình đã được tải chưa
    if model is None:
        st.warning("⚠️ Vui lòng tải mô hình trước khi quét thư mục!")
        st.info("💡 Hãy chọn nguồn mô hình trong sidebar và tải mô hình lên.")
    else:
        # Đặt giá trị mặc định cho các biến
        show_clean_files = False
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            option = st.radio("Chọn cách tải thư mục", ["Tải lên file ZIP", "Nhập đường dẫn thư mục"])
        
        with col2:
            scan_button = st.button("🔍 Bắt đầu quét", type="primary", use_container_width=True)

        if option == "Tải lên file ZIP":
            zip_file = st.file_uploader("Chọn file ZIP chứa các file cần quét", type=['zip'])
            if zip_file is not None and scan_button:
                with st.spinner("Đang giải nén file ZIP..."):
                    temp_dir = extract_zip_to_temp(zip_file)
                    st.session_state['scan_dir'] = temp_dir
                    st.session_state['do_scan'] = True
                    
        elif option == "Nhập đường dẫn thư mục":
            folder_path = st.text_input("Nhập đường dẫn đến thư mục cần quét", 
                                placeholder="Ví dụ: C:\\Windows\\System32")
        
            if scan_button:
                if not folder_path:
                    st.markdown("""
                        <style>
                            .toast-alert {
                                animation: fadeInDown 0.5s;
                                background: #fff;
                                border-left: 5px solid #d7263d;
                                color: #222;
                                padding: 16px 22px;
                                border-radius: 8px;
                                font-weight: 500;
                                margin-top: 10px;
                                box-shadow: 0 6px 18px rgba(215,38,61,0.07);
                                font-size: 17px;
                                display: flex;
                                align-items: center;
                                gap: 10px;
                            }
                            @keyframes fadeInDown {
                                0% { opacity: 0; transform: translateY(-20px);}
                                100% { opacity: 1; transform: translateY(0);}
                            }
                        </style>
                        <div class="toast-alert">
                            <svg width="22" height="22" fill="none" style="margin-right:6px;"><circle cx="11" cy="11" r="11" fill="#d7263d"/><text x="11" y="16" font-size="13" text-anchor="middle" fill="#fff">!</text></svg>
                            Vui lòng nhập đường dẫn thư mục!
                        </div>
                    """, unsafe_allow_html=True)
                elif not os.path.isdir(folder_path):
                    st.markdown("""
                        <style>
                            .toast-alert {
                                animation: fadeInDown 0.5s;
                                background: #fff;
                                border-left: 5px solid #f39c12;
                                color: #222;
                                padding: 16px 22px;
                                border-radius: 8px;
                                font-weight: 500;
                                margin-top: 10px;
                                box-shadow: 0 6px 18px rgba(243,156,18,0.10);
                                font-size: 17px;
                                display: flex;
                                align-items: center;
                                gap: 10px;
                            }
                            @keyframes fadeInDown {
                                0% { opacity: 0; transform: translateY(-20px);}
                                100% { opacity: 1; transform: translateY(0);}
                            }
                        </style>
                        <div class="toast-alert">
                            <svg width="22" height="22" fill="none"
                            <svg width="22" height="22" fill="none" style="margin-right:6px;"><circle cx="11" cy="11" r="11" fill="#f39c12"/><text x="11" y="16" font-size="13" text-anchor="middle" fill="#fff">!</text></svg>
                            Đường dẫn thư mục không tồn tại!
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.session_state['scan_dir'] = folder_path
                    st.session_state['do_scan'] = True

        # # Cấu hình quét nâng cao
        # with st.expander("⚙️ Cấu hình quét nâng cao"):
        #     col1, col2, col3 = st.columns(3)
            
        #     with col1:
        #         min_size_kb = st.number_input("Kích thước file tối thiểu (KB)", value=0, min_value=0, max_value=1000)
                
        #     with col2:
        #         max_size_mb = st.number_input("Kích thước file tối đa (MB)", value=max_file_size, min_value=1, max_value=max_file_size)
                
        #     with col3:
        #         analysis_depth = st.selectbox("Độ sâu phân tích", ["Nhanh", "Cân bằng", "Sâu"])

        # Thực hiện quét nếu có yêu cầu
        if model is not None and 'do_scan' in st.session_state and st.session_state['do_scan']:
            scan_dir = st.session_state['scan_dir']
            with st.spinner("Đang quét thư mục..."):
                results, malware_count, uncertain_count, total_files = scan_directory(
                    scan_dir, model, class_names, device, benign_classes, threshold
                )
            st.session_state['do_scan'] = False
            if results:
                    # Hiển thị tổng quan kết quả
                    st.markdown('<div class="sub-header">📊 Tổng quan kết quả</div>', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Tổng số file", total_files)
                    
                    with col2:
                        st.metric("File mã độc", malware_count, delta=f"{malware_count/total_files*100:.1f}%" if total_files > 0 else "0%")
                    
                    with col3:
                        benign_count = total_files - malware_count - uncertain_count
                        st.metric("File lành tính", benign_count, delta=f"{benign_count/total_files*100:.1f}%" if total_files > 0 else "0%")
                    
                    with col4:
                        st.metric("Không chắc chắn", uncertain_count, delta=f"{uncertain_count/total_files*100:.1f}%" if total_files > 0 else "0%")

<<<<<<< HEAD
                    # Tạo biểu đồ tròn
                    if total_files > 0:
                        benign_count = total_files - malware_count - uncertain_count
                        sizes = [malware_count, max(0, benign_count), uncertain_count]
                        if any(x < 0 for x in sizes):
                            sizes = [max(0, x) for x in sizes]
                        if sum(sizes) == 0:
                            st.warning("Không có dữ liệu để vẽ biểu đồ tròn.")
                        else:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            labels = ['Mã độc', 'Lành tính', 'Không chắc chắn']
                            colors = ['#ff6b6b', '#51cf66', '#ffd43b']
                            explode = (0.1, 0, 0)
                            ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                                   shadow=True, startangle=90)
                            ax.set_title('Phân bố kết quả quét')
                            st.pyplot(fig)
                            plt.close()
=======
                    # TÍNH NĂNG MỚI: Thêm biểu đồ phân bố loại mã độc
                    if results:
                        # Tất cả các code duyệt results nằm ở đây!
                        if malware_count > 0:
                            st.markdown("### 📊 Phân bố loại mã độc")
                            malware_types = {}
                            for r in results:
                
                            # Tạo DataFrame cho biểu đồ
                                malware_types = {}
                                for r in results:
                                    if r["Là mã độc"] and r["Kết quả"] == 1:
                                        malware_type = r["Top dự đoán"][0]["Lớp"]
                                        if malware_type in malware_types:
                                            malware_types[malware_type] += 1
                                        else:
                                            malware_types[malware_type] = 1
                        
                        if malware_types:
                            malware_df = pd.DataFrame({
                                'Loại mã độc': list(malware_types.keys()),
                                'Số lượng': list(malware_types.values())
                            })
                            
                        # Hiển thị biểu đồ
                        col1, col2 = st.columns(2)
                        # BIỂU ĐỒ CỘT
                        with col1:
                            num_types = len(malware_df['Loại mã độc'])
                            cmap = plt.get_cmap('tab10')
                            colors = [cmap(i % cmap.N) for i in range(num_types)]
                            
                            fig, ax = plt.subplots(figsize=(8, 5))  # Giữ nguyên kích thước
                            bars = ax.bar(
                                malware_df['Loại mã độc'],
                                malware_df['Số lượng'],
                                width=0.3,
                                color=colors,        # Thêm dòng này để set màu theo tab10
                                edgecolor='gray',    # (tùy chọn) thêm đường viền cho chuyên nghiệp
                                linewidth=0.7
                            )
                            ax.set_ylabel('Số lượng', fontsize=10)
                            ax.set_title('Phân bố loại mã độc', fontsize=11)
                            ax.tick_params(axis='x', labelsize=9)
                            ax.tick_params(axis='y', labelsize=9)
                            plt.xticks(rotation=90, ha='right', fontsize=9)
                            plt.yticks(fontsize=9)
                            plt.tight_layout()
                            st.pyplot(fig)


                        # BIỂU ĐỒ TRÒN
                        with col2:
                            fig, ax = plt.subplots(figsize=(5, 3))
                            def small_pct(pct):
                                return f'{pct:.1f}%' if pct > 2 else ''  # chỉ hiển thị nếu >2%
                            wedges, texts, autotexts = ax.pie(
                                malware_df['Số lượng'],
                                labels=malware_df['Loại mã độc'],
                                autopct=small_pct,
                                textprops={'fontsize': 9}
                            )
                            for autotext in autotexts:
                                autotext.set_fontsize(8)  # font nhỏ cho %
                            for text in texts:
                                text.set_fontsize(9)      # font nhỏ cho label
                            ax.axis('equal')
                            plt.tight_layout()
                            st.pyplot(fig)

                    # # Tạo biểu đồ tròn
                    # if total_files > 0:
                    #     benign_count = total_files - malware_count - uncertain_count
                    #     sizes = [malware_count, max(0, benign_count), uncertain_count]
                    #     if any(x < 0 for x in sizes):
                    #         sizes = [max(0, x) for x in sizes]
                    #     if sum(sizes) == 0:
                    #         st.warning("Không có dữ liệu để vẽ biểu đồ tròn.")
                    #     else:
                    #         fig, ax = plt.subplots(figsize=(8, 6))
                    #         labels = ['Mã độc', 'Lành tính', 'Không chắc chắn']
                    #         colors = ['#ff6b6b', '#51cf66', '#ffd43b']
                    #         explode = (0.1, 0, 0)
                    #         ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                    #                shadow=True, startangle=90)
                    #         ax.set_title('Phân bố kết quả quét')
                    #         st.pyplot(fig)
                    #         plt.close()
>>>>>>> 58fc52c (Cập nhật code: sửa bug/thêm tính năng XYZ)

                    # Hiển thị kết quả chi tiết
                    st.markdown('<div class="sub-header">📋 Kết quả chi tiết</div>', unsafe_allow_html=True)
                    
                    # Tạo tabs cho các loại kết quả
                    if malware_count > 0:
                        result_tabs = st.tabs(["🔴 Mã độc", "✅ Lành tính", "⚠️ Không chắc chắn", "📊 Tất cả"])
                    else:
                        result_tabs = st.tabs(["✅ Lành tính", "⚠️ Không chắc chắn", "📊 Tất cả"])
                    
                    # Phân loại kết quả
                    
                    malware_files = [r for r in results if r["Là mã độc"] and r["Kết quả"] == 1]
                    benign_files = [r for r in results if not r["Là mã độc"] and r["Kết quả"] == 0]
                    uncertain_files = [r for r in results if r["Kết quả"] == 2]
                    
                    tab_index = 0
                    
                    # Tab mã độc (chỉ hiển thị nếu có)
                    if malware_count > 0:
                        with result_tabs[tab_index]:
                            if malware_files:
                                st.error(f"⚠️ Phát hiện {len(malware_files)} file mã độc!")
                                
                                # Hiển thị danh sách file mã độc
                                malware_df = pd.DataFrame([{
                                    "Tên file": r["Tên file"],
                                    "Kích thước (KB)": r["Kích thước (KB)"],
                                    "Loại mã độc": r["Loại"],
                                    "Xác suất": f"{r['Xác suất']:.2%}",
                                    "Đường dẫn": r["Đường dẫn"]
                                } for r in malware_files])
                                
                                st.dataframe(malware_df, use_container_width=True)
                                
                                # Nút tải xuống danh sách mã độc
                                csv_malware = malware_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Tải xuống danh sách mã độc (CSV)",
                                    data=csv_malware,
                                    file_name=f"malware_detected_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                        tab_index += 1
                    
                    # Tab lành tính
                    with result_tabs[tab_index]:
                        if benign_files:
                            st.success(f"✅ {len(benign_files)} file được xác định là lành tính")
                            
                            # Checkbox để hiển thị file lành tính
                            # show_clean_files = st.checkbox("Hiển thị danh sách file lành tính", value=False)
                            # show_clean_files = st.checkbox("Hiển thị danh sách file lành tính", value=False, key="show_clean_files_benign_tab")
                            
                            if show_clean_files:
                                benign_df = pd.DataFrame([{
                                "Tên file": r["Tên file"],
                                "Kích thước (KB)": r["Kích thước (KB)"],
                                "Loại": r["Loại"],
                                "Kết quả": r["Kết quả"],    # Thêm dòng này để debug
                                "Xác suất": f"{r['Xác suất']:.2%}",
                                "Đường dẫn": r["Đường dẫn"]
                            } for r in benign_files])
                                
                                st.dataframe(benign_df, use_container_width=True)
                        else:
                            st.info("Không có file nào được xác định là lành tính với độ tin cậy cao")
                    tab_index += 1
                    
                    # Tab không chắc chắn
                    with result_tabs[tab_index]:
                        if uncertain_files:
                            st.warning(f"⚠️ {len(uncertain_files)} file có kết quả không chắc chắn")
                            st.info("Những file này cần được kiểm tra thêm bằng các công cụ khác")
                            
                            uncertain_df = pd.DataFrame([{
                                "Tên file": r["Tên file"],
                                "Kích thước (KB)": r["Kích thước (KB)"],
                                "Loại": r["Loại"],
                                "Xác suất": f"{r['Xác suất']:.2%}",
                                "Đường dẫn": r["Đường dẫn"]
                            } for r in uncertain_files])
                            
                            st.dataframe(uncertain_df, use_container_width=True)
                        else:
                            st.success("Không có file nào có kết quả không chắc chắn")
                    
                    tab_index += 1
                    
                    # Tab tất cả
                    with result_tabs[tab_index]:
                        st.info(f"Hiển thị tất cả {len(results)} file đã quét")
                        
                        # Tạo DataFrame với tất cả kết quả
                        all_results_df = pd.DataFrame([{
                            "Tên file": r["Tên file"],
                            "Kích thước (KB)": r["Kích thước (KB)"],
                            "Kết quả": "Mã độc" if r["Là mã độc"] and r["Kết quả"] == 1 else 
                                     "Lành tính" if not r["Là mã độc"] and r["Kết quả"] == 0 else "Không chắc chắn",
                            "Loại": r["Loại"],
                            "Xác suất": f"{r['Xác suất']:.2%}",
                            "Đường dẫn": r["Đường dẫn"]
                        } for r in results])
                        
                        st.dataframe(all_results_df, use_container_width=True)
                        
                        # Nút tải xuống tất cả kết quả
                        csv_all = all_results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Tải xuống tất cả kết quả (CSV)",
                            data=csv_all,
                            file_name=f"scan_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

                    # # Thống kê nâng cao
                    # if analysis_depth == "Sâu":
                    #     st.markdown('<div class="sub-header">📈 Thống kê nâng cao</div>', unsafe_allow_html=True)
                        
                        # # Phân tích entropy
                        # entropy_values = [r["Entropy"] for r in scan_results if r["Entropy"] is not None]
                        # if entropy_values:
                        #     col1, col2 = st.columns(2)
                            
                        #     with col1:
                        #         st.markdown("##### Phân bố Entropy")
                        #         fig, ax = plt.subplots(figsize=(8, 4))
                        #         ax.hist(entropy_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                        #         ax.set_xlabel('Entropy')
                        #         ax.set_ylabel('Số lượng file')
                        #         ax.set_title('Phân bố Entropy của các file')
                        #         st.pyplot(fig)
                        #         plt.close()
                            
                        #     with col2:
                        #         st.markdown("##### Thống kê Entropy")
                        #         st.write(f"**Entropy trung bình:** {np.mean(entropy_values):.4f}")
                        #         st.write(f"**Entropy cao nhất:** {np.max(entropy_values):.4f}")
                        #         st.write(f"**Entropy thấp nhất:** {np.min(entropy_values):.4f}")
                        #         st.write(f"**Độ lệch chuẩn:** {np.std(entropy_values):.4f}")
                                
                        #         # Cảnh báo về entropy cao
                        #         high_entropy_files = [r for r in scan_results if r["Entropy"] and r["Entropy"] > 7.0]
                        #         if high_entropy_files:
                        #             st.warning(f"⚠️ {len(high_entropy_files)} file có entropy > 7.0 (có thể được đóng gói/mã hóa)")

                        # else:
                        #     st.warning("Không tìm thấy file .exe hoặc .dll nào trong thư mục được chỉ định.")
            
            # Dọn dẹp thư mục tạm nếu cần
            if option == "Tải lên file ZIP" and 'scan_dir' in st.session_state:
                try:
                    shutil.rmtree(st.session_state['scan_dir'])
                    del st.session_state['scan_dir']
                except:
                    pass
<<<<<<< HEAD
=======

            st.markdown(
                """
                <style>
                    .info-text {
                        color: #888888; /* Màu xám nhạt */
                        font-size: 16px;
                        opacity: 0.7;   /* Chìm nhẹ */
                    }
                </style>
                <div class="info-text">⚠️ Lưu ý: Hệ thống sử dụng các thuật toán và mô hình trí tuệ nhân tạo để phát hiện mã độc, các kết quả phân tích không thể đảm bảo chính xác tuyệt đối trong mọi trường hợp!</div>
                """,
                unsafe_allow_html=True
            )
      
>>>>>>> 58fc52c (Cập nhật code: sửa bug/thêm tính năng XYZ)
with tab3:
     # Hiển thị bản đồ Việt Nam với thông tin về mối đe dọa
        st.markdown('<div class="map-header">🗺️ Giám sát an ninh mạng quốc gia: Bản đồ Việt Nam</div>', unsafe_allow_html=True)

        # Tạo HTML cho bản đồ
        map_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vietnam Cybersecurity Threat Map</title>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
            <style>
                body { margin: 0; padding: 0; }
                #map { height: 500px; width: 100%; }
                .threat-info {
                    background: rgba(255, 255, 255, 0.9);
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                }
                .threat-level-high { color: #d32f2f; font-weight: bold; }
                .threat-level-medium { color: #f57c00; font-weight: bold; }
                .threat-level-low { color: #388e3c; font-weight: bold; }
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
            <script>
                // Khởi tạo bản đồ tập trung vào Việt Nam
                var map = L.map('map').setView([16.0583, 108.2772], 6);
                // Thêm tile layer
                L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {attribution: '© OpenStreetMap, © CartoDB'}).addTo(map);
                // Dữ liệu mối đe dọa giả lập 63 tỉnh/thành Việt Nam 2024
                var threatData = [
                    // Nhóm High (từ 700+)
                    {city: "TP. Hồ Chí Minh", lat: 10.7769, lng: 106.7009, threats: 2100, level: "high", malware: 750, phishing: 520, ddos: 315, other: 515},
                    {city: "Hà Nội", lat: 21.0285, lng: 105.8542, threats: 1480, level: "high", malware: 525, phishing: 365, ddos: 212, other: 378},
                    {city: "Bình Dương", lat: 10.9804, lng: 106.6519, threats: 855, level: "high", malware: 312, phishing: 201, ddos: 121, other: 221},
                    {city: "Đồng Nai", lat: 10.9452, lng: 106.8246, threats: 802, level: "high", malware: 285, phishing: 189, ddos: 116, other: 212},

                    // Nhóm Medium (300–700)
                    {city: "Hải Phòng", lat: 20.8449, lng: 106.6881, threats: 626, level: "medium", malware: 213, phishing: 168, ddos: 104, other: 141},
                    {city: "Cần Thơ", lat: 10.0452, lng: 105.7469, threats: 563, level: "medium", malware: 186, phishing: 153, ddos: 80, other: 144},
                    {city: "Đà Nẵng", lat: 16.0544, lng: 108.2022, threats: 488, level: "medium", malware: 159, phishing: 126, ddos: 74, other: 129},
                    {city: "Thanh Hóa", lat: 19.8072, lng: 105.7768, threats: 473, level: "medium", malware: 154, phishing: 121, ddos: 77, other: 121},
                    {city: "Nghệ An", lat: 18.6796, lng: 105.6813, threats: 452, level: "medium", malware: 147, phishing: 115, ddos: 70, other: 120},
                    {city: "Quảng Ninh", lat: 21.0064, lng: 107.2925, threats: 441, level: "medium", malware: 141, phishing: 112, ddos: 66, other: 122},
                    {city: "Thừa Thiên Huế", lat: 16.4637, lng: 107.5909, threats: 407, level: "medium", malware: 135, phishing: 98, ddos: 62, other: 112},
                    {city: "Hải Dương", lat: 20.9401, lng: 106.3336, threats: 396, level: "medium", malware: 123, phishing: 101, ddos: 62, other: 110},
                    {city: "Bắc Ninh", lat: 21.1861, lng: 106.0763, threats: 384, level: "medium", malware: 118, phishing: 95, ddos: 66, other: 105},
                    {city: "Thái Bình", lat: 20.4509, lng: 106.3406, threats: 376, level: "medium", malware: 112, phishing: 92, ddos: 65, other: 107},
                    {city: "Vĩnh Phúc", lat: 21.3081, lng: 105.6046, threats: 372, level: "medium", malware: 109, phishing: 90, ddos: 62, other: 111},
                    {city: "Bắc Giang", lat: 21.2731, lng: 106.1946, threats: 361, level: "medium", malware: 106, phishing: 84, ddos: 66, other: 105},
                    {city: "Nam Định", lat: 20.4388, lng: 106.1621, threats: 358, level: "medium", malware: 103, phishing: 88, ddos: 61, other: 106},
                    {city: "Phú Thọ", lat: 21.3457, lng: 105.2120, threats: 347, level: "medium", malware: 100, phishing: 85, ddos: 55, other: 107},
                    {city: "Quảng Nam", lat: 15.5394, lng: 108.0191, threats: 341, level: "medium", malware: 98, phishing: 83, ddos: 54, other: 106},

                    // Nhóm Low (dưới 300)
                    {city: "Bình Định", lat: 13.7820, lng: 109.2191, threats: 298, level: "low", malware: 85, phishing: 76, ddos: 46, other: 91},
                    {city: "Quảng Ngãi", lat: 15.1202, lng: 108.7922, threats: 286, level: "low", malware: 84, phishing: 63, ddos: 43, other: 96},
                    {city: "Lâm Đồng", lat: 11.5753, lng: 108.1429, threats: 275, level: "low", malware: 78, phishing: 59, ddos: 40, other: 98},
                    {city: "Kiên Giang", lat: 10.0086, lng: 105.0807, threats: 265, level: "low", malware: 75, phishing: 58, ddos: 36, other: 96},
                    {city: "Long An", lat: 10.5435, lng: 106.4106, threats: 257, level: "low", malware: 72, phishing: 55, ddos: 38, other: 92},
                    {city: "Bến Tre", lat: 10.2415, lng: 106.3754, threats: 243, level: "low", malware: 69, phishing: 52, ddos: 33, other: 89},
                    {city: "An Giang", lat: 10.5216, lng: 105.1259, threats: 239, level: "low", malware: 67, phishing: 54, ddos: 31, other: 87},
                    {city: "Đắk Lắk", lat: 12.7100, lng: 108.2378, threats: 233, level: "low", malware: 65, phishing: 47, ddos: 32, other: 89},
                    {city: "Tiền Giang", lat: 10.4493, lng: 106.3421, threats: 231, level: "low", malware: 66, phishing: 46, ddos: 31, other: 88},
                    {city: "Bà Rịa - Vũng Tàu", lat: 10.5418, lng: 107.2428, threats: 228, level: "low", malware: 64, phishing: 49, ddos: 28, other: 87},
                    {city: "Quảng Bình", lat: 17.4689, lng: 106.6228, threats: 224, level: "low", malware: 62, phishing: 44, ddos: 31, other: 87},
                    {city: "Tây Ninh", lat: 11.3352, lng: 106.1099, threats: 220, level: "low", malware: 61, phishing: 43, ddos: 29, other: 87},
                    {city: "Thái Nguyên", lat: 21.5672, lng: 105.8252, threats: 217, level: "low", malware: 60, phishing: 42, ddos: 28, other: 87},
                    {city: "Vĩnh Long", lat: 10.2536, lng: 105.9722, threats: 211, level: "low", malware: 59, phishing: 41, ddos: 25, other: 86},
                    {city: "Quảng Trị", lat: 16.8187, lng: 107.0917, threats: 207, level: "low", malware: 57, phishing: 40, ddos: 26, other: 84},
                    {city: "Sóc Trăng", lat: 9.6026, lng: 105.9731, threats: 203, level: "low", malware: 56, phishing: 39, ddos: 25, other: 83},
                    {city: "Gia Lai", lat: 13.8079, lng: 108.1095, threats: 200, level: "low", malware: 54, phishing: 37, ddos: 24, other: 85},
                    {city: "Bạc Liêu", lat: 9.2941, lng: 105.7278, threats: 199, level: "low", malware: 54, phishing: 38, ddos: 23, other: 84},
                    {city: "Hà Tĩnh", lat: 18.3559, lng: 105.8875, threats: 197, level: "low", malware: 52, phishing: 38, ddos: 24, other: 83},
                    {city: "Ninh Bình", lat: 20.2506, lng: 105.9745, threats: 194, level: "low", malware: 52, phishing: 36, ddos: 22, other: 84},
                    {city: "Hưng Yên", lat: 20.6463, lng: 106.0511, threats: 192, level: "low", malware: 51, phishing: 35, ddos: 22, other: 84},
                    {city: "Đắk Nông", lat: 12.2644, lng: 107.6098, threats: 188, level: "low", malware: 49, phishing: 34, ddos: 21, other: 84},
                    {city: "Tuyên Quang", lat: 21.8230, lng: 105.2148, threats: 185, level: "low", malware: 49, phishing: 33, ddos: 21, other: 82},
                    {city: "Phú Yên", lat: 13.0882, lng: 109.0929, threats: 181, level: "low", malware: 48, phishing: 31, ddos: 22, other: 80},
                    {city: "Bình Phước", lat: 11.7512, lng: 106.7235, threats: 178, level: "low", malware: 47, phishing: 31, ddos: 20, other: 80},
                    {city: "Vĩnh Long", lat: 10.2536, lng: 105.9722, threats: 175, level: "low", malware: 46, phishing: 30, ddos: 20, other: 79},
                    {city: "Hà Nam", lat: 20.5833, lng: 105.9160, threats: 174, level: "low", malware: 45, phishing: 31, ddos: 19, other: 79},
                    {city: "Yên Bái", lat: 21.7051, lng: 104.8800, threats: 173, level: "low", malware: 44, phishing: 31, ddos: 18, other: 80},
                    {city: "Cà Mau", lat: 9.1768, lng: 105.1500, threats: 172, level: "low", malware: 43, phishing: 30, ddos: 18, other: 81},
                    {city: "Lào Cai", lat: 22.4804, lng: 103.9756, threats: 170, level: "low", malware: 42, phishing: 30, ddos: 18, other: 80},
                    {city: "Kon Tum", lat: 14.3549, lng: 108.0076, threats: 168, level: "low", malware: 41, phishing: 29, ddos: 17, other: 81},
                    {city: "Hòa Bình", lat: 20.8171, lng: 105.3376, threats: 167, level: "low", malware: 41, phishing: 28, ddos: 17, other: 81},
                    {city: "Trà Vinh", lat: 9.9347, lng: 106.3452, threats: 163, level: "low", malware: 41, phishing: 27, ddos: 16, other: 79},
                    {city: "Lạng Sơn", lat: 21.8528, lng: 106.7610, threats: 159, level: "low", malware: 40, phishing: 27, ddos: 16, other: 76},
                    {city: "Quảng Nam", lat: 15.5394, lng: 108.0191, threats: 158, level: "low", malware: 39, phishing: 27, ddos: 15, other: 77},
                    {city: "Bắc Kạn", lat: 22.1485, lng: 105.8348, threats: 156, level: "low", malware: 39, phishing: 25, ddos: 15, other: 77},
                    {city: "Cao Bằng", lat: 22.6666, lng: 106.2579, threats: 154, level: "low", malware: 38, phishing: 25, ddos: 14, other: 77},
                    {city: "Bình Thuận", lat: 11.0904, lng: 108.0721, threats: 153, level: "low", malware: 37, phishing: 26, ddos: 13, other: 77},
                    {city: "Điện Biên", lat: 21.3860, lng: 103.0230, threats: 151, level: "low", malware: 36, phishing: 24, ddos: 14, other: 77},
                    {city: "Ninh Thuận", lat: 11.6739, lng: 109.0147, threats: 149, level: "low", malware: 35, phishing: 24, ddos: 13, other: 77},
                    {city: "Hà Giang", lat: 22.8233, lng: 104.9836, threats: 147, level: "low", malware: 34, phishing: 23, ddos: 13, other: 77},
                    {city: "Quảng Ngãi", lat: 15.1202, lng: 108.7922, threats: 146, level: "low", malware: 33, phishing: 23, ddos: 12, other: 78},
                    {city: "Sơn La", lat: 21.3256, lng: 103.9188, threats: 144, level: "low", malware: 33, phishing: 23, ddos: 12, other: 76},
                    {city: "Bắc Ninh", lat: 21.1861, lng: 106.0763, threats: 143, level: "low", malware: 32, phishing: 22, ddos: 12, other: 77},
                    {city: "Phú Thọ", lat: 21.3457, lng: 105.2120, threats: 139, level: "low", malware: 31, phishing: 21, ddos: 11, other: 76},
                    {city: "Khánh Hòa", lat: 12.2388, lng: 109.1967, threats: 126, level: "low", malware: 27, phishing: 18, ddos: 10, other: 71},
                    {city: "Hậu Giang", lat: 9.7845, lng: 105.4701, threats: 123, level: "low", malware: 26, phishing: 17, ddos: 9, other: 71},
                    // ... (bạn có thể thêm tiếp các huyện/thị xã nếu cần)
                ];

                // Hàm xác định màu sắc dựa trên mức độ đe dọa
                function getThreatColor(level) {
                    switch(level) {
                        case 'high': return '#d32f2f';
                        case 'medium': return '#f57c00';
                        case 'low': return '#388e3c';
                        default: return '#666666';
                    }
                }
                
                // Hàm xác định kích thước marker dựa trên số lượng đe dọa
                function getMarkerSize(threats) {
                    if (threats > 1000) return 25;
                    if (threats > 500) return 15;
                    if (threats > 200) return 10;
                    return 10;
                }
                
                // Thêm markers cho từng thành phố
                threatData.forEach(function(data) {
                    var color = getThreatColor(data.level);
                    var size = getMarkerSize(data.threats);
                    
                    // Tạo custom icon
                    var threatIcon = L.divIcon({
                        className: 'threat-marker',
                        html: '<div style="background-color: ' + color + '; width: ' + size + 'px; height: ' + size + 'px; border-radius: 50%; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>',
                        iconSize: [size, size],
                        iconAnchor: [size/2, size/2]
                    });
                    
                    // Tạo popup content
                    var popupContent = `
                        <div class="threat-info">
                            <h3>${data.city}</h3>
                            <p><strong>Tổng mối đe dọa:</strong> <span class="threat-level-${data.level}">${data.threats}</span></p>
                            <hr>
                            <p><strong>Phân loại:</strong></p>
                            <ul>
                                <li>🦠 Malware: ${data.malware}</li>
                                <li>🎣 Phishing: ${data.phishing}</li>
                                <li>⚡ DDoS: ${data.ddos}</li>
                                <li>🔧 Khác: ${data.other}</li>
                            </ul>
                            <p><strong>Mức độ:</strong> <span class="threat-level-${data.level}">${data.level.toUpperCase()}</span></p>
                        </div>
                    `;
                    
                    // Thêm marker vào bản đồ
                    L.marker([data.lat, data.lng], {icon: threatIcon})
                        .bindPopup(popupContent)
                        .addTo(map);
                });
                
                // Thêm legend
                var legend = L.control({position: 'bottomright'});
                legend.onAdd = function (map) {
                    var div = L.DomUtil.create('div', 'info legend');
                    div.innerHTML = `
                        <div style="background: rgba(255,255,255,0.9); padding: 10px; border-radius: 5px; box-shadow: 0 1px 10px rgba(0,0,0,0.2);">
                            <h4>Mức độ đe dọa</h4>
                            <div><span style="color: #d32f2f;">●</span> Cao (>1000)</div>
                            <div><span style="color: #f57c00;">●</span> Trung bình (200-1000)</div>
                            <div><span style="color: #388e3c;">●</span> Thấp (<200)</div>
                            <hr>
                            <small>Dữ liệu cập nhật: ${new Date().toLocaleDateString('vi-VN')}</small>
                        </div>
                    `;
                    return div;
                };
                legend.addTo(map);
                
                // Thêm thông tin tổng quan
                var info = L.control({position: 'topleft'});
                info.onAdd = function (map) {
                    var div = L.DomUtil.create('div', 'info');
                    var totalThreats = threatData.reduce((sum, data) => sum + data.threats, 0);
                    div.innerHTML = `
                        <div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);">
                            <h3>🇻🇳 Tình hình An ninh mạng Việt Nam</h3>
                            <p><strong>Tổng mối đe dọa:</strong> <span style="color: #d32f2f; font-weight: bold;">${totalThreats}</span></p>
                            <p><strong>Khu vực nguy hiểm nhất:</strong> TP. Hồ Chí Minh</p>
                            <p><strong>Loại đe dọa phổ biến:</strong> Malware</p>
                            <hr>
                            <small>⚠️Dữ liệu sử dụng cho mục đích minh họa</small>
                        </div>
                    `;
                    return div;
                };
                info.addTo(map);

                
                // Thêm hiệu ứng pulse cho các marker có mức đe dọa cao
                threatData.forEach(function(data) {
                    if (data.level === 'high') {
                        var pulseIcon = L.divIcon({
                            className: 'pulse-marker',
                            html: '<div class="pulse-dot"></div>',
                            iconSize: [20, 20],
                            iconAnchor: [10, 10]
                        });
                        
                        L.marker([data.lat, data.lng], {icon: pulseIcon}).addTo(map);
                    }
                });
                
                // CSS cho hiệu ứng pulse
                var style = document.createElement('style');
                style.innerHTML = `
                    .pulse-dot {
                        width: 20px;
                        height: 20px;
                        background-color: #d32f2f;
                        border-radius: 50%;
                        animation: pulse 2s infinite;
                        opacity: 0.8;
                    }
                    
                    @keyframes pulse {
                        0% {
                            transform: scale(0.8);
                            opacity: 1;
                        }
                        50% {
                            transform: scale(1.2);
                            opacity: 0.5;
                        }
                        100% {
                            transform: scale(0.8);
                            opacity: 1;
                        }
                    }
                `;
                document.head.appendChild(style);
                
            </script>
        </body>
        </html>
        """

        # Hiển thị bản đồ
        st.components.v1.html(map_html, height=500)

        # Thêm thông tin cảnh báo bảo mật
        st.markdown('<div class="sub-header">🚨 Cảnh báo Bảo mật</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ff6b6b, #ee5a52); color: white; padding: 20px; border-radius: 10px; text-align: center;">
                <h4>⚠️ Mức độ đe dọa</h4>
                <h2>CAO</h2>
                <p>Phát hiện nhiều mã độc mới</p>
                <small>Cập nhật: Hôm nay</small>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4ecdc4, #44a08d); color: white; padding: 20px; border-radius: 10px; text-align: center;">
                <h4>🛡️ Tỷ lệ phát hiện</h4>
                <h2>94.7%</h2>
                <p>Độ chính xác của hệ thống</p>
                <small>Dựa trên 10,000+ mẫu</small>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #45b7d1, #96c93d); color: white; padding: 20px; border-radius: 10px; text-align: center;">
                <h4>📊 Files đã quét</h4>
                <h2>2,847</h2>
                <p>Trong tháng này</p>
                <small>Tăng 23% so với tháng trước</small>
            </div>
            """, unsafe_allow_html=True)

        # Thêm RSS feed giả lập về tin tức bảo mật
        st.markdown('<div class="sub-header">📰 Tin tức Bảo mật mới nhất</div>', unsafe_allow_html=True)

        news_data = [
            {
                "title": "Phát hiện biến thể mới của ransomware LockBit tại Việt Nam",
                "summary": "Các chuyên gia bảo mật cảnh báo về sự xuất hiện của biến thể mới của ransomware LockBit đang nhắm mục tiêu vào các doanh nghiệp Việt Nam.",
                "time": "2 giờ trước",
                "severity": "high"
            },
            {
                "title": "Cập nhật bản vá bảo mật khẩn cấp cho Windows",
                "summary": "Microsoft phát hành bản vá khẩn cấp để sửa lỗi zero-day đang được khai thác tích cực bởi các nhóm APT.",
                "time": "5 giờ trước", 
                "severity": "high"
            },
            {
                "title": "Chiến dịch phishing mạo danh ngân hàng gia tăng",
                "summary": "Số lượng email phishing mạo danh các ngân hàng lớn tại Việt Nam tăng 45% trong tuần qua.",
                "time": "1 ngày trước",
                "severity": "medium"
            },
            {
                "title": "Hướng dẫn bảo vệ hệ thống khỏi malware mới",
                "summary": "Các biện pháp phòng ngừa và phát hiện sớm các loại malware mới xuất hiện gần đây.",
                "time": "2 ngày trước",
                "severity": "low"
            }
        ]

        for news in news_data:
            severity_color = {"high": "#ff6b6b", "medium": "#ffd43b", "low": "#51cf66"}[news["severity"]]
            severity_text = {"high": "🔴 Nghiêm trọng", "medium": "🟡 Trung bình", "low": "🟢 Thông tin"}[news["severity"]]
            
            st.markdown(f"""
            <div style="border-left: 4px solid {severity_color}; background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 0 5px 5px 0;">
                <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 8px;">
                    <h4 style="margin: 0; color: #333;">{news['title']}</h4>
                    <span style="color: {severity_color}; font-size: 0.9em; font-weight: bold;">{severity_text}</span>
                </div>
                <p style="margin: 8px 0; color: #666; line-height: 1.5;">{news['summary']}</p>
                <small style="color: #999;">⏰ {news['time']}</small>
            </div>
            """, unsafe_allow_html=True)

        # Thêm biểu đồ thống kê thời gian thực
        st.markdown('<div class="sub-header">📈 Thống kê Thời gian thực</div>', unsafe_allow_html=True)

        # Tạo dãy ngày
        dates = pd.date_range(start='2024-07-01', end='2025-06-30', freq='D')

        # Thiết lập seed để tái lập kết quả
        np.random.seed(42)

        # Tạo hệ số dao động cho từng tháng (giả lập xu hướng thực tế: cuối năm tăng, đầu năm giảm)
        monthly_malware_base = {
            7:  14,  8: 13,  9: 16, 10: 19, 11: 22, 12: 28,  # Từ tháng 7 đến tháng 12/2024
            1:  32,  2: 29,  3: 22,  4: 18,  5: 16,  6: 15   # Từ tháng 1 đến tháng 6/2025
        }
        monthly_clean_base = {
            7:  55,  8: 56,  9: 58, 10: 62, 11: 66, 12: 70,
            1:  75,  2: 70,  3: 65,  4: 60,  5: 58,  6: 56
        }

        dates = pd.date_range(start='2024-07-01', end='2025-06-30', freq='D')
        np.random.seed(42)

        malware_detections = []
        clean_files = []
        base_mal = 10
        base_clean = 50
        trend_increase = 0.08  # mức tăng nhẹ theo ngày

        for i, d in enumerate(dates):
            # Dao động mạnh quanh giá trị trung bình nhưng vẫn tăng dần theo thời gian
            daily_mal = base_mal + (i * trend_increase) + np.random.normal(0, 5)
            daily_clean = base_clean + (i * trend_increase * 2) + np.random.normal(0, 12)
            malware_detections.append(max(0, int(daily_mal)))
            clean_files.append(max(0, int(daily_clean)))

        stats_df = pd.DataFrame({
            'Ngày': dates,
            'Mã độc phát hiện': malware_detections,
            'File lành tính': clean_files,
            'Tổng file quét': np.array(malware_detections) + np.array(clean_files)
        })

        # Optional: Xem thử thống kê theo tháng
        #print(stats_df.groupby(stats_df['Ngày'].dt.month)[['Mã độc phát hiện', 'File lành tính']].mean())

        # Tạo tabs cho các biểu đồ khác nhau
        chart_tabs = st.tabs(["📊 Tổng quan", "🦠 Mã độc", "📈 Xu hướng", "🌍 Phân bố địa lý"])

        with chart_tabs[0]:
            
            # Biểu đồ cột
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Lấy dữ liệu theo tháng - SỬA LỖI Ở ĐÂY
            monthly_stats = stats_df.groupby(stats_df['Ngày'].dt.to_period('M')).agg({
                'Mã độc phát hiện': 'sum',
                'File lành tính': 'sum',
                'Tổng file quét': 'sum'
            })
            
            x = range(len(monthly_stats))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], monthly_stats['Mã độc phát hiện'], width, 
                label='Mã độc phát hiện', color='#ff6b6b', alpha=0.8)
            ax.bar([i + width/2 for i in x], monthly_stats['File lành tính'], width,
                label='File lành tính', color='#51cf66', alpha=0.8)
            
            ax.set_xlabel('Tháng')
            ax.set_ylabel('Số lượng file')
            ax.set_title('Thống kê phát hiện mã độc theo tháng')
            ax.set_xticks(x)
            ax.set_xticklabels([str(period) for period in monthly_stats.index])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with chart_tabs[1]:
            # Lấy dữ liệu và sắp xếp theo số lượng giảm dần
            malware_types_full = [
                'Adware', 'Backdoor', 'Dialer', 'Obfuscated mal', 'PWS', 'Rogue',
                'TDownloader', 'Trojan', 'TrojanDownl', 'Virus', 'Worm'
            ]
            malware_counts_full = [
                4961, 5669, 553, 1228, 679, 381, 564, 5852, 848, 1997, 8869
            ]

            # Sắp xếp để lấy top 5 loại có số lượng nhiều nhất
            malware_data = list(zip(malware_types_full, malware_counts_full))
            malware_data_sorted = sorted(malware_data, key=lambda x: x[1], reverse=True)
            top5_types, top5_counts = zip(*malware_data_sorted[:5])

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 8))
                colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96c93d', '#feca57']

                wedges, texts, autotexts = ax.pie(
                    top5_counts,
                    labels=top5_types,
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    explode=[0.05]*5
                )

                ax.set_title('Top 5 loại mã độc phát hiện nhiều nhất', fontsize=15, fontweight='bold')

                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')

                st.pyplot(fig)
                plt.close()

            
            with col2:
                st.markdown("**Top 5 nhóm mã độc có số phát hiện cao nhất:**")
                top_malware = [
                    {"name": "Worm", "detections": 8869, "risk": "Cực cao"},
                    {"name": "Trojan", "detections": 5852, "risk": "Cao"},
                    {"name": "Backdoor", "detections": 5669, "risk": "Cao"},
                    {"name": "Adware", "detections": 4961, "risk": "Trung bình"},
                    {"name": "Virus", "detections": 1997, "risk": "Trung bình"}
                ]
                for i, malware in enumerate(top_malware, 1):
                    risk_color = {
                        "Cực cao": "#d32f2f",
                        "Cao": "#f57c00",
                        "Trung bình": "#fbc02d"
                    }[malware["risk"]]
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid {risk_color};">
                        <strong>{i}. {malware['name']}</strong><br>
                        <small>Phát hiện: {malware['detections']:,} lần | Mức độ: <span style="color: {risk_color};">{malware['risk']}</span></small>
                    </div>
                    """, unsafe_allow_html=True)

        with chart_tabs[2]:
            st.markdown("##### Xu hướng phát hiện mã độc")

            fig, ax = plt.subplots(figsize=(12, 6))

            # Dữ liệu trung bình theo tuần
            weekly_stats = stats_df.groupby(stats_df['Ngày'].dt.to_period('W')).mean()

            ax.plot(range(len(weekly_stats)), weekly_stats['Mã độc phát hiện'],
                    marker='o', linewidth=2, markersize=4, color='#ff6b6b', label='Mã độc')
            ax.plot(range(len(weekly_stats)), weekly_stats['File lành tính'],
                    marker='s', linewidth=2, markersize=4, color='#51cf66', label='File lành tính')

            # Thêm đường xu hướng tổng thể (polyfit bậc 1)
            z1 = np.polyfit(range(len(weekly_stats)), weekly_stats['Mã độc phát hiện'], 1)
            p1 = np.poly1d(z1)
            ax.plot(range(len(weekly_stats)), p1(range(len(weekly_stats))),
                    "--", alpha=0.7, color='#183153', label='Trend mã độc')

            ax.set_xlabel('Tuần')
            ax.set_ylabel('Số lượng file trung bình')
            ax.set_title('Xu hướng phát hiện mã độc theo tuần')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Hiển thị thông tin xu hướng
            trend_slope = z1[0]
            if trend_slope > 0:
                trend_text = f"📈 Tổng thể: số lượng file mã độc/tuần đang có xu hướng **tăng** (+{trend_slope:.2f}/tuần)"
                trend_color = "#ff6b6b"
            else:
                trend_text = f"📉 Tổng thể: số lượng file mã độc/tuần đang có xu hướng **giảm** (-{abs(trend_slope):.2f}/tuần)"
                trend_color = "#51cf66"

            st.markdown(f"""
            <div style="background: {trend_color}; color: white; padding: 15px; border-radius: 5px; text-align: center; margin: 10px 0;">
                <h4>{trend_text}</h4>
            </div>
            """, unsafe_allow_html=True)


        with chart_tabs[3]:


            # Dữ liệu theo 8 vùng địa lý chính của Việt Nam (giả lập, điền lại số liệu thực tế nếu có)
            region_data = [
                {"region": "Đông Bắc", "threats": 1350, "population": "15M", "density": 90.0},
                {"region": "Tây Bắc", "threats": 620, "population": "5.7M", "density": 108.8},
                {"region": "Đồng bằng sông Hồng", "threats": 2680, "population": "22M", "density": 121.8},
                {"region": "Bắc Trung Bộ", "threats": 1220, "population": "10.5M", "density": 116.2},
                {"region": "Nam Trung Bộ", "threats": 980, "population": "9.1M", "density": 107.7},
                {"region": "Tây Nguyên", "threats": 870, "population": "6.2M", "density": 140.3},
                {"region": "Đông Nam Bộ", "threats": 3350, "population": "18.2M", "density": 184.1},
                {"region": "Đồng bằng sông Cửu Long", "threats": 2110, "population": "17.5M", "density": 120.6}
            ]
            region_df = pd.DataFrame(region_data)

            # Đặt font chữ mặc định cho matplotlib (nên dùng font "DejaVu Sans" hoặc "Arial", hoặc font tiếng Việt như "Roboto", "Tahoma" nếu có hỗ trợ)
            plt.rcParams['font.family'] = 'DejaVu Sans'  # hoặc 'Arial', 'Tahoma', 'Roboto', v.v.
            plt.rcParams['font.size'] = 15

            fig, ax = plt.subplots(figsize=(15, 7))

            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96c93d', '#feca57', '#ff9ff3', '#54a0ff', '#51cf66']
            bars = ax.barh(region_df['region'], region_df['threats'], color=colors, alpha=0.88, height=0.55)

            ax.set_xlabel('Số lượng mối đe dọa', fontsize=17, labelpad=15, fontweight='bold')
            ax.set_title('Mối đe dọa theo khu vực hành chính', fontsize=23, fontweight='bold', pad=25)
            ax.tick_params(axis='y', labelsize=16)
            ax.tick_params(axis='x', labelsize=15)
            ax.grid(True, alpha=0.25, axis='x', linestyle='--', linewidth=1.2)

            for bar in bars:
                width = bar.get_width()
                ax.text(width + max(region_df['threats']) * 0.015, bar.get_y() + bar.get_height()/2,
                        f'{int(width):,}', ha='left', va='center', fontsize=15, fontweight='bold', color='#222')

            plt.tight_layout(pad=2.2)
            st.pyplot(fig)
            plt.close()


            # with col2:
            # # Thống kê chi tiết rút gọn
            #     for region in region_data:
            #         st.markdown(f"""
            #         <div style="background: #f8f9fa; padding: 12px; margin: 8px 0; border-radius: 5px;">
            #             <h5 style="margin: 0 0 -18px 0; color: #333;">{region['region']}</h5>
            #             <div style="display: flex; justify-content: space-between;">
            #                 <span>Mối đe dọa:</span>
            #                 <strong>{region['threats']:,}</strong>
            #             </div>
            #             <div style="display: flex; justify-content: space-between;">
            #                 <span>Tỷ lệ nhiễm:</span>
            #                 <strong>{region['density']:.1f}%</strong>
            #             </div>
            #         </div>
            #         """, unsafe_allow_html=True)




        # Thêm phần cảnh báo và khuyến nghị
        st.markdown('<div class="sub-header">💡 Khuyến nghị Bảo mật</div>', unsafe_allow_html=True)
        recommendations = [
            {
                "icon": "🛡️",
                "title": "Cập nhật hệ thống thường xuyên",
                "desc": "Luôn cài đặt các bản vá bảo mật mới nhất cho hệ điều hành và phần mềm",
                "priority": "high"
            },
            {
                "icon": "🔍",
                "title": "Quét mã độc định kỳ",
                "desc": "Sử dụng công cụ này để quét các file đáng ngờ ít nhất 1 lần/tuần",
                "priority": "high"
            },
            {
                "icon": "📧",
                "title": "Cẩn thận với email lạ",
                "desc": "Không mở file đính kèm hoặc click link từ email không rõ nguồn gốc",
                "priority": "medium"
            },
            {
                "icon": "💾",
                "title": "Sao lưu dữ liệu quan trọng",
                "desc": "Thực hiện backup định kỳ và lưu trữ ở nơi an toàn, tách biệt",
                "priority": "medium"
            },
            {
                "icon": "🔐",
                "title": "Sử dụng mật khẩu mạnh",
                "desc": "Tạo mật khẩu phức tạp và bật xác thực 2 yếu tố khi có thể",
                "priority": "low"
            },
            {
                "icon": "🌐",
                "title": "Duyệt web an toàn",
                "desc": "Tránh truy cập các trang web đáng ngờ và tải phần mềm từ nguồn không tin cậy",
                "priority": "low"
            }
        ]

        # Hiển thị khuyến nghị theo mức độ ưu tiên
        priority_colors = {"high": "#ff6b6b", "medium": "#ffd43b", "low": "#51cf66"}
        priority_labels = {"high": "Ưu tiên cao", "medium": "Ưu tiên trung bình", "low": "Ưu tiên thấp"}

        for priority in ["high", "medium", "low"]:
            priority_recs = [r for r in recommendations if r["priority"] == priority]
            if priority_recs:
                st.markdown(f"""
                <div style="background: {priority_colors[priority]}; color: white; padding: 10px; border-radius: 5px 5px 0 0; margin-top: 20px;">
                    <h4 style="margin: 0;">{priority_labels[priority]}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for rec in priority_recs:
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 15px; margin: 0 0 2px 0; border-left: 4px solid {priority_colors[priority]};">
                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 24px; margin-right: 10px;">{rec['icon']}</span>
                            <h5 style="margin: 0; color: #333;">{rec['title']}</h5>
                        </div>
                        <p style="margin: 0; color: #666; line-height: 1.4;">{rec['desc']}</p>
                    </div>
                    """, unsafe_allow_html=True)
# --- Tab thông tin ---
with tab4:
    st.markdown('<div class="sub-header">ℹ️ Thông tin về ứng dụng</div>', unsafe_allow_html=True)
    
    # Thông tin về mô hình
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 Thông tin mô hình")
        if model is not None and class_names is not None:
            st.success("✅ Mô hình đã được tải thành công")
            st.write(f"**Số lượng lớp:** {len(class_names)}")
            st.write(f"**Kiến trúc:** Inception v3")
            st.write(f"**Thiết bị:** {device}")
            
            # Hiển thị danh sách lớp
            with st.expander("Xem danh sách lớp"):
                for i, class_name in enumerate(class_names):
                    st.write(f"{i+1}. {class_name}")
        else:
            st.warning("⚠️ Chưa tải mô hình")
    
    with col2:
        st.markdown("### ⚙️ Cấu hình hiện tại")
        st.write(f"**Kích thước ảnh model:** {IMAGE_SIZE}")
        st.write(f"**Kích thước ảnh hiển thị:** {IMAGE_SIZE2}")
        st.write(f"**Ngưỡng phát hiện:** {threshold:.2%}")
        st.write(f"**Kích thước file tối đa:** {max_file_size} MB")
        st.write(f"**Số kết quả top hiển thị:** {top_k_results}")
        
        # Hiển thị danh sách lớp lành tính
        with st.expander("Xem lớp lành tính"):
            for benign_class in benign_classes:
                st.write(f"• {benign_class}")

    # Hướng dẫn sử dụng
    st.markdown("### 📖 Hướng dẫn sử dụng")
    
    with st.expander("🔍 Cách phân tích file đơn lẻ"):
        st.markdown("""
        1. **Tải mô hình:** Chọn nguồn mô hình trong sidebar (đường dẫn cục bộ hoặc upload từ máy tính)
        2. **Chọn file:** Trong tab "Quét chương trình đơn lẻ", click "Browse files" và chọn file .exe hoặc .dll
        3. **Xem kết quả:** Hệ thống sẽ hiển thị:
           - Kết luận về file (mã độc/lành tính/không chắc chắn)
           - Top dự đoán với xác suất
           - Phân tích chi tiết (hash, entropy, PE header, imports, strings...)
        4. **Đánh giá:** Dựa vào xác suất và các thông tin chi tiết để đưa ra quyết định cuối cùng
        """)
    
    with st.expander("📁 Cách quét thư mục"):
        st.markdown("""
        1. **Tải mô hình:** Đảm bảo đã tải mô hình thành công
        2. **Chọn nguồn:** 
           - **File ZIP:** Upload file ZIP chứa các file cần quét
           - **Đường dẫn:** Nhập đường dẫn thư mục trên máy tính
        3. **Cấu hình:** Mở "Cấu hình quét nâng cao" để điều chỉnh:
           - Kích thước file tối thiểu/tối đa
           - Độ sâu phân tích (Nhanh/Cân bằng/Sâu)
        4. **Bắt đầu quét:** Click "Bắt đầu quét" và chờ kết quả
        5. **Xem kết quả:** Hệ thống hiển thị tổng quan và chi tiết theo từng loại
        """)
    
    with st.expander("⚙️ Cấu hình nâng cao"):
        st.markdown("""
        **Trong Sidebar:**
        - **Nguồn mô hình:** Chọn cách tải mô hình (đường dẫn hoặc upload)
        - **Kích thước ảnh:** Điều chỉnh kích thước ảnh cho mô hình và hiển thị
        - **Ngưỡng phát hiện:** Xác suất tối thiểu để xác định mã độc (0.0-1.0)
        - **Kích thước file tối đa:** Giới hạn kích thước file được phân tích
        - **Lớp lành tính:** Danh sách tên lớp được coi là lành tính
        
        **Độ sâu phân tích:**
        - **Nhanh:** Phân tích cơ bản, tốc độ cao
        - **Cân bằng:** Phân tích vừa phải (mặc định)
        - **Sâu:** Phân tích chi tiết, bao gồm entropy và thống kê nâng cao
        """)

    # Thông tin kỹ thuật
    st.markdown("### 🔬 Thông tin kỹ thuật")
    
    with st.expander("Xem chi tiết kỹ thuật"):
        st.markdown("""
        **Kiến trúc mô hình:** Inception v3
        - Mạng neural tích chập sâu được thiết kế bởi Google
        - Tối ưu hóa cho phân loại hình ảnh với độ chính xác cao
        - Sử dụng factorized convolutions và auxiliary classifiers
        
        **Quy trình xử lý:**
        1. Chuyển đổi file binary thành ảnh grayscale
        2. Chuyển đổi grayscale sang RGB (3 kênh)
        3. Resize ảnh về kích thước mô hình yêu cầu
        4. Chuẩn hóa pixel values theo ImageNet standards
        5. Dự đoán qua mô hình và tính softmax probabilities
        
        **Phân tích tĩnh:**
        - **PE Header:** Thông tin về file PE (Portable Executable)
        - **Sections:** Phân tích các section và entropy của chúng
        - **Imports:** Danh sách các hàm được import từ DLL
        - **Strings:** Trích xuất và phân tích chuỗi ASCII
        - **Entropy:** Đo độ ngẫu nhiên của dữ liệu (0-8)
        
        **Metrics:**
        - **Entropy < 6.0:** File thông thường
        - **Entropy 6.0-7.0:** Có thể nén/mã hóa một phần  
        - **Entropy > 7.0:** Có thể được nén/mã hóa/đóng gói
        """)

     # Hiển thị bản đồ Việt Nam với thông tin về mối đe dọa
        st.markdown('<div class="map-header">🗺️ Giám sát an ninh mạng quốc gia: Bản đồ Việt Nam</div>', unsafe_allow_html=True)

        # Tạo HTML cho bản đồ
        map_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vietnam Cybersecurity Threat Map</title>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
            <style>
                body { margin: 0; padding: 0; }
                #map { height: 500px; width: 100%; }
                .threat-info {
                    background: rgba(255, 255, 255, 0.9);
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                }
                .threat-level-high { color: #d32f2f; font-weight: bold; }
                .threat-level-medium { color: #f57c00; font-weight: bold; }
                .threat-level-low { color: #388e3c; font-weight: bold; }
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
            <script>
                // Khởi tạo bản đồ tập trung vào Việt Nam
                var map = L.map('map').setView([16.0583, 108.2772], 6);
                // Thêm tile layer
                L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {attribution: '© OpenStreetMap, © CartoDB'}).addTo(map);
                // Dữ liệu mối đe dọa giả lập 63 tỉnh/thành Việt Nam 2024
                var threatData = [
                    // Nhóm High (từ 700+)
                    {city: "TP. Hồ Chí Minh", lat: 10.7769, lng: 106.7009, threats: 2100, level: "high", malware: 750, phishing: 520, ddos: 315, other: 515},
                    {city: "Hà Nội", lat: 21.0285, lng: 105.8542, threats: 1480, level: "high", malware: 525, phishing: 365, ddos: 212, other: 378},
                    {city: "Bình Dương", lat: 10.9804, lng: 106.6519, threats: 855, level: "high", malware: 312, phishing: 201, ddos: 121, other: 221},
                    {city: "Đồng Nai", lat: 10.9452, lng: 106.8246, threats: 802, level: "high", malware: 285, phishing: 189, ddos: 116, other: 212},

                    // Nhóm Medium (300–700)
                    {city: "Hải Phòng", lat: 20.8449, lng: 106.6881, threats: 626, level: "medium", malware: 213, phishing: 168, ddos: 104, other: 141},
                    {city: "Cần Thơ", lat: 10.0452, lng: 105.7469, threats: 563, level: "medium", malware: 186, phishing: 153, ddos: 80, other: 144},
                    {city: "Đà Nẵng", lat: 16.0544, lng: 108.2022, threats: 488, level: "medium", malware: 159, phishing: 126, ddos: 74, other: 129},
                    {city: "Thanh Hóa", lat: 19.8072, lng: 105.7768, threats: 473, level: "medium", malware: 154, phishing: 121, ddos: 77, other: 121},
                    {city: "Nghệ An", lat: 18.6796, lng: 105.6813, threats: 452, level: "medium", malware: 147, phishing: 115, ddos: 70, other: 120},
                    {city: "Quảng Ninh", lat: 21.0064, lng: 107.2925, threats: 441, level: "medium", malware: 141, phishing: 112, ddos: 66, other: 122},
                    {city: "Thừa Thiên Huế", lat: 16.4637, lng: 107.5909, threats: 407, level: "medium", malware: 135, phishing: 98, ddos: 62, other: 112},
                    {city: "Hải Dương", lat: 20.9401, lng: 106.3336, threats: 396, level: "medium", malware: 123, phishing: 101, ddos: 62, other: 110},
                    {city: "Bắc Ninh", lat: 21.1861, lng: 106.0763, threats: 384, level: "medium", malware: 118, phishing: 95, ddos: 66, other: 105},
                    {city: "Thái Bình", lat: 20.4509, lng: 106.3406, threats: 376, level: "medium", malware: 112, phishing: 92, ddos: 65, other: 107},
                    {city: "Vĩnh Phúc", lat: 21.3081, lng: 105.6046, threats: 372, level: "medium", malware: 109, phishing: 90, ddos: 62, other: 111},
                    {city: "Bắc Giang", lat: 21.2731, lng: 106.1946, threats: 361, level: "medium", malware: 106, phishing: 84, ddos: 66, other: 105},
                    {city: "Nam Định", lat: 20.4388, lng: 106.1621, threats: 358, level: "medium", malware: 103, phishing: 88, ddos: 61, other: 106},
                    {city: "Phú Thọ", lat: 21.3457, lng: 105.2120, threats: 347, level: "medium", malware: 100, phishing: 85, ddos: 55, other: 107},
                    {city: "Quảng Nam", lat: 15.5394, lng: 108.0191, threats: 341, level: "medium", malware: 98, phishing: 83, ddos: 54, other: 106},

                    // Nhóm Low (dưới 300)
                    {city: "Bình Định", lat: 13.7820, lng: 109.2191, threats: 298, level: "low", malware: 85, phishing: 76, ddos: 46, other: 91},
                    {city: "Quảng Ngãi", lat: 15.1202, lng: 108.7922, threats: 286, level: "low", malware: 84, phishing: 63, ddos: 43, other: 96},
                    {city: "Lâm Đồng", lat: 11.5753, lng: 108.1429, threats: 275, level: "low", malware: 78, phishing: 59, ddos: 40, other: 98},
                    {city: "Kiên Giang", lat: 10.0086, lng: 105.0807, threats: 265, level: "low", malware: 75, phishing: 58, ddos: 36, other: 96},
                    {city: "Long An", lat: 10.5435, lng: 106.4106, threats: 257, level: "low", malware: 72, phishing: 55, ddos: 38, other: 92},
                    {city: "Bến Tre", lat: 10.2415, lng: 106.3754, threats: 243, level: "low", malware: 69, phishing: 52, ddos: 33, other: 89},
                    {city: "An Giang", lat: 10.5216, lng: 105.1259, threats: 239, level: "low", malware: 67, phishing: 54, ddos: 31, other: 87},
                    {city: "Đắk Lắk", lat: 12.7100, lng: 108.2378, threats: 233, level: "low", malware: 65, phishing: 47, ddos: 32, other: 89},
                    {city: "Tiền Giang", lat: 10.4493, lng: 106.3421, threats: 231, level: "low", malware: 66, phishing: 46, ddos: 31, other: 88},
                    {city: "Bà Rịa - Vũng Tàu", lat: 10.5418, lng: 107.2428, threats: 228, level: "low", malware: 64, phishing: 49, ddos: 28, other: 87},
                    {city: "Quảng Bình", lat: 17.4689, lng: 106.6228, threats: 224, level: "low", malware: 62, phishing: 44, ddos: 31, other: 87},
                    {city: "Tây Ninh", lat: 11.3352, lng: 106.1099, threats: 220, level: "low", malware: 61, phishing: 43, ddos: 29, other: 87},
                    {city: "Thái Nguyên", lat: 21.5672, lng: 105.8252, threats: 217, level: "low", malware: 60, phishing: 42, ddos: 28, other: 87},
                    {city: "Vĩnh Long", lat: 10.2536, lng: 105.9722, threats: 211, level: "low", malware: 59, phishing: 41, ddos: 25, other: 86},
                    {city: "Quảng Trị", lat: 16.8187, lng: 107.0917, threats: 207, level: "low", malware: 57, phishing: 40, ddos: 26, other: 84},
                    {city: "Sóc Trăng", lat: 9.6026, lng: 105.9731, threats: 203, level: "low", malware: 56, phishing: 39, ddos: 25, other: 83},
                    {city: "Gia Lai", lat: 13.8079, lng: 108.1095, threats: 200, level: "low", malware: 54, phishing: 37, ddos: 24, other: 85},
                    {city: "Bạc Liêu", lat: 9.2941, lng: 105.7278, threats: 199, level: "low", malware: 54, phishing: 38, ddos: 23, other: 84},
                    {city: "Hà Tĩnh", lat: 18.3559, lng: 105.8875, threats: 197, level: "low", malware: 52, phishing: 38, ddos: 24, other: 83},
                    {city: "Ninh Bình", lat: 20.2506, lng: 105.9745, threats: 194, level: "low", malware: 52, phishing: 36, ddos: 22, other: 84},
                    {city: "Hưng Yên", lat: 20.6463, lng: 106.0511, threats: 192, level: "low", malware: 51, phishing: 35, ddos: 22, other: 84},
                    {city: "Đắk Nông", lat: 12.2644, lng: 107.6098, threats: 188, level: "low", malware: 49, phishing: 34, ddos: 21, other: 84},
                    {city: "Tuyên Quang", lat: 21.8230, lng: 105.2148, threats: 185, level: "low", malware: 49, phishing: 33, ddos: 21, other: 82},
                    {city: "Phú Yên", lat: 13.0882, lng: 109.0929, threats: 181, level: "low", malware: 48, phishing: 31, ddos: 22, other: 80},
                    {city: "Bình Phước", lat: 11.7512, lng: 106.7235, threats: 178, level: "low", malware: 47, phishing: 31, ddos: 20, other: 80},
                    {city: "Vĩnh Long", lat: 10.2536, lng: 105.9722, threats: 175, level: "low", malware: 46, phishing: 30, ddos: 20, other: 79},
                    {city: "Hà Nam", lat: 20.5833, lng: 105.9160, threats: 174, level: "low", malware: 45, phishing: 31, ddos: 19, other: 79},
                    {city: "Yên Bái", lat: 21.7051, lng: 104.8800, threats: 173, level: "low", malware: 44, phishing: 31, ddos: 18, other: 80},
                    {city: "Cà Mau", lat: 9.1768, lng: 105.1500, threats: 172, level: "low", malware: 43, phishing: 30, ddos: 18, other: 81},
                    {city: "Lào Cai", lat: 22.4804, lng: 103.9756, threats: 170, level: "low", malware: 42, phishing: 30, ddos: 18, other: 80},
                    {city: "Kon Tum", lat: 14.3549, lng: 108.0076, threats: 168, level: "low", malware: 41, phishing: 29, ddos: 17, other: 81},
                    {city: "Hòa Bình", lat: 20.8171, lng: 105.3376, threats: 167, level: "low", malware: 41, phishing: 28, ddos: 17, other: 81},
                    {city: "Trà Vinh", lat: 9.9347, lng: 106.3452, threats: 163, level: "low", malware: 41, phishing: 27, ddos: 16, other: 79},
                    {city: "Lạng Sơn", lat: 21.8528, lng: 106.7610, threats: 159, level: "low", malware: 40, phishing: 27, ddos: 16, other: 76},
                    {city: "Quảng Nam", lat: 15.5394, lng: 108.0191, threats: 158, level: "low", malware: 39, phishing: 27, ddos: 15, other: 77},
                    {city: "Bắc Kạn", lat: 22.1485, lng: 105.8348, threats: 156, level: "low", malware: 39, phishing: 25, ddos: 15, other: 77},
                    {city: "Cao Bằng", lat: 22.6666, lng: 106.2579, threats: 154, level: "low", malware: 38, phishing: 25, ddos: 14, other: 77},
                    {city: "Bình Thuận", lat: 11.0904, lng: 108.0721, threats: 153, level: "low", malware: 37, phishing: 26, ddos: 13, other: 77},
                    {city: "Điện Biên", lat: 21.3860, lng: 103.0230, threats: 151, level: "low", malware: 36, phishing: 24, ddos: 14, other: 77},
                    {city: "Ninh Thuận", lat: 11.6739, lng: 109.0147, threats: 149, level: "low", malware: 35, phishing: 24, ddos: 13, other: 77},
                    {city: "Hà Giang", lat: 22.8233, lng: 104.9836, threats: 147, level: "low", malware: 34, phishing: 23, ddos: 13, other: 77},
                    {city: "Quảng Ngãi", lat: 15.1202, lng: 108.7922, threats: 146, level: "low", malware: 33, phishing: 23, ddos: 12, other: 78},
                    {city: "Sơn La", lat: 21.3256, lng: 103.9188, threats: 144, level: "low", malware: 33, phishing: 23, ddos: 12, other: 76},
                    {city: "Bắc Ninh", lat: 21.1861, lng: 106.0763, threats: 143, level: "low", malware: 32, phishing: 22, ddos: 12, other: 77},
                    {city: "Phú Thọ", lat: 21.3457, lng: 105.2120, threats: 139, level: "low", malware: 31, phishing: 21, ddos: 11, other: 76},
                    {city: "Khánh Hòa", lat: 12.2388, lng: 109.1967, threats: 126, level: "low", malware: 27, phishing: 18, ddos: 10, other: 71},
                    {city: "Hậu Giang", lat: 9.7845, lng: 105.4701, threats: 123, level: "low", malware: 26, phishing: 17, ddos: 9, other: 71},
                    // ... (bạn có thể thêm tiếp các huyện/thị xã nếu cần)
                ];

                // Hàm xác định màu sắc dựa trên mức độ đe dọa
                function getThreatColor(level) {
                    switch(level) {
                        case 'high': return '#d32f2f';
                        case 'medium': return '#f57c00';
                        case 'low': return '#388e3c';
                        default: return '#666666';
                    }
                }
                
                // Hàm xác định kích thước marker dựa trên số lượng đe dọa
                function getMarkerSize(threats) {
                    if (threats > 1000) return 25;
                    if (threats > 500) return 15;
                    if (threats > 200) return 10;
                    return 10;
                }
                
                // Thêm markers cho từng thành phố
                threatData.forEach(function(data) {
                    var color = getThreatColor(data.level);
                    var size = getMarkerSize(data.threats);
                    
                    // Tạo custom icon
                    var threatIcon = L.divIcon({
                        className: 'threat-marker',
                        html: '<div style="background-color: ' + color + '; width: ' + size + 'px; height: ' + size + 'px; border-radius: 50%; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>',
                        iconSize: [size, size],
                        iconAnchor: [size/2, size/2]
                    });
                    
                    // Tạo popup content
                    var popupContent = `
                        <div class="threat-info">
                            <h3>${data.city}</h3>
                            <p><strong>Tổng mối đe dọa:</strong> <span class="threat-level-${data.level}">${data.threats}</span></p>
                            <hr>
                            <p><strong>Phân loại:</strong></p>
                            <ul>
                                <li>🦠 Malware: ${data.malware}</li>
                                <li>🎣 Phishing: ${data.phishing}</li>
                                <li>⚡ DDoS: ${data.ddos}</li>
                                <li>🔧 Khác: ${data.other}</li>
                            </ul>
                            <p><strong>Mức độ:</strong> <span class="threat-level-${data.level}">${data.level.toUpperCase()}</span></p>
                        </div>
                    `;
                    
                    // Thêm marker vào bản đồ
                    L.marker([data.lat, data.lng], {icon: threatIcon})
                        .bindPopup(popupContent)
                        .addTo(map);
                });
                
                // Thêm legend
                var legend = L.control({position: 'bottomright'});
                legend.onAdd = function (map) {
                    var div = L.DomUtil.create('div', 'info legend');
                    div.innerHTML = `
                        <div style="background: rgba(255,255,255,0.9); padding: 10px; border-radius: 5px; box-shadow: 0 1px 10px rgba(0,0,0,0.2);">
                            <h4>Mức độ đe dọa</h4>
                            <div><span style="color: #d32f2f;">●</span> Cao (>1000)</div>
                            <div><span style="color: #f57c00;">●</span> Trung bình (200-1000)</div>
                            <div><span style="color: #388e3c;">●</span> Thấp (<200)</div>
                            <hr>
                            <small>Dữ liệu cập nhật: ${new Date().toLocaleDateString('vi-VN')}</small>
                        </div>
                    `;
                    return div;
                };
                legend.addTo(map);
                
                // Thêm thông tin tổng quan
                var info = L.control({position: 'topleft'});
                info.onAdd = function (map) {
                    var div = L.DomUtil.create('div', 'info');
                    var totalThreats = threatData.reduce((sum, data) => sum + data.threats, 0);
                    div.innerHTML = `
                        <div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);">
                            <h3>🇻🇳 Tình hình An ninh mạng Việt Nam</h3>
                            <p><strong>Tổng mối đe dọa:</strong> <span style="color: #d32f2f; font-weight: bold;">${totalThreats}</span></p>
                            <p><strong>Khu vực nguy hiểm nhất:</strong> TP. Hồ Chí Minh</p>
                            <p><strong>Loại đe dọa phổ biến:</strong> Malware</p>
                            <hr>
                            <small>⚠️Dữ liệu sử dụng cho mục đích minh họa</small>
                        </div>
                    `;
                    return div;
                };
                info.addTo(map);

                
                // Thêm hiệu ứng pulse cho các marker có mức đe dọa cao
                threatData.forEach(function(data) {
                    if (data.level === 'high') {
                        var pulseIcon = L.divIcon({
                            className: 'pulse-marker',
                            html: '<div class="pulse-dot"></div>',
                            iconSize: [20, 20],
                            iconAnchor: [10, 10]
                        });
                        
                        L.marker([data.lat, data.lng], {icon: pulseIcon}).addTo(map);
                    }
                });
                
                // CSS cho hiệu ứng pulse
                var style = document.createElement('style');
                style.innerHTML = `
                    .pulse-dot {
                        width: 20px;
                        height: 20px;
                        background-color: #d32f2f;
                        border-radius: 50%;
                        animation: pulse 2s infinite;
                        opacity: 0.8;
                    }
                    
                    @keyframes pulse {
                        0% {
                            transform: scale(0.8);
                            opacity: 1;
                        }
                        50% {
                            transform: scale(1.2);
                            opacity: 0.5;
                        }
                        100% {
                            transform: scale(0.8);
                            opacity: 1;
                        }
                    }
                `;
                document.head.appendChild(style);
                
            </script>
        </body>
        </html>
        """

        # Hiển thị bản đồ
        st.components.v1.html(map_html, height=500)

        # Thêm thông tin cảnh báo bảo mật
        st.markdown('<div class="sub-header">🚨 Cảnh báo Bảo mật</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ff6b6b, #ee5a52); color: white; padding: 20px; border-radius: 10px; text-align: center;">
                <h4>⚠️ Mức độ đe dọa</h4>
                <h2>CAO</h2>
                <p>Phát hiện nhiều mã độc mới</p>
                <small>Cập nhật: Hôm nay</small>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4ecdc4, #44a08d); color: white; padding: 20px; border-radius: 10px; text-align: center;">
                <h4>🛡️ Tỷ lệ phát hiện</h4>
                <h2>94.7%</h2>
                <p>Độ chính xác của hệ thống</p>
                <small>Dựa trên 10,000+ mẫu</small>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #45b7d1, #96c93d); color: white; padding: 20px; border-radius: 10px; text-align: center;">
                <h4>📊 Files đã quét</h4>
                <h2>2,847</h2>
                <p>Trong tháng này</p>
                <small>Tăng 23% so với tháng trước</small>
            </div>
            """, unsafe_allow_html=True)

        # Thêm RSS feed giả lập về tin tức bảo mật
        st.markdown('<div class="sub-header">📰 Tin tức Bảo mật mới nhất</div>', unsafe_allow_html=True)

        news_data = [
            {
                "title": "Phát hiện biến thể mới của ransomware LockBit tại Việt Nam",
                "summary": "Các chuyên gia bảo mật cảnh báo về sự xuất hiện của biến thể mới của ransomware LockBit đang nhắm mục tiêu vào các doanh nghiệp Việt Nam.",
                "time": "2 giờ trước",
                "severity": "high"
            },
            {
                "title": "Cập nhật bản vá bảo mật khẩn cấp cho Windows",
                "summary": "Microsoft phát hành bản vá khẩn cấp để sửa lỗi zero-day đang được khai thác tích cực bởi các nhóm APT.",
                "time": "5 giờ trước", 
                "severity": "high"
            },
            {
                "title": "Chiến dịch phishing mạo danh ngân hàng gia tăng",
                "summary": "Số lượng email phishing mạo danh các ngân hàng lớn tại Việt Nam tăng 45% trong tuần qua.",
                "time": "1 ngày trước",
                "severity": "medium"
            },
            {
                "title": "Hướng dẫn bảo vệ hệ thống khỏi malware mới",
                "summary": "Các biện pháp phòng ngừa và phát hiện sớm các loại malware mới xuất hiện gần đây.",
                "time": "2 ngày trước",
                "severity": "low"
            }
        ]

        for news in news_data:
            severity_color = {"high": "#ff6b6b", "medium": "#ffd43b", "low": "#51cf66"}[news["severity"]]
            severity_text = {"high": "🔴 Nghiêm trọng", "medium": "🟡 Trung bình", "low": "🟢 Thông tin"}[news["severity"]]
            
            st.markdown(f"""
            <div style="border-left: 4px solid {severity_color}; background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 0 5px 5px 0;">
                <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 8px;">
                    <h4 style="margin: 0; color: #333;">{news['title']}</h4>
                    <span style="color: {severity_color}; font-size: 0.9em; font-weight: bold;">{severity_text}</span>
                </div>
                <p style="margin: 8px 0; color: #666; line-height: 1.5;">{news['summary']}</p>
                <small style="color: #999;">⏰ {news['time']}</small>
            </div>
            """, unsafe_allow_html=True)

        # Thêm biểu đồ thống kê thời gian thực
        st.markdown('<div class="sub-header">📈 Thống kê Thời gian thực</div>', unsafe_allow_html=True)

        # Tạo dãy ngày
        dates = pd.date_range(start='2024-07-01', end='2025-06-30', freq='D')

        # Thiết lập seed để tái lập kết quả
        np.random.seed(42)

        # Tạo hệ số dao động cho từng tháng (giả lập xu hướng thực tế: cuối năm tăng, đầu năm giảm)
        monthly_malware_base = {
            7:  14,  8: 13,  9: 16, 10: 19, 11: 22, 12: 28,  # Từ tháng 7 đến tháng 12/2024
            1:  32,  2: 29,  3: 22,  4: 18,  5: 16,  6: 15   # Từ tháng 1 đến tháng 6/2025
        }
        monthly_clean_base = {
            7:  55,  8: 56,  9: 58, 10: 62, 11: 66, 12: 70,
            1:  75,  2: 70,  3: 65,  4: 60,  5: 58,  6: 56
        }

        dates = pd.date_range(start='2024-07-01', end='2025-06-30', freq='D')
        np.random.seed(42)

        malware_detections = []
        clean_files = []
        base_mal = 10
        base_clean = 50
        trend_increase = 0.08  # mức tăng nhẹ theo ngày

        for i, d in enumerate(dates):
            # Dao động mạnh quanh giá trị trung bình nhưng vẫn tăng dần theo thời gian
            daily_mal = base_mal + (i * trend_increase) + np.random.normal(0, 5)
            daily_clean = base_clean + (i * trend_increase * 2) + np.random.normal(0, 12)
            malware_detections.append(max(0, int(daily_mal)))
            clean_files.append(max(0, int(daily_clean)))

        stats_df = pd.DataFrame({
            'Ngày': dates,
            'Mã độc phát hiện': malware_detections,
            'File lành tính': clean_files,
            'Tổng file quét': np.array(malware_detections) + np.array(clean_files)
        })

        # Optional: Xem thử thống kê theo tháng
        #print(stats_df.groupby(stats_df['Ngày'].dt.month)[['Mã độc phát hiện', 'File lành tính']].mean())

        # Tạo tabs cho các biểu đồ khác nhau
        chart_tabs = st.tabs(["📊 Tổng quan", "🦠 Mã độc", "📈 Xu hướng", "🌍 Phân bố địa lý"])

        with chart_tabs[0]:
            
            # Biểu đồ cột
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Lấy dữ liệu theo tháng - SỬA LỖI Ở ĐÂY
            monthly_stats = stats_df.groupby(stats_df['Ngày'].dt.to_period('M')).agg({
                'Mã độc phát hiện': 'sum',
                'File lành tính': 'sum',
                'Tổng file quét': 'sum'
            })
            
            x = range(len(monthly_stats))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], monthly_stats['Mã độc phát hiện'], width, 
                label='Mã độc phát hiện', color='#ff6b6b', alpha=0.8)
            ax.bar([i + width/2 for i in x], monthly_stats['File lành tính'], width,
                label='File lành tính', color='#51cf66', alpha=0.8)
            
            ax.set_xlabel('Tháng')
            ax.set_ylabel('Số lượng file')
            ax.set_title('Thống kê phát hiện mã độc theo tháng')
            ax.set_xticks(x)
            ax.set_xticklabels([str(period) for period in monthly_stats.index])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with chart_tabs[1]:
            # Lấy dữ liệu và sắp xếp theo số lượng giảm dần
            malware_types_full = [
                'Adware', 'Backdoor', 'Dialer', 'Obfuscated mal', 'PWS', 'Rogue',
                'TDownloader', 'Trojan', 'TrojanDownl', 'Virus', 'Worm'
            ]
            malware_counts_full = [
                4961, 5669, 553, 1228, 679, 381, 564, 5852, 848, 1997, 8869
            ]

            # Sắp xếp để lấy top 5 loại có số lượng nhiều nhất
            malware_data = list(zip(malware_types_full, malware_counts_full))
            malware_data_sorted = sorted(malware_data, key=lambda x: x[1], reverse=True)
            top5_types, top5_counts = zip(*malware_data_sorted[:5])

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 8))
                colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96c93d', '#feca57']

                wedges, texts, autotexts = ax.pie(
                    top5_counts,
                    labels=top5_types,
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    explode=[0.05]*5
                )

                ax.set_title('Top 5 loại mã độc phát hiện nhiều nhất', fontsize=15, fontweight='bold')

                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')

                st.pyplot(fig)
                plt.close()

            
            with col2:
                st.markdown("**Top 5 nhóm mã độc có số phát hiện cao nhất:**")
                top_malware = [
                    {"name": "Worm", "detections": 8869, "risk": "Cực cao"},
                    {"name": "Trojan", "detections": 5852, "risk": "Cao"},
                    {"name": "Backdoor", "detections": 5669, "risk": "Cao"},
                    {"name": "Adware", "detections": 4961, "risk": "Trung bình"},
                    {"name": "Virus", "detections": 1997, "risk": "Trung bình"}
                ]
                for i, malware in enumerate(top_malware, 1):
                    risk_color = {
                        "Cực cao": "#d32f2f",
                        "Cao": "#f57c00",
                        "Trung bình": "#fbc02d"
                    }[malware["risk"]]
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid {risk_color};">
                        <strong>{i}. {malware['name']}</strong><br>
                        <small>Phát hiện: {malware['detections']:,} lần | Mức độ: <span style="color: {risk_color};">{malware['risk']}</span></small>
                    </div>
                    """, unsafe_allow_html=True)

        with chart_tabs[2]:
            st.markdown("##### Xu hướng phát hiện mã độc")

            fig, ax = plt.subplots(figsize=(12, 6))

            # Dữ liệu trung bình theo tuần
            weekly_stats = stats_df.groupby(stats_df['Ngày'].dt.to_period('W')).mean()

            ax.plot(range(len(weekly_stats)), weekly_stats['Mã độc phát hiện'],
                    marker='o', linewidth=2, markersize=4, color='#ff6b6b', label='Mã độc')
            ax.plot(range(len(weekly_stats)), weekly_stats['File lành tính'],
                    marker='s', linewidth=2, markersize=4, color='#51cf66', label='File lành tính')

            # Thêm đường xu hướng tổng thể (polyfit bậc 1)
            z1 = np.polyfit(range(len(weekly_stats)), weekly_stats['Mã độc phát hiện'], 1)
            p1 = np.poly1d(z1)
            ax.plot(range(len(weekly_stats)), p1(range(len(weekly_stats))),
                    "--", alpha=0.7, color='#183153', label='Trend mã độc')

            ax.set_xlabel('Tuần')
            ax.set_ylabel('Số lượng file trung bình')
            ax.set_title('Xu hướng phát hiện mã độc theo tuần')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Hiển thị thông tin xu hướng
            trend_slope = z1[0]
            if trend_slope > 0:
                trend_text = f"📈 Tổng thể: số lượng file mã độc/tuần đang có xu hướng **tăng** (+{trend_slope:.2f}/tuần)"
                trend_color = "#ff6b6b"
            else:
                trend_text = f"📉 Tổng thể: số lượng file mã độc/tuần đang có xu hướng **giảm** (-{abs(trend_slope):.2f}/tuần)"
                trend_color = "#51cf66"

            st.markdown(f"""
            <div style="background: {trend_color}; color: white; padding: 15px; border-radius: 5px; text-align: center; margin: 10px 0;">
                <h4>{trend_text}</h4>
            </div>
            """, unsafe_allow_html=True)


        with chart_tabs[3]:


            # Dữ liệu theo 8 vùng địa lý chính của Việt Nam (giả lập, điền lại số liệu thực tế nếu có)
            region_data = [
                {"region": "Đông Bắc", "threats": 1350, "population": "15M", "density": 90.0},
                {"region": "Tây Bắc", "threats": 620, "population": "5.7M", "density": 108.8},
                {"region": "Đồng bằng sông Hồng", "threats": 2680, "population": "22M", "density": 121.8},
                {"region": "Bắc Trung Bộ", "threats": 1220, "population": "10.5M", "density": 116.2},
                {"region": "Nam Trung Bộ", "threats": 980, "population": "9.1M", "density": 107.7},
                {"region": "Tây Nguyên", "threats": 870, "population": "6.2M", "density": 140.3},
                {"region": "Đông Nam Bộ", "threats": 3350, "population": "18.2M", "density": 184.1},
                {"region": "Đồng bằng sông Cửu Long", "threats": 2110, "population": "17.5M", "density": 120.6}
            ]
            region_df = pd.DataFrame(region_data)

            # Đặt font chữ mặc định cho matplotlib (nên dùng font "DejaVu Sans" hoặc "Arial", hoặc font tiếng Việt như "Roboto", "Tahoma" nếu có hỗ trợ)
            plt.rcParams['font.family'] = 'DejaVu Sans'  # hoặc 'Arial', 'Tahoma', 'Roboto', v.v.
            plt.rcParams['font.size'] = 15

            fig, ax = plt.subplots(figsize=(15, 7))

            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96c93d', '#feca57', '#ff9ff3', '#54a0ff', '#51cf66']
            bars = ax.barh(region_df['region'], region_df['threats'], color=colors, alpha=0.88, height=0.55)

            ax.set_xlabel('Số lượng mối đe dọa', fontsize=17, labelpad=15, fontweight='bold')
            ax.set_title('Mối đe dọa theo khu vực hành chính', fontsize=23, fontweight='bold', pad=25)
            ax.tick_params(axis='y', labelsize=16)
            ax.tick_params(axis='x', labelsize=15)
            ax.grid(True, alpha=0.25, axis='x', linestyle='--', linewidth=1.2)

            for bar in bars:
                width = bar.get_width()
                ax.text(width + max(region_df['threats']) * 0.015, bar.get_y() + bar.get_height()/2,
                        f'{int(width):,}', ha='left', va='center', fontsize=15, fontweight='bold', color='#222')

            plt.tight_layout(pad=2.2)
            st.pyplot(fig)
            plt.close()


            # with col2:
            # # Thống kê chi tiết rút gọn
            #     for region in region_data:
            #         st.markdown(f"""
            #         <div style="background: #f8f9fa; padding: 12px; margin: 8px 0; border-radius: 5px;">
            #             <h5 style="margin: 0 0 -18px 0; color: #333;">{region['region']}</h5>
            #             <div style="display: flex; justify-content: space-between;">
            #                 <span>Mối đe dọa:</span>
            #                 <strong>{region['threats']:,}</strong>
            #             </div>
            #             <div style="display: flex; justify-content: space-between;">
            #                 <span>Tỷ lệ nhiễm:</span>
            #                 <strong>{region['density']:.1f}%</strong>
            #             </div>
            #         </div>
            #         """, unsafe_allow_html=True)




        # Thêm phần cảnh báo và khuyến nghị
        st.markdown('<div class="sub-header">💡 Khuyến nghị Bảo mật</div>', unsafe_allow_html=True)
        recommendations = [
            {
                "icon": "🛡️",
                "title": "Cập nhật hệ thống thường xuyên",
                "desc": "Luôn cài đặt các bản vá bảo mật mới nhất cho hệ điều hành và phần mềm",
                "priority": "high"
            },
            {
                "icon": "🔍",
                "title": "Quét mã độc định kỳ",
                "desc": "Sử dụng công cụ này để quét các file đáng ngờ ít nhất 1 lần/tuần",
                "priority": "high"
            },
            {
                "icon": "📧",
                "title": "Cẩn thận với email lạ",
                "desc": "Không mở file đính kèm hoặc click link từ email không rõ nguồn gốc",
                "priority": "medium"
            },
            {
                "icon": "💾",
                "title": "Sao lưu dữ liệu quan trọng",
                "desc": "Thực hiện backup định kỳ và lưu trữ ở nơi an toàn, tách biệt",
                "priority": "medium"
            },
            {
                "icon": "🔐",
                "title": "Sử dụng mật khẩu mạnh",
                "desc": "Tạo mật khẩu phức tạp và bật xác thực 2 yếu tố khi có thể",
                "priority": "low"
            },
            {
                "icon": "🌐",
                "title": "Duyệt web an toàn",
                "desc": "Tránh truy cập các trang web đáng ngờ và tải phần mềm từ nguồn không tin cậy",
                "priority": "low"
            }
        ]

        # Hiển thị khuyến nghị theo mức độ ưu tiên
        priority_colors = {"high": "#ff6b6b", "medium": "#ffd43b", "low": "#51cf66"}
        priority_labels = {"high": "Ưu tiên cao", "medium": "Ưu tiên trung bình", "low": "Ưu tiên thấp"}

        for priority in ["high", "medium", "low"]:
            priority_recs = [r for r in recommendations if r["priority"] == priority]
            if priority_recs:
                st.markdown(f"""
                <div style="background: {priority_colors[priority]}; color: white; padding: 10px; border-radius: 5px 5px 0 0; margin-top: 20px;">
                    <h4 style="margin: 0;">{priority_labels[priority]}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for rec in priority_recs:
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 15px; margin: 0 0 2px 0; border-left: 4px solid {priority_colors[priority]};">
                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 24px; margin-right: 10px;">{rec['icon']}</span>
                            <h5 style="margin: 0; color: #333;">{rec['title']}</h5>
                        </div>
                        <p style="margin: 0; color: #666; line-height: 1.4;">{rec['desc']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    # Footer cuối cùng
    st.markdown("---")
    st.markdown("""
    <div style="
        max-width: 520px;
        margin: 0 auto 0 auto;
        background: linear-gradient(90deg, #e0eafc 0%, #cfdef3 100%);
        color: #222;
        padding: 18px 22px 10px 22px;
        text-align: center;
        border-radius: 14px;
        box-shadow: 0 2px 12px rgba(90,120,180,0.08);">
        <div style="display: flex; align-items: center; justify-content: center; gap:13px;">
            <img src="https://img.icons8.com/color/48/000000/user-male-circle--v1.png" width="44">
            <div style="text-align:left;">
                <b>© 2024 - Huỳnh Hải Công Huy</b><br>
                <span style="font-size:15px;">Trường ĐH Kỹ Thuật - Hậu Cần CAND</span>
            </div>
        </div>
        <div style="margin:8px 0 2px 0; font-size:15px;">
            <img src='https://img.icons8.com/color/48/000000/home-page.png' width='17' style="vertical-align:middle;">
            Phường Hồ, Thuận Thành, Bắc Ninh
        </div>
        <div style="margin: 6px 0;">
            <img src="https://img.icons8.com/color/48/domain.png" width="16" style="vertical-align: middle;">
            <a href='https://dhkthtc.bacninh.gov.vn/TrangChu/' target='_blank' style='color:#2d69c7;font-weight:500;text-decoration:underline;'>dhkthtc.bacninh.gov.vn</a>
        </div>
        <div style="display:flex; justify-content:center; align-items:center; gap:13px; margin:9px 0 7px 0;">
            <div>
                <img src='https://img.icons8.com/color/48/000000/phone.png' width='20' style="vertical-align:middle;"> 
                0379095633
            </div>
            <a href="mailto:conghuy062000@gmail.com" target="_blank">
                <img src="https://img.icons8.com/color/48/000000/gmail.png" width="22">
            </a>
            <a href="https://www.facebook.com/block.huy" target="_blank">
                <img src="https://img.icons8.com/color/48/000000/facebook-new.png" width="22">
            </a>
            <a href="https://github.com/HICKER-WH" target="_blank">
                <img src="https://img.icons8.com/ios-filled/50/000000/github.png" width="20" style="background:white; border-radius:50%;">
            </a>
            <a href="https://www.linkedin.com/in/huy-hu%E1%BB%B3nh-h%E1%BA%A3i-c%C3%B4ng-2a640a177/" target="_blank">
                <img src="https://img.icons8.com/color/48/000000/linkedin.png" width="22">
            </a>
        </div>
<<<<<<< HEAD
        <div style="margin-top:3px;color:#444;font-size:13px;">
            <small>⚠️ Kết quả chỉ tham khảo. Luôn kết hợp nhiều công cụ để đảm bảo an toàn tối đa.</small>
        </div>
=======
>>>>>>> 58fc52c (Cập nhật code: sửa bug/thêm tính năng XYZ)
    </div>
    """, unsafe_allow_html=True)

# Hiển thị thông tin về mô hình
if model is not None and class_names is not None:
    st.markdown(f"""
    <div style="text-align: center; color: #666; margin-top: 20px;">
<<<<<<< HEAD
        <b> © 2024 - Huynh Hai Cong Huy</b>
    </div>
    """, unsafe_allow_html=True)
=======
        <b> © 2025 - Huynh Hai Cong Huy</b>
    </div>
    """, unsafe_allow_html=True)

>>>>>>> 58fc52c (Cập nhật code: sửa bug/thêm tính năng XYZ)
else:
    st.error("Mô hình chưa được tải thành công. Vui lòng kiểm tra đường dẫn đến file model.")
