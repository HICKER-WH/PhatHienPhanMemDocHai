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
        <img src="https://raw.githubusercontent.com/HickerWH/PhanTichMaDoc/main/Logot07.png" alt="Logo" style="height: 60px;">
    </a>
    <div class="title-container">
        <span class="title-line">🔍 Phát Hiện Phần Mềm Hại</span>
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
tab1, tab2, tab3 = st.tabs(["📄 Quét chương trình đơn lẻ", "📁 Quét nhanh chương trình trong thư mục", "ℹ️ Thông tin chung"])

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

        # Cấu hình quét nâng cao
        with st.expander("⚙️ Cấu hình quét nâng cao"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_size_kb = st.number_input("Kích thước file tối thiểu (KB)", value=0, min_value=0, max_value=1000)
                
            with col2:
                max_size_mb = st.number_input("Kích thước file tối đa (MB)", value=max_file_size, min_value=1, max_value=max_file_size)
                
            with col3:
                analysis_depth = st.selectbox("Độ sâu phân tích", ["Nhanh", "Cân bằng", "Sâu"])

        # Thực hiện quét nếu có yêu cầu
        if st.session_state.get('do_scan', False):
            scan_dir = st.session_state.get('scan_dir')
            
            if scan_dir and os.path.exists(scan_dir):
                st.markdown('<div class="sub-header">🔍 Đang quét thư mục...</div>', unsafe_allow_html=True)
                
                # Thực hiện quét
                scan_results, malware_count, uncertain_count, total_files = scan_directory(
                    scan_dir, model, class_names, device, benign_classes, threshold,
                    min_size_kb, max_size_mb, analysis_depth
                )
                
                # Xóa cờ quét để tránh quét lại
                st.session_state['do_scan'] = False
                
                if scan_results:
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

                    # Tạo biểu đồ tròn
                    if total_files > 0:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        labels = ['Mã độc', 'Lành tính', 'Không chắc chắn']
                        sizes = [malware_count, benign_count, uncertain_count]
                        colors = ['#ff6b6b', '#51cf66', '#ffd43b']
                        explode = (0.1, 0, 0)  # Làm nổi bật phần mã độc
                        
                        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                               shadow=True, startangle=90)
                        ax.set_title('Phân bố kết quả quét')
                        
                        st.pyplot(fig)
                        plt.close()

                    # Hiển thị kết quả chi tiết
                    st.markdown('<div class="sub-header">📋 Kết quả chi tiết</div>', unsafe_allow_html=True)
                    
                    # Tạo tabs cho các loại kết quả
                    if malware_count > 0:
                        result_tabs = st.tabs(["🔴 Mã độc", "✅ Lành tính", "⚠️ Không chắc chắn", "📊 Tất cả"])
                    else:
                        result_tabs = st.tabs(["✅ Lành tính", "⚠️ Không chắc chắn", "📊 Tất cả"])
                    
                    # Phân loại kết quả
                    malware_files = [r for r in scan_results if r["Là mã độc"] and r["Kết quả"] == 1]
                    benign_files = [r for r in scan_results if not r["Là mã độc"] and r["Kết quả"] == 0]
                    uncertain_files = [r for r in scan_results if r["Kết quả"] == 2]
                    
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
                                    file_name=f"malware_detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                        tab_index += 1
                    
                    # Tab lành tính
                    with result_tabs[tab_index]:
                        if benign_files:
                            st.success(f"✅ {len(benign_files)} file được xác định là lành tính")
                            
                            # Checkbox để hiển thị file lành tính
                            show_clean_files = st.checkbox("Hiển thị danh sách file lành tính", value=False)
                            
                            if show_clean_files:
                                benign_df = pd.DataFrame([{
                                    "Tên file": r["Tên file"],
                                    "Kích thước (KB)": r["Kích thước (KB)"],
                                    "Loại": r["Loại"],
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
                        st.info(f"Hiển thị tất cả {len(scan_results)} file đã quét")
                        
                        # Tạo DataFrame với tất cả kết quả
                        all_results_df = pd.DataFrame([{
                            "Tên file": r["Tên file"],
                            "Kích thước (KB)": r["Kích thước (KB)"],
                            "Kết quả": "Mã độc" if r["Là mã độc"] and r["Kết quả"] == 1 else 
                                     "Lành tính" if not r["Là mã độc"] and r["Kết quả"] == 0 else "Không chắc chắn",
                            "Loại": r["Loại"],
                            "Xác suất": f"{r['Xác suất']:.2%}",
                            "Đường dẫn": r["Đường dẫn"]
                        } for r in scan_results])
                        
                        st.dataframe(all_results_df, use_container_width=True)
                        
                        # Nút tải xuống tất cả kết quả
                        csv_all = all_results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Tải xuống tất cả kết quả (CSV)",
                            data=csv_all,
                            file_name=f"scan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

                    # Thống kê nâng cao
                    if analysis_depth == "Sâu":
                        st.markdown('<div class="sub-header">📈 Thống kê nâng cao</div>', unsafe_allow_html=True)
                        
                        # Phân tích entropy
                        entropy_values = [r["Entropy"] for r in scan_results if r["Entropy"] is not None]
                        if entropy_values:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("##### Phân bố Entropy")
                                fig, ax = plt.subplots(figsize=(8, 4))
                                ax.hist(entropy_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                                ax.set_xlabel('Entropy')
                                ax.set_ylabel('Số lượng file')
                                ax.set_title('Phân bố Entropy của các file')
                                st.pyplot(fig)
                                plt.close()
                            
                            with col2:
                                st.markdown("##### Thống kê Entropy")
                                st.write(f"**Entropy trung bình:** {np.mean(entropy_values):.4f}")
                                st.write(f"**Entropy cao nhất:** {np.max(entropy_values):.4f}")
                                st.write(f"**Entropy thấp nhất:** {np.min(entropy_values):.4f}")
                                st.write(f"**Độ lệch chuẩn:** {np.std(entropy_values):.4f}")
                                
                                # Cảnh báo về entropy cao
                                high_entropy_files = [r for r in scan_results if r["Entropy"] and r["Entropy"] > 7.0]
                                if high_entropy_files:
                                    st.warning(f"⚠️ {len(high_entropy_files)} file có entropy > 7.0 (có thể được đóng gói/mã hóa)")

                else:
                    st.warning("Không tìm thấy file .exe hoặc .dll nào trong thư mục được chỉ định.")
            
            # Dọn dẹp thư mục tạm nếu cần
            if option == "Tải lên file ZIP" and 'scan_dir' in st.session_state:
                try:
                    shutil.rmtree(st.session_state['scan_dir'])
                    del st.session_state['scan_dir']
                except:
                    pass

# --- Tab thông tin ---
with tab3:
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

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        <p>🔍 <strong>Malware Detection System</strong> | Phiên bản 2.0</p>
        <p>Sử dụng mô hình Inception v3 để phát hiện phần mềm độc hại</p>
        <p>© 2024 - Phát triển bởi AI Security Team</p>
    </div>
    """, unsafe_allow_html=True)

# Hiển thị bản đồ Việt Nam với thông tin về mối đe dọa
st.markdown('<div class="map-header">🗺️ Bản đồ Mối đe dọa An ninh mạng Việt Nam</div>', unsafe_allow_html=True)

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
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
        
        // Dữ liệu mối đe dọa giả lập cho các thành phố lớn
        var threatData = [
            {
                city: "Hà Nội",
                lat: 21.0285,
                lng: 105.8542,
                threats: 1247,
                level: "high",
                malware: 423,
                phishing: 301,
                ddos: 189,
                other: 334
            },
            {
                city: "TP. Hồ Chí Minh",
                lat: 10.8231,
                lng: 106.6297,
                threats: 1891,
                level: "high",
                malware: 634,
                phishing: 445,
                ddos: 287,
                other: 525
            },
            {
                city: "Đà Nẵng",
                lat: 16.0544,
                lng: 108.2022,
                threats: 456,
                level: "medium",
                malware: 145,
                phishing: 123,
                ddos: 78,
                other: 110
            },
            {
                city: "Hải Phòng",
                lat: 20.8449,
                lng: 106.6881,
                threats: 312,
                level: "medium",
                malware: 98,
                phishing: 87,
                ddos: 56,
                other: 71
            },
            {
                city: "Cần Thơ",
                lat: 10.0452,
                lng: 105.7469,
                threats: 234,
                level: "low",
                malware: 67,
                phishing: 54,
                ddos: 43,
                other: 70
            }
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
            if (threats > 500) return 20;
            if (threats > 200) return 15;
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
                <div style="background: rgba(255,255,255,0.9); padding: 10px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);">
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
                <div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 5px; box-shadow: 0
                <div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);">
                    <h3>🇻🇳 Tình hình An ninh mạng Việt Nam</h3>
                    <p><strong>Tổng mối đe dọa:</strong> <span style="color: #d32f2f; font-weight: bold;">${totalThreats}</span></p>
                    <p><strong>Khu vực nguy hiểm nhất:</strong> TP. Hồ Chí Minh</p>
                    <p><strong>Loại đe dọa phổ biến:</strong> Malware</p>
                    <hr>
                    <small>⚠️ Dữ liệu mô phỏng cho mục đích minh họa</small>
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
st.components.v1.html(map_html, height=600)

# Thêm thông tin cảnh báo bảo mật
st.markdown("---")
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

# Tạo dữ liệu giả lập cho biểu đồ
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
np.random.seed(42)
malware_detections = np.random.poisson(15, len(dates)) + np.random.randint(0, 10, len(dates))
clean_files = np.random.poisson(50, len(dates)) + np.random.randint(10, 30, len(dates))

# Tạo DataFrame
stats_df = pd.DataFrame({
    'Ngày': dates,
    'Mã độc phát hiện': malware_detections,
    'File lành tính': clean_files,
    'Tổng file quét': malware_detections + clean_files
})

# Tạo tabs cho các biểu đồ khác nhau
chart_tabs = st.tabs(["📊 Tổng quan", "🦠 Mã độc", "📈 Xu hướng", "🌍 Phân bố địa lý"])

with chart_tabs[0]:
    st.markdown("##### Thống kê tổng quan trong năm 2024")
    
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
    st.markdown("##### Phân tích chi tiết về mã độc")
    
    # Biểu đồ tròn cho các loại mã độc
    malware_types = ['Trojan', 'Virus', 'Worm', 'Adware', 'Spyware', 'Ransomware', 'Rootkit']
    malware_counts = [1234, 987, 756, 543, 432, 321, 234]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96c93d', '#feca57', '#ff9ff3', '#54a0ff']
        
        wedges, texts, autotexts = ax.pie(malware_counts, labels=malware_types, colors=colors, 
                                         autopct='%1.1f%%', startangle=90, explode=[0.05]*len(malware_types))
        
        ax.set_title('Phân bố các loại mã độc phát hiện', fontsize=14, fontweight='bold')
        
        # Làm đẹp text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("**Top 5 mã độc nguy hiểm nhất:**")
        top_malware = [
            {"name": "Emotet", "detections": 2847, "risk": "Cực cao"},
            {"name": "TrickBot", "detections": 1923, "risk": "Cao"},
            {"name": "Dridex", "detections": 1456, "risk": "Cao"},
            {"name": "Qbot", "detections": 1234, "risk": "Trung bình"},
            {"name": "IcedID", "detections": 987, "risk": "Trung bình"}
        ]
        
        for i, malware in enumerate(top_malware, 1):
            risk_color = {"Cực cao": "#d32f2f", "Cao": "#f57c00", "Trung bình": "#fbc02d"}[malware["risk"]]
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid {risk_color};">
                <strong>{i}. {malware['name']}</strong><br>
                <small>Phát hiện: {malware['detections']:,} lần | Mức độ: <span style="color: {risk_color};">{malware['risk']}</span></small>
            </div>
            """, unsafe_allow_html=True)

with chart_tabs[2]:
    st.markdown("##### Xu hướng phát hiện mã độc")
    
    # Biểu đồ đường xu hướng
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Tạo dữ liệu xu hướng theo tuần
    weekly_stats = stats_df.groupby(stats_df['Ngày'].dt.to_period('W')).mean()
    
    ax.plot(range(len(weekly_stats)), weekly_stats['Mã độc phát hiện'], 
            marker='o', linewidth=2, markersize=4, color='#ff6b6b', label='Mã độc')
    ax.plot(range(len(weekly_stats)), weekly_stats['File lành tính'], 
            marker='s', linewidth=2, markersize=4, color='#51cf66', label='File lành tính')
    
    # Thêm đường xu hướng
    z1 = np.polyfit(range(len(weekly_stats)), weekly_stats['Mã độc phát hiện'], 1)
    p1 = np.poly1d(z1)
    ax.plot(range(len(weekly_stats)), p1(range(len(weekly_stats))), 
            "--", alpha=0.7, color='#ff6b6b', label='Xu hướng mã độc')
    
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
        trend_text = f"📈 Xu hướng tăng {trend_slope:.2f} file mã độc/tuần"
        trend_color = "#ff6b6b"
    else:
        trend_text = f"📉 Xu hướng giảm {abs(trend_slope):.2f} file mã độc/tuần"
        trend_color = "#51cf66"
    
    st.markdown(f"""
    <div style="background: {trend_color}; color: white; padding: 15px; border-radius: 5px; text-align: center; margin: 10px 0;">
        <h4>{trend_text}</h4>
    </div>
    """, unsafe_allow_html=True)

with chart_tabs[3]:
    st.markdown("##### Phân bố mối đe dọa theo khu vực")
    
    # Tạo bảng thống kê theo khu vực
    region_data = [
        {"region": "Miền Bắc", "threats": 2156, "population": "25M", "density": 86.2},
        {"region": "Miền Trung", "threats": 1234, "population": "15M", "density": 82.3},
        {"region": "Miền Nam", "threats": 3421, "population": "35M", "density": 97.7},
        {"region": "Tây Nguyên", "threats": 456, "population": "6M", "density": 76.0},
        {"region": "Đồng bằng Mekong", "threats": 789, "population": "12M", "density": 65.8}
    ]
    
    region_df = pd.DataFrame(region_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Biểu đồ cột ngang
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96c93d', '#feca57']
        bars = ax.barh(region_df['region'], region_df['threats'], color=colors, alpha=0.8)
        
        ax.set_xlabel('Số lượng mối đe dọa')
        ax.set_title('Mối đe dọa theo khu vực')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Thêm nhãn số liệu
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 20, bar.get_y() + bar.get_height()/2, 
                   f'{width:,}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("**Thống kê chi tiết:**")
        
        for region in region_data:
            threat_per_capita = region['threats'] / float(region['population'].replace('M', ''))
            
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 12px; margin: 8px 0; border-radius: 5px;">
                <h5 style="margin: 0 0 8px 0; color: #333;">{region['region']}</h5>
                <div style="display: flex; justify-content: space-between;">
                    <span>Mối đe dọa:</span>
                    <strong>{region['threats']:,}</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Dân số:</span>
                    <strong>{region['population']}</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Mật độ đe dọa:</span>
                    <strong>{threat_per_capita:.1f}/1M dân</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Tỷ lệ nhiễm:</span>
                    <strong>{region['density']:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Thêm phần cảnh báo và khuyến nghị
st.markdown("---")
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
<div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin: 20px 0;">
    <h3>🔐 Malware Detection System</h3>
    <p>Hệ thống phát hiện phần mềm độc hại sử dụng AI</p>
    <p><strong>Phiên bản:</strong> 2.0 >
    <p><small>Được phát triển với ❤️ bởi AI Security Team</small></p>
    <hr style="border-color: rgba(255,255,255,0.3);">
    <p><small>⚠️ Lưu ý: Kết quả từ hệ thống chỉ mang tính chất tham khảo. Luôn sử dụng nhiều công cụ khác nhau để đảm bảo an toàn tối đa.</small></p>
</div>
"""
, unsafe_allow_html=True)