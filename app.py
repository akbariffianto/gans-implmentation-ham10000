import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import cv2  # Library Computer Vision
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Skin Disease Classifier",
    page_icon="🔬",
    layout="wide"
)

# ==========================================
# 1. DEFINISI ARSITEKTUR MODEL
# ==========================================
class SoftAttention(nn.Module):
    def __init__(self, in_channels, k_maps=16):
        super(SoftAttention, self).__init__()
        self.in_channels = in_channels
        self.k = k_maps
        self.attn_conv = nn.Conv2d(in_channels, k_maps, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        attn_maps = self.attn_conv(x)
        b, k, h, w = attn_maps.size()
        attn_maps = attn_maps.view(b, k, -1)
        attn_maps = F.softmax(attn_maps, dim=-1)
        attn_maps = attn_maps.view(b, k, h, w)
        scaled_attn = attn_maps * self.gamma
        out = torch.cat([x, scaled_attn], dim=1)
        return out

class SoftAttentionModule(nn.Module):
    def __init__(self, in_channels, k_maps=16):
        super(SoftAttentionModule, self).__init__()
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.soft_attention = SoftAttention(in_channels, k_maps)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu1(x)
        path1 = self.maxpool1(x)
        path2_out = self.soft_attention(x)
        path2 = self.maxpool2(path2_out)
        out = torch.cat([path1, path2], dim=1)
        out = self.relu2(out)
        out = self.dropout(out)
        return out

class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes=7, k_attention_maps=16):
        super(ModifiedResNet50, self).__init__()
        original_resnet = models.resnet50(weights=None) 
        self.features = nn.Sequential(
            original_resnet.conv1,
            original_resnet.bn1,
            original_resnet.relu,
            original_resnet.maxpool,
            original_resnet.layer1,
            original_resnet.layer2,
            original_resnet.layer3
        )
        feature_channels = 1024 
        self.sa_module = SoftAttentionModule(in_channels=feature_channels, k_maps=k_attention_maps)
        final_channels = feature_channels + (feature_channels + k_attention_maps)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(final_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.sa_module(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ==========================================
# 2. HELPER FUNCTION: SMART PREPROCESSING
# ==========================================
def remove_hair(image_cv):
    """Menghilangkan rambut menggunakan BlackHat Morphological Operation"""
    # 1. Convert ke Grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    
    # 2. Kernel untuk mendeteksi struktur tipis (rambut)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    
    # 3. BlackHat transform (menemukan objek gelap di latar terang)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 4. Thresholding untuk membuat mask rambut
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # 5. Inpainting (menambal area rambut dengan pixel sekitarnya)
    final_image = cv2.inpaint(image_cv, thresh, 1, cv2.INPAINT_TELEA)
    
    return final_image

def crop_contour(image_cv):
    """Mencari kontur terbesar (kulit) dan crop area tersebut"""
    # 1. Convert ke Grayscale & Blur
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Otsu Thresholding (Memisahkan foreground/kulit dari background gelap/terang)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Cari Kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image_cv # Jika gagal nemu kontur, kembalikan gambar asli
    
    # 4. Ambil kontur dengan area TERBESAR (asumsi itu adalah lesi/kulit)
    c = max(contours, key=cv2.contourArea)
    
    # 5. Buat Bounding Box (Kotak)
    x, y, w, h = cv2.boundingRect(c)
    
    # Validasi: Jangan crop jika kotaknya terlalu kecil (noise)
    h_img, w_img, _ = image_cv.shape
    if w < w_img * 0.1 or h < h_img * 0.1: 
        return image_cv
        
    # 6. Crop Gambar
    cropped = image_cv[y:y+h, x:x+w]
    return cropped

def process_image_opencv(pil_image, use_hair_removal=True, use_cropping=True):
    # Convert PIL ke OpenCV (Numpy)
    img_cv = np.array(pil_image)
    
    # OpenCV pakai BGR, PIL pakai RGB. Streamlit tampilkan RGB.
    # Kita proses dalam mode RGB saja biar aman.
    
    processed = img_cv.copy()
    
    if use_cropping:
        try:
            processed = crop_contour(processed)
        except Exception:
            pass # Fallback ke gambar asli jika error
            
    if use_hair_removal:
        try:
            processed = remove_hair(processed)
        except Exception:
            pass

    # Convert balik ke PIL
    return Image.fromarray(processed)


# ==========================================
# 3. CONFIG & LOAD MODEL
# ==========================================
CLASS_NAMES = [
    'Actinic Keratoses (akiec)', 'Basal Cell Carcinoma (bcc)', 'Benign Keratosis (bkl)', 
    'Dermatofibroma (df)', 'Melanoma (mel)', 'Melanocytic Nevi (nv)', 'Vascular Lesions (vasc)'
]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

@st.cache_resource
def load_models():
    models_dict = {}
    model_paths = {
        "ResNet50 (Augmentasi Normal)": "models/1_ResNet50_Conventional.pth",
        "ResNet50 (ACGAN)": "models/1_ResNet50_ACGAN.pth",
        "ResNet50 (WGAN-GP)": "models/1_ResNet50_WGAN.pth",
        "ResNet50 (ACWGAN-GP)": "models/1_ResNet50_ACWGAN.pth",
    }
    for name, path in model_paths.items():
        try:
            if os.path.exists(path):
                model = ModifiedResNet50(num_classes=NUM_CLASSES)
                state_dict = torch.load(path, map_location=DEVICE)
                model.load_state_dict(state_dict)
                model.to(DEVICE)
                model.eval()
                models_dict[name] = model
            else:
                models_dict[name] = None
        except Exception:
            models_dict[name] = None
    return models_dict

def predict_probabilities(model, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()[0]

# --- UI UTAMA ---
st.title("🔬 Analisis Penyakit Kulit Multi-Model")

loaded_models = load_models()

with st.sidebar:
    st.header("Upload Citra")
    uploaded_file = st.file_uploader("Format: JPG, PNG", type=["jpg", "png", "jpeg"])
    
    st.divider()
    st.header("⚙️ Preprocessing (PENTING)")
    st.info("Aktifkan fitur ini jika gambar mengandung rambut atau background mengganggu.")
    
    # OPSI PREPROCESSING
    use_cleaner = st.toggle("🔍 Aktifkan Auto-Cleaning", value=True)
    
    st.caption(f"Device: {str(DEVICE).upper()}")

if uploaded_file is not None:
    col_img, col_ctrl = st.columns([1, 1], gap="large") 
    
    # 1. LOAD IMAGE ASLI
    original_image = Image.open(uploaded_file).convert('RGB')
    
    # 2. PROSES GAMBAR (JIKA TOGGLE AKTIF)
    if use_cleaner:
        final_image = process_image_opencv(original_image, use_hair_removal=True, use_cropping=True)
    else:
        final_image = original_image

    with col_img:
        # Tampilkan Perbandingan jika Cleaning Aktif
        if use_cleaner:
            tab1, tab2 = st.tabs(["🖼️ Final (Bersih)", "📂 Asli"])
            with tab1:
                st.image(final_image, use_container_width=True, caption="Siap Diprediksi (Hair Removal + Crop)")
            with tab2:
                st.image(original_image, use_container_width=True, caption="Upload Asli")
        else:
            st.image(original_image, use_container_width=True, caption="Citra Input")

    with col_ctrl:
        st.subheader("Persetujuan Medis")
        st.warning("""
        **PERHATIAN:** Aplikasi ini menggunakan AI dan hasil prediksi **bukan** diagnosis medis final. 
        Kesalahan prediksi mungkin terjadi. Konsultasikan dengan dokter spesialis kulit.
        """, icon="⚠️")
        
        agree = st.checkbox("Saya mengerti dan menyetujui pernyataan di atas.")
        st.markdown("---")
        run_btn = st.button("🚀 Jalankan Diagnosis", type="primary", use_container_width=True)

    if run_btn:
        if not agree:
            with col_ctrl:
                st.error("⛔ Harap centang persetujuan di atas.")
        else:
            st.divider()
            with st.spinner("Menganalisis citra yang sudah dibersihkan..."):
                transform = get_transform()
                # GUNAKAN FINAL IMAGE (YANG SUDAH BERSIH) UNTUK PREDIKSI
                img_tensor = transform(final_image).unsqueeze(0)
                
                all_results = {"Jenis Penyakit": CLASS_NAMES}
                model_names = []
                
                for model_name, model in loaded_models.items():
                    if model:
                        model_names.append(model_name)
                        probs = predict_probabilities(model, img_tensor)
                        all_results[model_name] = probs
                
                df_results = pd.DataFrame(all_results)

                # SUMMARY
                st.subheader("🏆 Prediksi Utama")
                summary_cols = st.columns(len(model_names))
                for idx, m_name in enumerate(model_names):
                    best_idx = df_results[m_name].idxmax()
                    best_class = df_results.loc[best_idx, "Jenis Penyakit"]
                    best_conf = df_results.loc[best_idx, m_name]
                    with summary_cols[idx]:
                        st.info(f"**{m_name}**\n\n### {best_class}\n\nConf: **{best_conf*100:.2f}%**")

                # TABEL DETAIL
                st.markdown("---")
                st.subheader("📊 Tabel Probabilitas Lengkap")
                column_config = {"Jenis Penyakit": st.column_config.TextColumn("Jenis Penyakit", width="medium")}
                for m_name in model_names:
                    column_config[m_name] = st.column_config.ProgressColumn(m_name, format="%.2f%%", min_value=0, max_value=1)

                st.dataframe(df_results, column_config=column_config, hide_index=True, use_container_width=True, height=300)
else:
    st.info("👋 Silakan upload gambar melalui sidebar.")