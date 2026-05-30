import streamlit as st
import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict

# --- IMPORT ARSITEKTUR ---
from model import ResNet50_Conventional, ResNet50_Deep
from gan_architectures import GeneratorACGAN, GeneratorWGANGP, GeneratorACWGANGP

# --- IMPORT UTILS ---
from utils import (
    process_image_opencv, 
    get_transform, 
    predict_probabilities,
    generate_heatmap,
    CLASS_NAMES, 
    DEVICE
)

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Skin Disease Analysis & Synthesis",
    layout="wide"
)

# Filter kelas untuk Generator
GEN_CLASSES = [c for c in CLASS_NAMES if 'nv' not in c.lower()]
NUM_GEN_CLASSES = len(GEN_CLASSES)

# ==========================================
# 1. FUNGSI LOAD MODEL (CACHED)
# ==========================================

@st.cache_resource
def load_classifiers():
    models_dict = {}
    
    model_configs = {
        "ResNet50 (Konvensional)": {
            "path": "models/1_ResNet50_Conventional.pth",
            "class": ResNet50_Conventional
        },
        "ResNet50 (ACGAN)": {
            "path": "models/ACGAN_SoftAttn.pth",
            "class": ResNet50_Deep
        },
        "ResNet50 (ACWGAN-GP)": {
            "path": "models/ACWGANGP_SoftAttn.pth",
            "class": ResNet50_Deep
        },
    }
    
    for name, config in model_configs.items():
        path = config["path"]
        ModelClass = config["class"]
        
        if os.path.exists(path):
            try:
                model = ModelClass(num_classes=len(CLASS_NAMES))
                state_dict = torch.load(path, map_location=DEVICE)
                
                # Fix key names if needed
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name_key = k.replace("module.", "")
                    new_state_dict[name_key] = v
                
                model.load_state_dict(new_state_dict, strict=False)
                model.to(DEVICE)
                model.eval()
                models_dict[name] = model
            except Exception as e:
                st.error(f"Gagal memuat {name}: {e}")
                models_dict[name] = None
        else:
            models_dict[name] = None
    return models_dict

@st.cache_resource
def load_generator(model_type):
    config = {}
    
    # Konfigurasi parameter (Pastikan gan_architectures.py sudah fixed ConvTranspose2d)
    if model_type == "ACGAN":
        config = {
            "path": "gans/ACGAN_WeightLoss.pth",
            "class": GeneratorACGAN,
            "feature_size": 64,  # ACGAN base 64
            "conditional": True
        }
    elif model_type == "WGAN-GP":
        config = {
            "path": "gans/WGANGP_WeightLoss.pth",
            "class": GeneratorWGANGP,
            "feature_size": 32,  # WGAN base 32
            "conditional": False
        }
    elif model_type == "ACWGAN-GP":
        config = {
            "path": "gans/ACWGANGP_WeightLoss.pth",
            "class": GeneratorACWGANGP,
            "feature_size": 32,  # ACWGAN base 32
            "conditional": True
        }

    if not os.path.exists(config["path"]):
        st.error(f"File model tidak ditemukan: {config['path']}")
        return None, False

    try:
        ModelClass = config["class"]
        
        # Inisialisasi Model
        if model_type == "ACGAN":
            generator = ModelClass(num_classes=NUM_GEN_CLASSES, latent_dim=100, feature_size=config["feature_size"])
        elif model_type == "WGAN-GP":
            generator = ModelClass(latent_dim=100, feature_size=config["feature_size"])
        elif model_type == "ACWGAN-GP":
            generator = ModelClass(num_classes=NUM_GEN_CLASSES, latent_dim=100, feature_size=config["feature_size"])

        # Load Checkpoint (Extract netG)
        checkpoint = torch.load(config["path"], map_location=DEVICE)
        
        if isinstance(checkpoint, dict) and 'netG' in checkpoint:
            state_dict = checkpoint['netG']
        else:
            state_dict = checkpoint

        # Clean keys
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name_key = k.replace("module.", "")
            new_state_dict[name_key] = v

        generator.load_state_dict(new_state_dict)
        generator.to(DEVICE)
        generator.eval()
        
        return generator, config["conditional"]

    except Exception as e:
        st.error(f"Error loading {model_type}: {e}")
        return None, False

# ==========================================
# UI UTAMA
# ==========================================

st.title("Sistem Klasifikasi & Sintesis Penyakit Kulit")

tab_classify, tab_synthesis = st.tabs(["Diagnosa & Klasifikasi", "Sintesis Generator (Augmentasi)"])

# ==========================================
# TAB 1: KLASIFIKASI & HEATMAP
# ==========================================
with tab_classify:
    loaded_models = load_classifiers()
    
    with st.sidebar:
        st.header("Input Citra Klasifikasi")
        uploaded_file = st.file_uploader("Upload Citra Klinis", type=["jpg", "png", "jpeg"], key="cls_uploader")
        st.divider()
        use_cleaner = st.toggle("Aktifkan Auto-Cleaning", value=True)
        st.caption(f"Device: {str(DEVICE).upper()}")

    if uploaded_file is not None:
        # --- BAGIAN 1: INPUT IMAGE ---
        col_img, col_btn = st.columns([1, 2]) 
        original_image = Image.open(uploaded_file).convert('RGB')
        
        if use_cleaner:
            final_image = process_image_opencv(original_image, use_hair_removal=True, use_cropping=True)
        else:
            final_image = original_image

        with col_img:
            t1, t2 = st.tabs(["Siap Diprediksi", "Original"])
            with t1: st.image(final_image, use_container_width=True, caption="Citra Input Model")
            with t2: st.image(original_image, use_container_width=True, caption="Citra Asli")

        with col_btn:
            st.info("Sistem siap. Klik tombol di bawah untuk menjalankan analisis komparatif dan visualisasi Soft Attention.")
            run_btn = st.button("🚀 Mulai Diagnosis", type="primary")

        # --- BAGIAN 2: HASIL HORIZONTAL (HEATMAP DI BAWAH INPUT) ---
        if run_btn:
            st.divider()
            st.subheader("📊 Hasil Diagnosis & Visualisasi Lesi")
            
            with st.spinner("Sedang menganalisis citra & generate heatmap..."):
                transform = get_transform()
                img_tensor = transform(final_image).unsqueeze(0).to(DEVICE)
                
                all_results = {"Kelas Penyakit": CLASS_NAMES}
                active_models = []
                attention_maps = {} 
                
                # Loop Prediksi
                for model_name, model in loaded_models.items():
                    if model:
                        active_models.append(model_name)
                        probs, attn_map = predict_probabilities(model, img_tensor)
                        all_results[model_name] = probs
                        attention_maps[model_name] = attn_map
                
                df_results = pd.DataFrame(all_results)
                
                # TAMPILAN HORIZONTAL
                if active_models:
                    # Membuat kolom sebanyak jumlah model aktif
                    cols = st.columns(len(active_models))
                    
                    for idx, m_name in enumerate(active_models):
                        best_idx = df_results[m_name].idxmax()
                        best_class = df_results.loc[best_idx, "Kelas Penyakit"]
                        best_conf = df_results.loc[best_idx, m_name]
                        
                        # Memasukkan konten ke dalam kolom spesifik
                        with cols[idx]:
                            st.markdown(f"#### {m_name}") # Nama Model
                            
                            # Tampilkan Metric
                            st.metric(
                                label="Prediksi",
                                value=best_class,
                                delta=f"{best_conf*100:.2f}% Conf"
                            )
                            
                            # Tampilkan Heatmap
                            if attention_maps[m_name] is not None:
                                heatmap_img = generate_heatmap(final_image, attention_maps[m_name], model_name=m_name)
                                st.image(heatmap_img, caption=f"Attention: {m_name}", use_container_width=True)
                            else:
                                st.image(final_image, caption="No Attention Map", use_container_width=True)

                # --- BAGIAN 3: TABEL PROBABILITAS (PALING BAWAH) ---
                st.divider()
                st.subheader("📑 Detail Probabilitas Lengkap")
                
                # Format Progress Bar
                col_config = {
                    "Kelas Penyakit": st.column_config.TextColumn("Kelas Penyakit", width="medium")
                }
                for m_name in active_models:
                    col_config[m_name] = st.column_config.ProgressColumn(
                        m_name, format="%.4f", min_value=0, max_value=1
                    )

                st.dataframe(df_results, use_container_width=True, hide_index=True, column_config=col_config)

# ==========================================
# TAB 2: SINTESIS GENERATOR
# ==========================================
with tab_synthesis:
    st.header("Visualisasi Augmentasi Generatif")
    
    col_params, col_display = st.columns([1, 2])
    
    with col_params:
        st.subheader("Parameter Sintesis")
        gen_model_choice = st.selectbox("Pilih Model Generatif", ["ACGAN", "WGAN-GP", "ACWGAN-GP"])
        
        is_conditional = (gen_model_choice != "WGAN-GP")
        
        target_label_idx = 0
        target_class_name = "Random"
        
        if is_conditional:
            target_class_name = st.selectbox("Pilih Kelas Target", GEN_CLASSES)
            target_label_idx = GEN_CLASSES.index(target_class_name)
        else:
            st.info("Mode Unconditional (Random)")
        
        num_samples = st.slider("Jumlah Sampel", 1, 10, 5)
        generate_btn = st.button("Generate Citra", type="primary")

    with col_display:
        st.subheader("Hasil Sintesis")
        
        if generate_btn:
            with st.spinner(f"Memuat model {gen_model_choice}..."):
                generator, model_is_conditional = load_generator(gen_model_choice)
                
                if generator:
                    latent_dim = 100 
                    noise = torch.randn(num_samples, latent_dim).to(DEVICE)
                    
                    fake_imgs_tensor = None
                    with torch.no_grad():
                        if model_is_conditional:
                            label_tensor = torch.full((num_samples,), target_label_idx, dtype=torch.long).to(DEVICE)
                            fake_imgs_tensor = generator(noise, labels=label_tensor)
                        else:
                            fake_imgs_tensor = generator(noise)
                    
                    # Tampilkan Grid
                    cols = st.columns(5) # Max 5 per row
                    for i in range(num_samples):
                        img_t = fake_imgs_tensor[i]
                        img_t = (img_t * 0.5) + 0.5 
                        ndarr = img_t.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                        im_result = Image.fromarray(ndarr)
                        
                        row_idx = i // 5
                        col_idx = i % 5
                        
                        with cols[col_idx]:
                            st.image(im_result, caption=f"Img {i+1}", use_container_width=True)
                    
                    st.success("Selesai.")
                    
# venv\Scripts\activate

