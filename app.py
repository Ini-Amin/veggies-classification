import os
import numpy as np
import streamlit as st
from PIL import Image

# Setup Halaman
st.set_page_config(
    page_title="Klasifikasi Gambar Sayur",
    page_icon="🥦",
    layout="centered"
)

st.title("🥦 Aplikasi Klasifikasi Sayuran")
st.write("Unggah foto sayuran untuk memprediksi jenisnya.")

# Path Model & Label TFLite
MODEL_PATH = "submission/tflite/model.tflite"
LABEL_PATH = "submission/tflite/label.txt"

# Memuat nama kelas/label
@st.cache_resource
def load_labels():
    if os.path.exists(LABEL_PATH):
        with open(LABEL_PATH, "r") as f:
            return [line.strip() for line in f.readlines()]
    else:
        # Default labels dari dataset
        return [
            "Bean", "Bitter Gourd", "Bottle Gourd", "Brinjal", "Broccoli",
            "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber",
            "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"
        ]

labels = load_labels()

# Memuat Interpreter TFLite (menggunakan tflite_runtime/tensorflow)
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model tidak ditemukan di {MODEL_PATH}. Pastikan Anda sudah menjalankan seluruh cell di notebook dan folder 'submission/' ada di direktori yang sama.")
        return None
    
    # Deteksi backend interpreter
    try:
        import ai_edge_litert.interpreter as tflite
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                st.error("Gagal memuat interpreter TFLite. Silakan install `ai-edge-litert`, `tflite-runtime` atau `tensorflow`.")
                return None

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Preprocessing Gambar
def preprocess_image(image, target_size=(224, 224)):
    img = image.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0).astype(np.float32)
    return arr

# Widget Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar sayur...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)
    
    if interpreter is not None:
        with st.spinner("Menganalisis gambar..."):
            # Preprocess
            input_data = preprocess_image(image)
            
            # Dapatkan tensor input & output
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Jalankan inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            # Ambil hasil prediksi
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]
            pred_idx = np.argmax(output_data)
            confidence = output_data[pred_idx]
            
            # Tampilkan Hasil
            st.success(f"### Prediksi: **{labels[pred_idx]}**")
            st.write(f"Confidence score: **{confidence * 100:.2f}%**")
            
            # Progres bar
            st.progress(float(confidence))
    else:
        st.warning("Model tidak siap.")
