# 🥦 Aplikasi Klasifikasi Sayuran (Vegetable Classification App)

Aplikasi web interaktif berbasis **Streamlit** untuk mengklasifikasikan jenis sayuran dari gambar secara real-time. Model deep learning dideploy menggunakan **LiteRT (TensorFlow Lite)** agar proses inferensi berjalan cepat dan ringan, bahkan di lingkungan komputasi terbatas seperti Streamlit Community Cloud.

🔗 **Link Aplikasi:** [veggies-classification-uas.streamlit.app](https://veggies-classification-uas.streamlit.app/)

---

## 📊 Dataset
Dataset yang digunakan adalah [Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset) dari Kaggle. Dataset ini terdiri dari **15 kelas sayuran**, dengan total 21.000 gambar (masing-masing kelas memiliki 1.000 gambar latih, 200 gambar validasi, dan 200 gambar uji):

*   *Bean* (Buncis)
*   *Bitter Gourd* (Pare)
*   *Bottle Gourd* (Labu Air)
*   *Brinjal* (Terung)
*   *Broccoli* (Brokoli)
*   *Cabbage* (Kubis)
*   *Capsicum* (Paprika)
*   *Carrot* (Wortel)
*   *Cauliflower* (Kembang Kol)
*   *Cucumber* (Timun)
*   *Papaya* (Pepaya)
*   *Potato* (Kentang)
*   *Pumpkin* (Labu Kuning)
*   *Radish* (Lobak)
*   *Tomato* (Tomat)

---

## 🧠 Arsitektur Model & Performa
Model klasifikasi dibangun menggunakan arsitektur **Custom CNN (Convolutional Neural Network)** berbasis TensorFlow/Keras:

1.  **Input Layer:** Menerima gambar dengan resolusi `224x224` piksel (RGB).
2.  **Feature Extraction:**
    *   **Block 1:** `Conv2D` (32 filter, kernel 3x3, ReLU) + `MaxPooling2D` (2x2)
    *   **Block 2:** `Conv2D` (64 filter, kernel 3x3, ReLU) + `MaxPooling2D` (2x2)
    *   **Block 3:** `Conv2D` (128 filter, kernel 3x3, ReLU) + `MaxPooling2D` (2x2)
    *   **Block 4:** `Conv2D` (128 filter, kernel 3x3, ReLU) + `MaxPooling2D` (2x2)
3.  **Classification Head:**
    *   `Flatten` & `Dropout(0.5)` untuk mencegah overfitting.
    *   `Dense` (512 unit, ReLU) & `Dropout(0.5)`.
    *   `Dense` (15 unit, Softmax) sebagai output klasifikasi.

### Hasil Evaluasi Model
*   **Test Accuracy:** **99.17%**
*   Model dikonversi menjadi format `.tflite` untuk digunakan pada aplikasi web agar ukuran file lebih kecil (~116MB pada Keras dikompresi menjadi format TFLite yang efisien) dan meminimalkan konsumsi RAM saat dijalankan.

---

## 📁 Struktur Direktori Project
```text
veggies-classification/
│
├── submission/
│   └── tflite/
│       ├── model.tflite          # Model TensorFlow Lite hasil konversi
│       └── label.txt             # Daftar label sayuran (15 kelas)
│
├── app.py                        # Kode utama aplikasi Streamlit
├── requirements.txt              # Daftar dependensi Python
├── veggies_classification.ipynb  # Notebook pelatihan dan konversi model
├── best_model.keras              # Model Keras terbaik (.keras format)
└── README.md                     # Dokumentasi project (file ini)
```

---

## 🚀 Cara Menjalankan Project Secara Lokal

### Prasyarat
Pastikan Anda sudah menginstal Python (disarankan versi 3.10 ke atas) dan `pip`.

### Langkah-langkah
1.  **Clone repositori ini:**
    ```bash
    git clone https://github.com/Ini-Amin/veggies-classification.git
    cd veggies-classification
    ```

2.  **Instal dependensi:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Jalankan aplikasi Streamlit:**
    ```bash
    streamlit run app.py
    ```

4.  Buka browser Anda dan akses aplikasi di `http://localhost:8501`.

---

## 🛠️ Stack Teknologi
*   **Deep Learning Framework:** TensorFlow & Keras
*   **Deployment Runtime:** LiteRT (oleh Google, suksesor TensorFlow Lite)
*   **Web Framework:** Streamlit
*   **Image Processing & Utils:** NumPy, Pillow (PIL)
*   **Programming Language:** Python 3.14 (Streamlit Cloud runtime) / Python 3.10+ (Local)
