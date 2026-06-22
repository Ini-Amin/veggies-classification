# 🥦 Panduan Presentasi UAS PPL: Klasifikasi Sayuran dengan Custom CNN & Streamlit

Dokumen ini disusun untuk membantu Anda mempersiapkan presentasi besok. Panduan ini mencakup struktur presentasi, poin penting yang harus Anda kuasai (pusat perhatian dosen), dan daftar simulasi pertanyaan dosen beserta jawabannya.

---

## 📅 Struktur Slide Presentasi yang Direkomendasikan
Berikut adalah outline slide yang bisa Anda gunakan agar alur penjelasan sistematis dan profesional:

| Slide # | Judul Slide | Konten Utama |
| :--- | :--- | :--- |
| **1** | **Judul Project & Anggota** | Judul: *Klasifikasi Gambar Sayuran Menggunakan Custom Convolutional Neural Network (CNN) dan Deployment Berbasis Streamlit Cloud*. |
| **2** | **Latar Belakang & Masalah** | Pentingnya klasifikasi sayuran secara otomatis (industri agrikultur, e-commerce, kasir pintar). Dataset dari Kaggle (15 kelas sayur, total 21.000 gambar). |
| **3** | **Pre-processing & Augmentasi** | Normalisasi gambar (1./255), resolusi `224x224` piksel. Teknik augmentasi data pada set latih (rotasi, zoom, geser, flip) untuk mencegah overfitting. |
| **4** | **Arsitektur Model (CNN)** | Penjelasan 4 block Convolutional (32, 64, 128, 128 filter) + MaxPooling2D, dilanjutkan dengan Flatten, Dropout (0.5), dan Dense Layer (512 unit & 15 unit Softmax). |
| **5** | **Proses & Optimasi Training** | Penggunaan Optimizer Adam (learning rate 0.001), loss function `categorical_crossentropy`, serta 3 Callbacks vital: `EarlyStopping`, `ReduceLROnPlateau`, dan `ModelCheckpoint`. |
| **6** | **Hasil Evaluasi Model** | Akurasi pengujian (Test Accuracy) mencapai **99.17%**, grafik training/validation loss & accuracy, Confusion Matrix, dan Classification Report (Precision, Recall, F1-Score). |
| **7** | **Optimasi Deployment (TFLite)** | Konversi model Keras (`.keras`) ke TensorFlow Lite (`.tflite`). Mengapa dikonversi? (Mengecilkan ukuran model dari ~116MB menjadi lebih ringan & hemat RAM saat dideploy). |
| **8** | **Demo Aplikasi (Streamlit)** | Tampilan web app Streamlit, fitur upload multi-file gambar, proses inferensi menggunakan interpreter TFLite, hasil prediksi, dan confidence score. |
| **9** | **Kesimpulan & Saran** | Model berhasil mendeteksi 15 kelas sayur dengan akurasi sangat tinggi. Pengembangan berikutnya bisa menambahkan kelas atau optimasi inferensi mobile. |

---

## 🔍 Poin Penting yang Harus Anda Kuasai (Perhatian Khusus)
Dosen penguji biasanya mencari pemahaman fundamental mengapa Anda memilih parameter atau metode tertentu. Kuasai 5 poin utama ini:

1. **Mengapa Data Augmentasi Hanya di Train Set?**
   > [!IMPORTANT]
   > Augmentasi data (seperti rotasi, flip, zoom) bertujuan untuk mensimulasikan variasi foto sayur di dunia nyata (misal: pencahayaan berbeda, sudut miring). Ini hanya dilakukan pada data **Training** agar model belajar generalisasi. Data **Validation** dan **Test** tidak boleh diaugmentasi karena bertindak sebagai data uji riil untuk mengukur performa model yang sebenarnya secara jujur.

2. **Perhitungan Dimensi Feature Map (Convolution & Pooling)**
   * Gambar input berukuran `224x224x3`.
   * Setelah **Conv2D + MaxPooling (2x2)** pertama, ukuran menyusut menjadi `112x112x32`.
   * Block 2: Menyusut menjadi `56x56x64`.
   * Block 3: Menyusut menjadi `28x28x128`.
   * Block 4: Menyusut menjadi `14x14x128`.
   * Setelah **Flatten**, jumlah fiturnya menjadi $14 \times 14 \times 128 = 25.088$ dimensi sebelum masuk ke Dense layer.

3. **Fungsi 3 Callbacks yang Anda Gunakan**
   * **EarlyStopping (patience=5):** Menghentikan training jika `val_loss` tidak membaik selama 5 epoch berturut-turut. Ini mencegah overfitting dan menghemat waktu komputasi.
   * **ReduceLROnPlateau (factor=0.5, patience=3):** Menurunkan learning rate setengahnya jika `val_loss` mandek selama 3 epoch. Membantu model menemukan minimum global secara perlahan tanpa melompati lembah gradien (overshooting).
   * **ModelCheckpoint:** Hanya menyimpan bobot terbaik (berdasarkan `val_accuracy` tertinggi) ke dalam file `best_model.keras`.

4. **Kenapa Categorical Crossentropy?**
   * Karena kelas target berjumlah 15 (multiclass) dan data generator diset menggunakan `class_mode="categorical"` yang mengubah label kelas menjadi format **One-Hot Encoded** (misal: kelas `0` menjadi `[1, 0, 0, ...]`). Jika label berupa integer (0, 1, 2...), maka loss function yang digunakan harusnya `sparse_categorical_crossentropy`.

5. **Mengapa Menggunakan TFLite untuk Deployment?**
   * Model asli `.keras` memiliki ukuran sekitar 116MB. Dengan konversi ke TFLite (`model.tflite`), struktur graf disederhanakan.
   * Format TFLite menggunakan engine inferensi yang sangat ringan (LiteRT), menghemat memori RAM server Streamlit Cloud (yang memiliki limitasi RAM gratis), sehingga aplikasi tidak mudah *crash* dan inferensi berjalan lebih instan (<0.5 detik per gambar).

---

## 🙋‍♂️ Simulasi Pertanyaan Dosen & Cara Menjawabnya (Mock Q&A)

### Kategori 1: Dataset & Preprocessing

**Q1: Mengapa Anda melakukan normalisasi gambar dengan membaginya dengan 255.0 (`rescale=1./255`)?**
* **Jawaban:** Nilai piksel gambar RGB berkisar antara 0 sampai 255. Jika langsung dimasukkan ke model, nilai input yang besar dapat membuat proses perhitungan gradien selama backpropagation menjadi tidak stabil (exploding gradients) dan memperlambat konvergensi. Dengan membaginya dengan 255.0, nilai diubah menjadi skala `[0, 1]`. Ini membuat rentang data seragam, mempercepat proses training, dan membantu optimasi gradien descent berjalan lebih mulus.

**Q2: Bagaimana Anda menangani ketidakseimbangan kelas (class imbalance) pada dataset ini?**
* **Jawaban:** Dataset ini sangat seimbang (balanced dataset). Masing-masing dari 15 kelas sayuran memiliki tepat 1.000 gambar untuk training, 200 gambar untuk validation, dan 200 gambar untuk testing. Karena distribusinya seimbang, kita tidak perlu melakukan teknik penanganan ketidakseimbangan seperti class weighting atau oversampling/undersampling.

---

### Kategori 2: Arsitektur Model (CNN)

**Q3: Mengapa Anda memilih membuat Custom CNN sendiri alih-alih menggunakan Transfer Learning (seperti MobileNet atau ResNet)?**
* **Jawaban:** Ada dua alasan utama. Pertama, untuk tujuan akademis/pembelajaran (UAS PPL), membangun Custom CNN membantu kami memahami cara kerja ekstraksi fitur (lokalitas piksel lewat kernel konvolusi) dan pengurangan dimensi secara langsung dari dasar. Kedua, dataset sayuran ini memiliki visual yang relatif sederhana dengan latar belakang bersih. Custom CNN dengan 4 block konvolusi sudah terbukti sangat memadai untuk mengekstrak fitur bentuk dan warna sayur, menghasilkan akurasi uji **99.17%** tanpa overhead parameter berlebih dari model besar (seperti ResNet) yang rentan overfitting pada dataset khusus ini.

**Q4: Apa fungsi dari layer `MaxPooling2D` dalam model Anda?**
* **Jawaban:** `MaxPooling2D` berfungsi untuk melakukan *downsampling* (pengurangan dimensi spasial) dari feature map. Cara kerjanya adalah mengambil nilai maksimum pada window tertentu (misal 2x2 piksel). Manfaatnya adalah:
  1. Mengurangi jumlah parameter dan komputasi di layer berikutnya (mencegah overfitting).
  2. Memberikan sifat *translation invariance* (model tetap mengenali objek sayur meskipun posisinya sedikit bergeser).
  3. Mempertahankan fitur yang paling dominan/menonjol.

**Q5: Mengapa Anda menggunakan fungsi aktivasi `Softmax` di output layer dan bukan `Sigmoid` atau `ReLU`?**
* **Jawaban:** 
  * `ReLU` tidak cocok di layer output karena menghasilkan nilai dari $0$ hingga tak terhingga, padahal kita membutuhkan probabilitas kelas.
  * `Sigmoid` digunakan untuk klasifikasi biner (2 kelas) atau multi-label (satu gambar bisa berisi banyak objek sayuran sekaligus).
  * `Softmax` digunakan karena ini adalah kasus klasifikasi multi-kelas (15 kelas) yang bersifat saling lepas (mutually exclusive) — artinya setiap gambar hanya boleh berisi satu jenis sayuran saja. `Softmax` akan menormalisasi output layer dense menjadi distribusi probabilitas di mana total nilai dari 15 kelas tersebut berjumlah tepat $1.0$ (atau 100%).

---

### Kategori 3: Pelatihan & Regulasi (Regularization)

**Q6: Anda meletakkan `Dropout(0.5)` sebelum Dense layer pertama dan kedua. Apa sebenarnya fungsi Dropout dan bagaimana nilai 0.5 itu bekerja?**
* **Jawaban:** `Dropout` adalah teknik regulasi untuk mencegah overfitting. Nilai `0.5` berarti pada setiap langkah training (batch), secara acak 50% neuron pada layer tersebut akan dinonaktifkan (di-set nilainya menjadi 0). Hal ini memaksa model untuk tidak bergantung pada neuron tertentu saja untuk membuat keputusan (mencegah *co-adaptation*), sehingga model terpaksa mempelajari representasi fitur yang lebih kuat dan tangguh. Saat tahap evaluasi (testing/inference), seluruh neuron diaktifkan kembali.

**Q7: Optimizer apa yang Anda gunakan dan mengapa memilih itu?**
* **Jawaban:** Kami menggunakan optimizer **Adam** dengan learning rate default sebesar `0.001` (`1e-3`). Adam menggabungkan keunggulan dari dua optimizer lain: SGD dengan momentum (mempercepat komputasi ke arah gradien yang benar) dan RMSprop (mengadaptasi learning rate secara individual untuk setiap parameter berdasarkan rata-rata kuadrat gradien terbaru). Kombinasi ini membuat Adam sangat populer karena cepat konvergen dan stabil pada sebagian besar masalah computer vision.

---

### Kategori 4: Evaluasi & Metrik Performa

**Q8: Jika akurasi uji Anda mencapai 99.17%, apakah model ini terindikasi Overfitting? Bagaimana Anda membuktikannya?**
* **Jawaban:** Tidak, model ini tidak mengalami overfitting yang signifikan. Kami membuktikannya dengan melihat grafik hasil training (Loss dan Accuracy):
  1. Nilai kurva `Train Loss` dan `Validation Loss` menurun bersama-sama dan stabil di nilai yang sangat rendah.
  2. Selisih antara `Train Accuracy` (~98%) dan `Validation/Test Accuracy` (~99%) sangat kecil dan tidak terjadi divergensi (di mana train loss terus turun tapi validation loss mendadak naik kembali).
  3. Regulasi kuat berupa dua layer `Dropout(0.5)` dan teknik *Early Stopping* telah berhasil menekan potensi overfitting.

**Q9: Apa perbedaan antara Precision, Recall, dan F1-Score pada laporan klasifikasi Anda? Mengapa Confusion Matrix penting?**
* **Jawaban:** 
  * **Precision:** Mengukur seberapa akurat prediksi positif model. Dari semua gambar yang diprediksi sebagai "Tomato", berapa persen yang benar-benar tomat asli.
  * **Recall (Sensitivity):** Mengukur seberapa baik model menemukan seluruh sampel positif. Dari seluruh gambar tomat asli di dataset, berapa persen yang berhasil dideteksi oleh model sebagai "Tomato".
  * **F1-Score:** Rata-rata harmonik antara Precision dan Recall. Menjadi indikator performa seimbang jika kedua metrik di atas sama pentingnya.
  * **Confusion Matrix** penting karena memetakan secara detail di mana letak kesalahan model. Misalnya, apakah ada gambar wortel (Carrot) yang sering salah diprediksi sebagai lobak (Radish). Ini membantu kami menganalisis kelemahan klasifikasi model secara visual.

---

### Kategori 5: Deployment & Arsitektur Sistem

**Q10: Mengapa Anda membuat backend aplikasi menggunakan TensorFlow Lite (TFLite) interpreter di Streamlit dan bukan model Keras penuh (`best_model.keras`)?**
* **Jawaban:** Ada beberapa keuntungan teknis utama:
  1. **Ukuran File:** File model Keras asli berukuran besar (~116MB). File TFLite yang dihasilkan lebih kompak.
  2. **Efisiensi RAM:** Mengimpor modul `tensorflow` penuh di server Streamlit Cloud memerlukan RAM dan CPU yang besar, yang sering kali melebihi batas resource gratis Streamlit Cloud (~1GB RAM) dan membuat server restart. Dengan TFLite, kita bisa menggunakan runtime interpreter yang sangat ringan (seperti `tflite_runtime` atau `ai_edge_litert`) tanpa perlu memuat seluruh library TensorFlow.
  3. **Kecepatan Inferensi:** Inferensi TFLite di CPU server cloud jauh lebih cepat dan efisien dibandingkan menjalankan model Keras penuh.

**Q11: Bagaimana alur pengolahan gambar pada aplikasi web Streamlit Anda sebelum gambar tersebut diklasifikasikan oleh TFLite?**
* **Jawaban:** Gambar yang diunggah oleh pengguna melalui widget `st.file_uploader` diproses melalui fungsi `preprocess_image` di [app.py](file:///D:/TungTungTungSahur/UAS%20PPL/app.py):
  1. Gambar dikonversi ke format warna **RGB** menggunakan Pillow untuk mengantisipasi gambar PNG transparan (RGBA).
  2. Gambar di-*resize* ke ukuran target **`224x224`** piksel (sesuai spesifikasi input model CNN).
  3. Gambar dikonversi menjadi array NumPy dan nilainya dibagi **`255.0`** untuk menormalisasi nilai piksel ke rentang `[0.0, 1.0]`.
  4. Dimensi array diperluas pada aksis pertama menggunakan `np.expand_dims(arr, 0)` sehingga bentuknya menjadi `(1, 224, 224, 3)`. Langkah ini penting karena model mengharapkan input dalam bentuk batch (meskipun batch size-nya hanya 1 gambar).
  5. Data bertipe `float32` diumpankan ke TFLite Interpreter untuk dilakukan proses inferensi.

---

## 💡 Tips Tambahan untuk Presentasi Esok Hari
* **Lakukan Demo Langsung:** Dosen sangat menyukai demo produk nyata. Jalankan aplikasi Streamlit Anda secara langsung atau tunjukkan tautan publiknya di [veggies-classification-uas.streamlit.app](https://veggies-classification-uas.streamlit.app/).
* **Siapkan Sampel Gambar Uji:** Siapkan 2-3 gambar sayuran dari Google (yang tidak ada di dataset) untuk dicoba langsung saat demo guna membuktikan bahwa model Anda benar-benar berfungsi dengan baik pada data baru.
* **Akui Limitasi Model:** Jika dosen bertanya tentang kekurangan, sampaikan secara jujur bahwa model ini dilatih dengan latar belakang sayur yang bersih (clean background). Model mungkin mengalami penurunan akurasi jika diuji dengan gambar sayur yang berada di tumpukan pasar yang padat atau dengan kondisi cahaya sangat minim. Ini menunjukkan sikap ilmiah yang objektif.

*Semoga sukses untuk presentasi UAS PPL Anda besok!*
