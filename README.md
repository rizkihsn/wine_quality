# 🍷 Wine Quality Prediction — Neural Network Backpropagation

Aplikasi web prediksi kualitas wine menggunakan algoritma **Backpropagation** pada **Jaringan Saraf Tiruan (JST)**, dibangun dengan Python, TensorFlow/Keras, dan Flask.

---

## 📋 Deskripsi Proyek

Proyek ini merupakan implementasi model Neural Network dengan algoritma Backpropagation untuk mengklasifikasikan kualitas wine menjadi dua kategori:
- **GOOD** — Wine berkualitas tinggi (skor ≥ 6)
- **BAD** — Wine berkualitas rendah (skor < 6)

Dataset yang digunakan: **Wine Quality Dataset (WineQT)** dari Kaggle dengan **1.143 sampel** dan **11 fitur fisikokimia**.

---

## 🏗️ Arsitektur Model

```
Input Layer  : 11 fitur fisikokimia
Hidden Layer 1: 64 neuron (ReLU) + Dropout(0.2)
Hidden Layer 2: 32 neuron (ReLU) + Dropout(0.2)
Hidden Layer 3: 16 neuron (ReLU) + Dropout(0.1)
Output Layer : 1 neuron (Sigmoid)
```

**Optimizer:** Adam | **Loss:** Binary Crossentropy | **Early Stopping:** patience=15

---

## 📊 Hasil Evaluasi Model

| Metrik | Nilai |
|--------|-------|
| Test Accuracy | ~73-76% |
| Precision (GOOD) | ~77% |
| Recall (GOOD) | ~72% |
| Teknik | Early Stopping |

---

## 🛠️ Teknologi yang Digunakan

- **Backend:** Python, Flask
- **Model:** TensorFlow/Keras (Backpropagation)
- **Frontend:** HTML, CSS (Bootstrap 5), JavaScript
- **Data:** Pandas, NumPy, Scikit-learn
- **Visualisasi:** Matplotlib, Seaborn

---

## 📁 Struktur Proyek

```
wine_quality/
├── data/
│   └── WineQT.csv          # Dataset
├── static/
│   ├── css/
│   │   └── style.css       # Custom CSS (Premium Dark Theme)
│   ├── js/
│   │   └── main.js         # JavaScript interaktif
│   └── img/
│       └── training_results.png  # Visualisasi hasil training
├── templates/
│   └── index.html          # Halaman utama (Bootstrap 5)
├── app.py                  # Flask web application
├── train_model.py          # Script pelatihan model
├── model.keras             # Model yang sudah dilatih
├── scaler.pkl              # StandardScaler yang sudah difit
├── config.py               # Konfigurasi aplikasi
├── requirements.txt        # Dependensi Python
└── README.md
```

---

## 🚀 Cara Menjalankan Secara Lokal

### 1. Clone Repository
```bash
git clone https://github.com/rizkihsn/wine_quality.git
cd wine_quality
```

### 2. Buat Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

### 4. (Opsional) Latih Ulang Model
```bash
python train_model.py
```

### 5. Jalankan Aplikasi
```bash
python app.py
```

Buka browser dan akses: **http://localhost:5000**

---

## 📖 Cara Menggunakan Aplikasi

1. Masukkan nilai 11 parameter fisikokimia wine pada formulir
2. Klik tombol **"Prediksi Kualitas"**
3. Hasil prediksi (GOOD/BAD) akan muncul beserta **Tingkat Kepercayaan Model**

---

## 📚 Dataset

- **Sumber:** [Wine Quality Dataset - Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
- **Jumlah Sampel:** 1.143
- **Jumlah Fitur:** 11 fitur fisikokimia
- **Target:** Kualitas wine (binary: GOOD/BAD)

---

## 👤 Author

**Rizki Hasan** — Tugas 7 Backpropagation

---

## 📄 Lisensi

MIT License
