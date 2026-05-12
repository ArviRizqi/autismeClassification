# 🧠 Autism Classification App

Streamlit app untuk klasifikasi **Autistic / Non-Autistic** dari gambar wajah
menggunakan **MobileViTV2 + Fusion Backbone Classifier + TTA**.

---

## Struktur Proyek

```
autism_classifier/
│
├── app.py                        # Entry point — jalankan file ini
│
├── requirements.txt
│
├── assets/
│   └── style.css                 # Custom CSS (opsional)
│
├── config/
│   ├── __init__.py
│   └── settings.py               # Semua konstanta & konfigurasi
│
├── utils/
│   ├── __init__.py
│   ├── model_architecture.py     # Definisi SELayer, FusionBlock, Classifier
│   ├── model_loader.py           # Load MTCNN + model (st.cache_resource)
│   ├── transforms.py             # Albumentations pipelines (standard + TTA)
│   └── predictor.py              # Face detection & inference logic (no UI)
│
└── components/
    ├── __init__.py
    ├── sidebar.py                # Sidebar widget → returns use_tta
    ├── tabs.py                   # Upload & Camera tab layout
    ├── prediction_result.py      # Hasil prediksi + interpretasi
    └── instructions.py           # Panduan penggunaan & quality tips
```

---

## Instalasi & Menjalankan

```bash
# 1. Clone / salin folder ini
cd autism_classifier

# 2. Buat virtual environment (opsional tapi disarankan)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan
streamlit run app.py
```

---

## Cara Menambahkan Fitur Baru

| Tujuan | File yang diedit |
|---|---|
| Ganti model / backbone | `config/settings.py` |
| Tambah augmentasi TTA baru | `utils/transforms.py` |
| Ubah logika prediksi | `utils/predictor.py` |
| Tambah widget sidebar | `components/sidebar.py` |
| Ubah tampilan hasil | `components/prediction_result.py` |
| Tambah tab baru | `components/tabs.py` |

---

## Disclaimer

Aplikasi ini adalah **demo penelitian** dan **TIDAK** dapat digunakan sebagai
diagnosis medis. Konsultasikan dengan profesional kesehatan untuk diagnosis
yang akurat.
