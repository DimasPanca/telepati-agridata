# telepati-agridata

Model object detection penyakit padi (11 kelas) untuk **AI AgriData Intelligence Race — TELEPATI 8.0** oleh tim **The Power of Kepepet**, dilatih dari nol pada dataset resmi Agritech Data.

> Status: **dalam pengembangan**. Bagian bertanda `TBD` dilengkapi sebelum submission (batas 25 Oktober 2026, 23:59 WIB).

| | |
|---|---|
| Tim | The Power of Kepepet — Universitas Sultan Ageng Tirtayasa |
| Anggota | Dimas (Ketua), Rahman, Maman |
| Dosen Pembimbing | *(nama)* |
| Notebook | [`notebooks/agridata_pipeline.ipynb`](notebooks/agridata_pipeline.ipynb) |
| Model weights | `TBD` (direct link GitHub Release) |
| Dokumen tim | [`docs/`](docs/) — mulai dari `docs/README_TIM.txt` |

---

## 1. Ringkasan

- **Tugas:** object detection 11 kelas kondisi tanaman padi (10 penyakit + Healthy) pada citra 640×640.
- **Data:** dataset resmi Agritech Data. 21 kategori COCO dipetakan ke 11 kelas canonical di dalam notebook.
- **Model:** keluarga YOLO (Ultralytics), dibangun dari file `.yaml`, **dilatih dari nol tanpa pretrained**.
- **Evaluasi:** F1-Score makro pada split test resmi.
- **Platform:** Google Colab, GPU T4.

---

## 2. How to Run

### 2.1 Reproduksi skor (inference-saja, untuk audit)

1. Buka notebook di Google Colab: `TBD (link Open in Colab)`.
2. Runtime → Change runtime type → **T4 GPU**.
3. Siapkan zip dataset resmi apa adanya (format COCO, split `train/`, `valid/`, `test/`; folder pembungkus dan file `__MACOSX` ditangani otomatis), lalu isi path-nya di `DATA_ZIP` pada sel konfigurasi (bagian 0). `DATA_ZIP` juga boleh berisi folder hasil ekstrak, misalnya dataset Kaggle yang diekstrak otomatis.
4. Set `MODE = "inference_only"`. `WEIGHTS_URL`, `WEIGHTS_SHA256`, dan `E2_CONF` sudah terisi.
5. Runtime → Run all. Notebook akan:
   - memasang dependency dari `requirements.txt`,
   - mengunduh weights dari direct link Release dan memeriksa SHA-256,
   - memetakan label test ke 11 kelas,
   - menjalankan inference pada split test resmi dan mencetak F1 makro.

Perkiraan waktu: `TBD` menit di Colab T4.

### 2.2 Reproduksi penuh (training dari data mentah)

1. Langkah 1–3 seperti di atas.
2. Set `MODE = "full"` dan isi `RUN_ID` dengan salah satu konfigurasi di tabel `RUNS` (bagian 7 notebook).
3. Runtime → Run all. Notebook menjalankan pemetaan kelas, pembersihan, konversi, training, dan evaluasi valid. Evaluasi test dijalankan sekali dengan `RUN_FINAL_TEST = True` setelah model final dipilih.

---

## 3. Dependencies

Versi dipin di [`requirements.txt`](requirements.txt).

| Paket | Versi |
|---|---|
| ultralytics | 8.4.184 |
| Python, torch, numpy, pandas | Versi bawaan Google Colab (dicetak di bagian 1 notebook; versi yang dipakai saat audit dicatat di sini) |

---

## 4. Arsitektur model

`TBD` — diisi setelah model final dipilih (YOLO11s atau YOLO26s dari `.yaml`, tanpa pretrained), beserta jumlah parameter dan alasan pemilihan.

---

## 5. Hyperparameter

| Parameter | Nilai |
|---|---|
| Ukuran input (`imgsz`) | 640 |
| Epoch | `TBD` |
| Batch | `TBD` |
| Optimizer | `TBD` |
| Seed | 42 |
| Deterministic | True |
| Pretrained | False |
| Augmentasi | `TBD` |

---

## 6. Model weights

| Versi | Direct link | SHA-256 |
|---|---|---|
| `TBD` | `TBD` | `TBD` |

---

## 7. Hasil

Skor mandiri diambil dari jalur inference-saja di Colab T4.

| Split | F1 makro | mAP@50 |
|---|---|---|
| valid | `TBD` | `TBD` |
| test | `TBD` | `TBD` |

---

## 8. Kepatuhan aturan

- Training hanya memakai split train. Valid untuk seleksi model. Test dievaluasi satu kali di akhir.
- Tidak ada pretrained model eksternal.
- Seluruh pemrosesan data dimulai dari dataset mentah di dalam notebook.
- Random seed tetap di setiap tahap.
- Dataset resmi tidak disertakan di repositori ini.
