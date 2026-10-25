# Rice Disease Detection — AI AgriData Intelligence Race (TELEPATI 8.0)

Repositori resmi tim untuk kompetisi **AI AgriData Intelligence Race — TELEPATI 8.0** (*"AgriTech: Growing the Golden Future"*, HIMATEL POLBAN). Sistem ini membangun **satu model object detection (`YOLO11s`)** yang dilatih **dari nol (`pretrained=False`)** pada resolusi `640×640` untuk mendeteksi **11 kelas kondisi daun/tanaman padi** (10 penyakit + `Healthy`) secara akurat, deterministik, dan siap diaudit pada **Google Colab GPU Tesla T4**.

---

## Daftar Isi
1. [Ringkasan Hasil & Skor Mandiri](#1-ringkasan-hasil--skor-mandiri)
2. [Tautan Model Weights (GitHub Releases)](#2-tautan-model-weights-github-releases)
3. [Cara Menjalankan (How to Run)](#3-cara-menjalankan-how-to-run)
4. [Daftar Dependencies & Lingkungan](#4-daftar-dependencies--lingkungan)
5. [Arsitektur Model & Metodologi Pipeline](#5-arsitektur-model--metodologi-pipeline)
6. [Hyperparameter Pelatihan & Inferensi](#6-hyperparameter-pelatihan--inferensi)
7. [Hasil Eksperimen, Fine-Tuning, dan Uji Stabilitas Audit (±2%)](#7-hasil-eksperimen-fine-tuning-dan-uji-stabilitas-audit-2)
8. [Struktur Repositori](#8-struktur-repositori)

---

## 1. Ringkasan Hasil & Skor Mandiri

Seluruh pemilihan model, pencarian hyperparameter, dan kalibrasi dilakukan **murni pada split `valid` resmi** (`2.106` gambar, `4.888` anotasi) tanpa menyentuh split `test` (`R-20`).

| Tahap / Kandidat Model | File Bobot (`.pt`) | Setting Evaluasi (`iou` / `TTA`) | **F1 Makro (`E1`) Valid** | **F1 Tetap (`E2 @0.24`) Valid** | **`mAP@50` Valid** | **Skor Default Ultralytics (`iou=0.70, TTA=False`)** | **Selisih Audit Maksimal (`≤ 2%`)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Awal (`Run A`, 150 Epoch)** | `best.pt` | `0.70` / `False` | `0.68424` | `0.68351` | `0.69272` | `0.68424` | Baseline |
| **Stage-2 Refinement (`Run A+`, 25 Epoch)** | `best.pt` | `0.38` / `True` | `0.71629` | `0.71629` | `0.69425` | `0.69385` | `+2,24%` |
| **Model Soup (`SWA` 3-Checkpoint)** | `swa_50best_25ep10_25ep120.pt` | `0.38` / `True` | `0.71677` | `0.71577` | `0.69745` | `0.69440` | `+2,24%` |
| **Kandidat Puncak Murni (`LBC`)** | `best_murni_cal_a55.pt` | `0.38` / `True` | **`0.72118`** | `0.71907` | `0.69406` | `~0.69750` | `+2,37%` |
| **Kandidat Tunggal Audit-Safe** | `best_murni_audit_safe.pt` | `0.42` / `True` | **`0.72046`** | `0.71906` | `0.69532` | **`0.70622`** | **`+1,42%` (AMAN)** |
| **Model Final Utama (`Audit-Safe`)** | **`best_audit_safe.pt`** | `0.38` / `True` | **`0.72053`** | **`0.71973`** | `0.69625` | **`0.70745`** | **`+1,31%` (AMAN)** |
| **Model Final (`Opsi IoU 0.45`)** | **`best_audit_safe.pt`** | `0.45` / `True` | **`0.71935`** | `0.71841` | **`0.69929`** | **`0.70745`** | **`+1,19%` (AMAN)** |
| **Model Final (`Opsi Tanpa TTA`)** | **`best_audit_safe.pt`** | `0.45` / `False` | **`0.71754`** | `0.71634` | `0.69071` | **`0.70745`** | **`+1,01%` (AMAN)** |

> **Skor Mandiri Split `test` Resmi (Diisi dari Latihan Audit Colab T4 — Bagian 11 / `S-11`)**:
> - **F1-Score Makro (`E1`)**: `*(diisi setelah S-11 dijalankan)*`
> - **F1-Score Tetap (`E2` pada `conf = 0.24`)**: `*(diisi setelah S-11 dijalankan)*`
> - **`mAP@50`**: `*(diisi setelah S-11 dijalankan)*`
> - **Waktu Inferensi Total (GPU Tesla T4)**: `~33 – 68 detik` (batas resmi: `< 30 menit`)

---

## 2. Tautan Model Weights (GitHub Releases)

Sesuai ketentuan **`R-13`**, bobot model akhir tidak disimpan di dalam pohon commit Git maupun Google Drive, melainkan dirilis melalui **GitHub Releases** dengan tautan unduh langsung (*direct download link* tanpa login):

| Artefak | Nama Berkas | Direct Link GitHub Release | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **Model Weights Utama (`v1.0`)** | `best_audit_safe.pt` / `telepati_agridata_v1.0.pt` | `https://github.com/DimasPanca/telepati-agridata/releases/download/v1.0/telepati_agridata_v1.0.pt` | `*(isi SHA-256 setelah rilis)*` |
| **Dataset Mentah Resmi** | `telepati-agridata-raw.zip` | *(Dataset resmi panitia — tidak diunggah ke repo publik sesuai `NFR-05`)* | `797e47749f91f4403e778dd0150716da1cd94cddbecee2ec56cf5545c50cec50` |

---

## 3. Cara Menjalankan (How to Run)

Seluruh alur dari ekstraksi dataset mentah (`.zip`), pemetaan 21 kategori ke 11 kelas canonical, pembersihan anotasi, pelatihan dari nol, seleksi validasi, hingga inferensi audit terpusat di dalam **satu Jupyter Notebook**: [`notebooks/agridata_pipeline.ipynb`](notebooks/agridata_pipeline.ipynb).

### A. Mode Audit / Inference-Saja (`MODE = "inference_only"`, Waktu `< 2 Menit` di Colab T4)
Mode ini digunakan oleh Juri untuk mereproduksi skor akhir menggunakan bobot rilis tanpa melatih ulang model (`R-18`):

1. Buka [`notebooks/agridata_pipeline.ipynb`](notebooks/agridata_pipeline.ipynb) di **Google Colab** dengan accelerator **GPU T4** (`Runtime -> Change runtime type -> T4 GPU`).
2. Pada **Bagian `0. Konfigurasi`**, pastikan sakelar diatur ke:
   ```python
   MODE = "inference_only"
   DATA_ZIP = "/content/drive/MyDrive/telepati/data/telepati-agridata-raw.zip"  # sesuaikan lokasi zip dataset resmi
   ```
3. Jalankan berurutan:
   - **Bagian `0. Konfigurasi`**
   - **Bagian `1. Lingkungan`** (memasang `ultralytics==8.4.184` dan mengunci `SEED = 42` secara deterministik)
   - **Bagian `2. Dataset mentah`** (verifikasi checksum & ekstraksi zip mentah ke `/content/work/raw`)
   - **Bagian `3. Pemetaan 21 kategori → 11 kelas canonical`**
   - **Bagian `11. Inference-saja (Audit)`** (mengunduh bobot `.pt` dari *direct link* GitHub Release, memverifikasi SHA-256, mengeksekusi inferensi pada split `test`, dan mencetak `E1`, `E2`, `mAP@50`, F1 per kelas, serta waktu eksekusi).

### B. Mode Reproduksi Penuh (`MODE = "full"`)
Untuk mereproduksi seluruh alur mulai dari konversi data, pelatihan Stage-1 (`150 epoch`), evaluasi checkpoint, pelatihan Stage-2 (`25 epoch`), *Model Soup (SWA)*, *Logit Bias Calibration (LBC)*, hingga uji stabilitas:
1. Atur `MODE = "full"` pada **Bagian `0. Konfigurasi`**.
2. Jalankan seluruh sel secara berurutan dari **Bagian `0`** hingga **Bagian `10`**.

---

## 4. Daftar Dependencies & Lingkungan

Eksperimen dan audit dijalankan pada lingkungan **Google Colab (GPU NVIDIA Tesla T4, 15 GB VRAM)** dengan versi pustaka yang dikunci di [`requirements.txt`](requirements.txt):

```text
python==3.13.16
torch==2.11.0+cu130
torchvision==0.26.0+cu130
ultralytics==8.4.184
numpy>=2.0.0
pandas>=2.2.0
PyYAML>=6.0.2
```

Pemasangan otomatis sudah tercakup di sel pertama **Bagian `1. Lingkungan`** di dalam notebook:
```bash
pip install -q ultralytics==8.4.184
```

---

## 5. Arsitektur Model & Metodologi Pipeline

### 5.1 Arsitektur Detektor (`YOLO11s` From Scratch — `R-10`)
- **Keluarga Arsitektur**: Ultralytics `YOLO11s` yang dibangun murni dari definisi konfigurasi `yolo11s.yaml` dengan **`pretrained=False`** (100 layers, `9.417.057` parameter, `21.4 GFLOPs`).
- **Alasan Pemilihan**: Dibandingkan detektor berbasis Transformer (RT-DETR) yang membutuhkan *pretrained backbone* berskala besar agar konvergen, arsitektur CNN multi-skala (`C3k2` + `C2PSA` + *decoupled head* `Detect` pada stride 8, 16, dan 32) mampu belajar dari nol secara stabil pada 10.133 gambar latih.

### 5.2 Pemetaan 21 Kategori Raw $\rightarrow$ 11 Kelas Canonical (`R-09`)
Dataset COCO mentah memiliki 21 entri kategori (18 variasi penulisan nama penyakit dan 3 *supercategory* kosong: `Leaf-blight`, `Rice-Leaf-Diseasee`, `paddy`). Seluruh anotasi dipetakan secara deterministik ke **11 kelas canonical** sesuai urutan alfabet resmi:
1. `Bacterial leaf blight`
2. `Bacterial panicle blight`
3. `Blast` *(menggabungkan `Blast`, `Leaf blast`, `Infected Blast`)*
4. `Brown spot` *(menggabungkan `Brown spot`, `BrownSpot`)*
5. `False smut` *(menggabungkan `False-Smut`)*
6. `Healthy` *(menggabungkan `Healthy`, `healthy`, `Healthy Rice Leaf`, `Healthy Rice beads`)*
7. `Leaf roller` *(menggabungkan `Leaf-roller`)*
8. `Leaf scald` *(menggabungkan `Leaf scald`, `Leaf Scald`)*
9. `Narrow brown`
10. `Sheath blight` *(menggabungkan `Sheath Blight`)*
11. `Tungro` *(menggabungkan `Rice-Tungro`)*

### 5.3 Pembersihan Anotasi (`R-07`)
Pembersihan **hanya diterapkan pada split `train`** (`20.163` box $\rightarrow$ `20.134` box), sedangkan split `valid` dan `test` dipertahankan apa adanya agar evaluasi 100% objektif:
- Memotong (*clipping*) koordinat *bounding box* yang melewati batas dimensi gambar.
- Membuang `22` *bounding box* rusak yang memiliki lebar atau tinggi `< 2.0 piksel`.
- Menghapus `7` *bounding box* duplikat identik pada gambar yang sama.

### 5.4 Strategi Optimasi Bertahap (`Bagian 8` – `Bagian 9f`)
1. **Seleksi Checkpoint Berbasis F1 Makro (`Bagian 8`)**: Mengevaluasi seluruh checkpoint berkala pada split `valid` menggunakan metrik `E1` dan `E2` (menemukan `epoch120.pt` lebih unggul dibanding `best.pt` bawaan yang berbasis mAP).
2. **Stage-2 Full-Resolution No-Mosaic Refinement (`Bagian 9`)**: Melanjutkan pelatihan dari `epoch120.pt` selama 25 epoch dengan mematikan `mosaic=0.0` dan `mixup=0.0` pada resolusi asli `640×640` (`lr0=0.0005`, `cls=1.0`, `scale=0.25`) agar detektor mempelajari morfologi lesi kecil (`Brown spot`, `Blast`) tanpa distorsi batas *mosaic*.
3. **Cross-Stage Model Soup / SWA (`Bagian 9c`)**: Merata-ratakan parameter bobot dari `50% A_plus/best.pt + 25% A_plus/epoch10.pt + 25% RunA/epoch120.pt` menjadi satu model tunggal untuk meredam varians lokalisasi.
4. **Logit Bias Calibration Lintas-Ambang (`Bagian 9d` – `9f`)**: Menyelaraskan vektor `bias` pada layer `1×1 Conv2d` terakhir di *classification head* (`Detect.cv3`) menggunakan pergeseran logit terkalibrasi ($\Delta b_c = \alpha \cdot [\text{logit}(t_{\text{target}}) - \text{logit}(p_c)]$) sehingga puncak kurva F1 dari ke-11 kelas berimpit pada ambang *confidence* yang sama baik pada evaluasi standar (`iou=0.70, TTA=False`) maupun evaluasi optimal (`iou=0.38–0.45`).

---

## 6. Hyperparameter Pelatihan & Inferensi

### 6.1 Hyperparameter Pelatihan (`Stage-1` & `Stage-2`)

| Parameter | Stage-1 (`A_yolo11s_s42`) | Stage-2 (`A_plus_nomosaic_s42`) |
| :--- | :--- | :--- |
| **Inisialisasi Model** | `yolo11s.yaml` (`pretrained=False`) | `A_yolo11s_s42/weights/epoch120.pt` |
| **Split Data Latih** | `train` (`10.133` gambar) | `train` (`10.133` gambar) |
| **Epochs** | `150` (`save_period=10`) | `25` (`save_period=5`) |
| **Image Size (`imgsz`) & Batch** | `640`, `batch=16` | `640`, `batch=16` |
| **Optimizer & Learning Rate** | `auto` (`SGD`), `lr0=0.01`, `lrf=0.01` | `SGD`, `lr0=0.0005`, `lrf=0.1`, `cos_lr=True` |
| **Warmup** | `warmup_epochs=3.0` | `warmup_epochs=0.0`, `warmup_bias_lr=0.0` |
| **Loss Gains (`box` / `cls` / `dfl`)** | `7.5` / `0.5` / `1.5` | `7.5` / `1.0` / `1.5` |
| **Augmentasi Spasial** | `mosaic=1.0` (`close_mosaic=10`), `mixup=0.1`, `scale=0.5`, `fliplr=0.5`, `flipud=0.5` | `mosaic=0.0`, `mixup=0.0`, `scale=0.25`, `fliplr=0.5`, `flipud=0.5` |
| **Seed & Determinisme** | `seed=42`, `deterministic=True`, `amp=True` | `seed=42`, `deterministic=True`, `amp=True` |

### 6.2 Parameter Inferensi & Evaluasi Terkunci (`S-10` & `S-11`)

| Parameter | Nilai Terkunci | Keterangan |
| :--- | :--- | :--- |
| **`imgsz`** | `640` | Resolusi standar dataset |
| **`batch`** | `16` | Ukuran batch evaluasi di GPU T4 |
| **`iou` (NMS Threshold)** | `0.38` *(atau `0.45`)* | Ambang NMS optimal untuk lesi daun yang bergerombol *(tetap stabil `0.7075` pada default `0.70`)* |
| **`augment` (TTA)** | `True` *(atau `False`)* | *Test-Time Augmentation* (`E1 = 0.7205` dengan TTA; `E1 = 0.7175` tanpa TTA) |
| **`E2_CONF` (Fixed Confidence)** | `0.24` | Ambang *confidence* tetap untuk perhitungan metrik `E2` |
| **`half`** | `False` (`FP32`) | Konsisten antara `FP32` (`0.71935`) dan `FP16` (`0.71944`) |
| **`SEED`** | `42` | Deterministik (`selisih antar-run = 0.00000`) |

---

## 7. Hasil Eksperimen, Fine-Tuning, dan Uji Stabilitas Audit (±2%)

### 7.1 Tabel Uji Stabilitas `best_audit_safe.pt` Lintas Skenario Evaluasi Juri (Split `valid`)

| Skenario Evaluasi | `iou` | `TTA` | `half` | **`E1_mf1`** | **`E2_f1`** | **`mAP@50`** | Waktu (`T4`) | **Selisih vs `S-11 (0.45)`** | **Selisih vs `NoTTA`** | **Status Audit (`≤ 2%`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1. S-11 Peak (iou=0.38, TTA=True, FP32)` | `0.38` | `True` | `False` | **`0.72053`** | **`0.71973`** | `0.69625` | `68.1s` | `+0.00118` | `+0.00299` | **AMAN (`<=2%`)** |
| `2. S-11 Optimal (iou=0.45, TTA=True, FP32)` | `0.45` | `True` | `False` | **`0.71935`** | `0.71841` | `0.69929` | `68.1s` | `0.00000` | `+0.00181` | **AMAN (`<=2%`)** |
| `3. S-11 + FP16 (iou=0.45, TTA=True, FP16)` | `0.45` | `True` | `True` | **`0.71944`** | `0.71862` | `0.69913` | `34.2s` | `+0.00009` | `+0.00190` | **AMAN (`<=2%`)** |
| `4. S-11 IoU 0.50 (iou=0.50, TTA=True, FP32)` | `0.50` | `True` | `False` | **`0.71794`** | `0.71713` | **`0.70157`** | `68.1s` | `-0.00141` | `+0.00040` | **AMAN (`<=2%`)** |
| `5. Tanpa TTA IoU 0.45 (iou=0.45, TTA=False)` | `0.45` | `False` | `False` | **`0.71754`** | `0.71634` | `0.69071` | `33.4s` | `-0.00181` | `0.00000` | **AMAN (`<=2%`)** |
| `6. Tanpa TTA IoU 0.50 (iou=0.50, TTA=False)` | `0.50` | `False` | `False` | **`0.71671`** | `0.71533` | `0.69470` | `33.5s` | `-0.00264` | `-0.00083` | **AMAN (`<=2%`)** |
| `7. Default Ultralytics (iou=0.70, TTA=False)` | `0.70` | `False` | `False` | **`0.70745`** | `0.70612` | `0.68852` | `33.5s` | **`-0.01190`** | **`-0.01009`** | **AMAN (`<=2%`)** |

### 7.2 Rincian F1-Score Per Kelas (11 Kelas Canonical pada Split `valid`)

| No | Kelas Penyakit | Jumlah Box (`valid`) | Baseline (`A_yolo11s_s42`) | Stage-2 (`A_plus_nomosaic`) | Puncak Murni (`best_murni_cal_a55.pt`) | **Final Audit-Safe (`best_audit_safe.pt`)** | Kenaikan vs Baseline |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `Bacterial leaf blight` | `125` | `0.5182` | `0.6006` | `0.5997` | **`0.5688`** *(Murni-Safe: `0.6005`)* | **`+0.0506`** |
| 2 | `Bacterial panicle blight` | `45` | `0.7675` | `0.7569` | `0.7665` | **`0.7503`** *(Murni-Safe: `0.7618`)* | `-0.0172` |
| 3 | `Blast` | `947` | `0.5601` | `0.5907` | `0.5933` | **`0.5923`** | **`+0.0322`** |
| 4 | `Brown spot` | `1.461` | `0.3712` | `0.3980` | `0.4181` | **`0.4175`** *(IoU 0.45: `0.4206`)* | **`+0.0463`** |
| 5 | `False smut` | `86` | `0.8781` | `0.9129` | `0.9238` | **`0.9351`** | **`+0.0570`** |
| 6 | `Healthy` | `424` | `0.8827` | `0.8974` | `0.8994` | **`0.9011`** | **`+0.0184`** |
| 7 | `Leaf roller` | `87` | `0.8362` | `0.8662` | `0.8729` | **`0.8788`** | **`+0.0426`** |
| 8 | `Leaf scald` | `390` | `0.5322` | `0.5392` | `0.5389` | **`0.5485`** | **`+0.0163`** |
| 9 | `Narrow brown` | `75` | `0.9300` | `0.9914` | `0.9905` | **`0.9923`** | **`+0.0623`** |
| 10 | `Sheath blight` | `488` | `0.6195` | `0.6679` | `0.6691` | **`0.6748`** | **`+0.0553`** |
| 11 | `Tungro` | `760` | `0.6307` | `0.6579` | `0.6608` | **`0.6664`** | **`+0.0357`** |
| — | **Rata-rata Makro (`E1`)** | **`4.888`** | **`0.68424`** | **`0.71629`** | **`0.72118`** | **`0.72053`** | **`+0.03629` (`+3,63%`)** |

---

## 8. Struktur Repositori

```text
telepati-agridata/
├── README.md                                       # Dokumentasi utama proyek (R-15)
├── requirements.txt                                # Daftar versi library yang dipin (NFR-03)
├── .gitignore                                      # Mencegah dataset, bobot .pt, dan kredensial masuk repo
├── notebooks/
│   └── agridata_pipeline.ipynb                     # Notebook end-to-end (Data Mentah -> Training -> Evaluasi -> Audit)
├── results/
│   ├── A_yolo11s_s42/                              # Log training & konfigurasi Stage-1 Baseline (150 epoch)
│   │   ├── results.csv
│   │   └── args.yaml
│   ├── A_plus_nomosaic_s42/                        # Log training & konfigurasi Stage-2 Refinement (25 epoch)
│   │   ├── results.csv
│   │   └── args.yaml
│   ├── A_yolo11s_s42_valid_checkpoints.csv         # Evaluasi checkpoint berkala Stage-1 pada split valid
│   ├── A_yolo11s_s42_postproc_tuning.csv           # Kalibrasi NMS IoU & TTA Stage-1 pada split valid
│   ├── A_plus_nomosaic_s42_valid_checkpoints.csv   # Evaluasi checkpoint berkala Stage-2 pada split valid
│   ├── checkpoint_eval_valid.csv                   # Tabel master evaluasi seluruh eksperimen pada split valid
│   ├── checkpoint_eval_valid_audit_safe.csv        # Peringkat kandidat final Audit-Safe beserta F1 11 kelas
│   └── A_plus_audit_stability_valid.csv            # Bukti uji stabilitas lintas skenario audit juri (<= 2%)
└── docs/                                           # Dokumen perancangan dan tata kelola tim
```
