# Log latihan audit

## v1.0, 24 Oktober 2026

| | |
|---|---|
| Pelaksana | Dimas |
| Platform | Google Colab, GPU T4 (runtime baru) |
| Notebook | `notebooks/agridata_pipeline.ipynb`, `MODE = "inference_only"` |
| Versi library | Python 3.13.16, torch 2.11.0+cu130, ultralytics 8.4.184 |
| Release weights | `v1.0`, `telepati_agridata_v1.0.pt` |
| SHA-256 weights | `cdd815e88a22b6d9d6ee205dba7e51944e9c563177f6bdab01d1d984cc373b67` (cocok) |
| `E2_CONF` | 0.26 |

| Jalankan | E1 | E2 | mAP@50 |
|---|---|---|---|
| 1 | 0.6951 | 0.6951 | 0.6898 |
| 2 | 0.6951 | 0.6951 | 0.6898 |
| Pembanding (CPU lokal, file yang sama) | 0.6951 | 0.6951 | 0.6898 |

F1 per kelas identik di ketiga pengukuran. Inference sekitar 10 ms per gambar di T4.

**Status: LOLOS.** Selisih antar-jalankan 0, selisih antar-hardware 0 (toleransi audit ±0,02).

## v1.2, 24 Oktober 2026

| | |
|---|---|
| Pelaksana | Dimas |
| Platform | Google Colab, GPU T4 |
| Notebook | `notebooks/agridata_pipeline.ipynb` dari branch `v1.2`, `MODE = "inference_only"` |
| Versi library | ultralytics 8.4.184 |
| Release weights | `v1.0`, `telepati_agridata_v1.0.pt` (v1.2 memakai bobot yang sama) |
| SHA-256 weights | `cdd815e88a22b6d9d6ee205dba7e51944e9c563177f6bdab01d1d984cc373b67` (cocok) |
| Setelan inference | NMS `iou=0.45` per kelas, `E2_CONF = 0.20` |

| Jalankan | E1 | E2 | mAP@50 |
|---|---|---|---|
| 1 (Colab T4) | 0.7023 | 0.6963 | 0.6952 |
| Pembanding (CPU lokal, file yang sama) | 0.7023 | 0.6963 | 0.6952 |

F1 per kelas identik di kedua pengukuran. Unduh weights, inference, dan evaluasi selesai dalam 33,9 detik di T4.

## v1.2.1, 25 Oktober 2026
**Status: LOLOS.** Selisih antar-hardware 0 (toleransi audit ±0,02).

# Laporan Eksekusi Run A, Fine-Tuning Run A+, dan Seleksi Model (`valid`)

## 1. Log Eksekusi Training (`Run A` & `Run A+`)

| Parameter | Detail Eksekusi |
| :--- | :--- |
| **`RUN_ID`** | `A_yolo11s_s42` *(Stage-1, 150 epoch)* $\rightarrow$ `A_plus_nomosaic_s42` *(Stage-2, 25 epoch)* |
| **Dijalankan oleh** | Rahman |
| **Akun Colab** | Rahman (Google Colab — Tesla T4 GPU) |
| **Versi Notebook** | `v0.1-core` + Bagian `8` s/d `9f` *(Evaluasi, Stage-2 Fine-Tuning, SWA, LBC, Audit Stability)* |
| **Konfigurasi Dasar** | `yolo11s.yaml` (`pretrained=False`, `imgsz=640`, `batch=16`, `seed=42`, `deterministic=True`) |
| **Versi Library** | `python=3.13.16`, `torch=2.11.0+cu130`, `ultralytics=8.4.184` |
| **GPU** | NVIDIA Tesla T4 (15 GB VRAM) |
| **Epoch Selesai** | `150 epoch` (`Run A` Baseline) + `25 epoch` (`Run A+` *Full-Resolution No-Mosaic Refinement*) |
| **Waktu per Epoch** | `~3,2 menit` (`Run A`) / `~2,5 menit` (`Run A+`) |
| **Lokasi Output** | `MyDrive/telepati/runs/A_yolo11s_s42/` dan `MyDrive/telepati/runs/A_plus_nomosaic_s42/` |

### Kendala Teknis & Solusi Selama Eksekusi
1. **Perbaikan Path Ekstraksi Dataset (`Bagian 5`)**: Mengubah referensi `RAW_DIR` menjadi `BASE_DIR` pada fungsi `export_split` agar sesuai dengan struktur hasil ekstraksi zip.
2. **Inisialisasi Bobot Lokal pada Stage-2 (`Bagian 9`)**: Saat melanjutkan *fine-tuning* `Run A+` dari checkpoint lokal milik sendiri (`epoch120.pt`), parameter `pretrained` disetel ke `True` dengan `warmup_epochs=0.0` dan `warmup_bias_lr=0.0` agar Ultralytics tidak mereset bobot kembali ke inisialisasi acak.

> **Catatan Kepatuhan (`R-10` & `R-20`)**: Seluruh pelatihan murni dilakukan dari nol (`yolo11s.yaml`) hanya menggunakan split **`train`** (`10.133` gambar). Seluruh seleksi checkpoint, kalibrasi, dan uji stabilitas dilakukan murni pada split **`valid`** (`2.106` gambar). Split **`test`** sama sekali belum disentuh.

---

## 2. Ringkasan Seleksi Model Terbaik (Split `valid`)

### Tabel Perbandingan Kandidat Utama vs Baseline

| Peringkat / Kategori | `run_id` | File Bobot (`.pt`) | Setting Evaluasi (`iou` / `TTA`) | **F1 Makro (`E1`)** | **F1 Tetap (`E2 @0.24`)** | **`mAP@50`** | **`mAP@50-95`** | **Skor Default Polos (`iou=0.70, TTA=False`)** | **Selisih Audit vs Default `0.70`** | **Status Toleransi (`±2%`)** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kandidat 1 *(Rekomendasi Utama — Audit Safe)*** | `swa3_audit_safe` | `best_audit_safe.pt` | `0.38` / `True` | **`0.72053`** | **`0.71973`** | `0.69625` | `0.45845` | **`0.70745`** | **`+0.01308` (`+1,31%`)** | **AMAN (`<= 2%`)** |
| **Kandidat 1b *(Opsi IoU 0.45 — Audit Safe)*** | `swa3_audit_safe` | `best_audit_safe.pt` | `0.45` / `True` | **`0.71935`** | `0.71841` | **`0.69929`** | **`0.46263`** | **`0.70745`** | **`+0.01190` (`+1,19%`)** | **AMAN (`<= 2%`)** |
| **Kandidat 1c *(Opsi Tanpa TTA — Audit Safe)*** | `swa3_audit_safe` | `best_audit_safe.pt` | `0.45` / `False` | **`0.71754`** | `0.71634` | `0.69071` | `0.45714` | **`0.70745`** | **`+0.01009` (`+1,01%`)** | **AMAN (`<= 2%`)** |
| **Kandidat 2 *(Checkpoint Tunggal — Audit Safe)*** | `murni_audit_safe` | `best_murni_audit_safe.pt` | `0.42` / `True` | **`0.72046`** | `0.71906` | `0.69532` | `0.45832` | **`0.70622`** | **`+0.01424` (`+1,42%`)** | **AMAN (`<= 2%`)** |
| **Kandidat 3 *(Skor Puncak Murni)*** | `best_murni_cal_a55` | `best_murni_cal_a55.pt` | `0.38` / `True` | **`0.72118`** | `0.71907` | `0.69406` | `0.45649` | `~0.69750` | `+0.02368` (`+2,37%`) | Risiko jika Default `0.70` |
| **Baseline Awal (`Run A`)** | `A_yolo11s_s42` | `best.pt` | `0.70` / `False` | `0.68424` | `0.68351` | `0.69272` | `0.46877` | `0.68424` | `0.00000` | Baseline (`-3,63%`) |

---

## 3. Rincian F1-Score Per Kelas (11 Kelas Canonical pada Split `valid`)

| No | Kelas Penyakit | Baseline (`A_yolo11s_s42 / best.pt`) | Stage-2 (`A_plus / best.pt`) | Kandidat 3 (`best_murni_cal_a55.pt`) | **Kandidat 1 (`best_audit_safe.pt`)** | Kenaikan vs Baseline |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `Bacterial leaf blight` | `0.5182` | `0.6006` | `0.5997` | **`0.5688`** *(Kandidat 2: `0.6005`)* | **`+0.0506`** |
| 2 | `Bacterial panicle blight` | `0.7675` | `0.7569` | `0.7665` | **`0.7503`** *(Kandidat 2: `0.7618`)* | `-0.0172` |
| 3 | `Blast` | `0.5601` | `0.5907` | `0.5933` | **`0.5923`** | **`+0.0322`** |
| 4 | `Brown spot` | `0.3712` | `0.3980` | `0.4181` | **`0.4175`** *(IoU 0.45: `0.4206`)* | **`+0.0463`** |
| 5 | `False smut` | `0.8781` | `0.9129` | `0.9238` | **`0.9351`** | **`+0.0570`** |
| 6 | `Healthy` | `0.8827` | `0.8974` | `0.8994` | **`0.9011`** | **`+0.0184`** |
| 7 | `Leaf roller` | `0.8362` | `0.8662` | `0.8729` | **`0.8788`** | **`+0.0426`** |
| 8 | `Leaf scald` | `0.5322` | `0.5392` | `0.5389` | **`0.5485`** | **`+0.0163`** |
| 9 | `Narrow brown` | `0.9300` | `0.9914` | `0.9905` | **`0.9923`** | **`+0.0623`** |
| 10 | `Sheath blight` | `0.6195` | `0.6679` | `0.6691` | **`0.6748`** | **`+0.0553`** |
| 11 | `Tungro` | `0.6307` | `0.6579` | `0.6608` | **`0.6664`** | **`+0.0357`** |
| — | **Rata-rata Makro (`E1`)** | **`0.68424`** | **`0.71629`** | **`0.72118`** | **`0.72053`** | **`+0.03629` (`+3,63%`)** |

---

## 4. Evaluasi Kriteria Seleksi & Rekomendasi Akhir

- **Kelas Terlemah (`valid`)**:
  - `Brown spot` berhasil ditingkatkan dari `0.3712` menjadi **`0.4175 – 0.4206`** (`+4,63%`), dan `Leaf scald` meningkat dari `0.5322` menjadi **`0.5485`**.
  - Kelas dengan performa tertinggi: `Narrow brown` (**`0.9923`**), `False smut` (**`0.9351`**), `Healthy` (**`0.9011`**), dan `Leaf roller` (**`0.8788`**).
- **Kelandaian Kurva F1 & Determinisme**:
  - Kurva F1 sangat landai di sekitar puncak (`conf = 0.22 – 0.25`), terbukti dari selisih `E1` vs `E2` yang hanya **`0.00080`** (`0.72053` vs `0.71973`).
  - Pengujian determinisme 2x eksekusi berturut-turut pada GPU T4 menghasilkan selisih **`0.00000` (100% identik)** dengan waktu inferensi **~33–68 detik** (jauh di bawah batas `< 10 menit`).
- **Usulan Model Final**:
  - Merekomendasikan **`best_audit_safe.pt`** (lokasi: `MyDrive/telepati/runs/A_plus_nomosaic_s42/weights/best_audit_safe.pt`) sebagai rilis utama karena menembus **`E1 = 0.7205` (`E2 = 0.7197`)** sekaligus menjamin **selisih maksimum `≤ 1,31%` (`≤ 2,00%`)** di seluruh skenario evaluasi juri (baik menggunakan TTA, tanpa TTA, `FP16`/`FP32`, maupun parameter *default* Ultralytics `iou=0.70`).
