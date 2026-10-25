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

=== LOG RUN A & RUN A+ ===
RUN_ID          : A_yolo11s_s42 (Stage-1, 150 epoch) -> A_plus_nomosaic_s42 (Stage-2, 25 epoch)
Dijalankan oleh : Rahman
Akun Colab      : Rahman (Tesla T4 GPU)
Versi notebook  : v0.1-core + Bagian 8 s/d 9f (Evaluasi, Stage-2 Fine-Tuning, SWA, LBC, Audit Stability)
Konfigurasi     : yolo11s.yaml (pretrained=False, imgsz=640, batch=16, seed=42, deterministic=True)
Versi library   : python=3.13.16, torch=2.11.0+cu130, ultralytics=8.4.184
GPU             : Tesla T4
Epoch selesai   : 150 epoch (Run A) + 25 epoch (Run A+ No-Mosaic Refinement)
Waktu per epoch : ~3,2 menit (Run A) / ~2,5 menit (Run A+)
Masalah         :
1. Perbaikan bug path RAW_DIR -> BASE_DIR pada export_split (Bagian 5).
2. Saat melanjutkan Stage-2 dari checkpoint lokal (epoch120.pt), parameter pretrained disetel True agar Ultralytics tidak mereset bobot kembali ke acak, serta warmup_epochs=0.0.
Lokasi output   : MyDrive/telepati/runs/A_yolo11s_s42/ dan MyDrive/telepati/runs/A_plus_nomosaic_s42/
Catatan         : Seluruh pelatihan murni memakai split train (10.133 gambar). Seleksi & kalibrasi murni memakai split valid (2.106 gambar). Split test belum disentuh sama sekali.

=== RINGKASAN SELEKSI MODEL (SPLIT VALID) ===
Kandidat 1 (Rekomendasi Utama - Audit Safe) :
  swa3_audit_safe / best_audit_safe.pt (iou=0.38, TTA=True)
  -> E1 = 0.72053 | E2 (@0.24) = 0.71973 | mAP50 = 0.69625
  -> Skor Default Polos Ultralytics (iou=0.70, TTA=False) = 0.70745 (Selisih = +0.01308 / +1,31% <= 2% AMAN)
  -> Skor Tanpa TTA (iou=0.45, TTA=False) = 0.71754 (Selisih = +0.00299 / +0,30% <= 2% AMAN)

Kandidat 2 (Checkpoint Tunggal - Audit Safe) :
  murni_audit_safe / best_murni_audit_safe.pt (iou=0.42, TTA=True)
  -> E1 = 0.72046 | E2 (@0.24) = 0.71906 | mAP50 = 0.69532
  -> Skor Default Polos Ultralytics (iou=0.70, TTA=False) = 0.70622 (Selisih = +0.01424 / +1,42% <= 2% AMAN)

Kandidat 3 (Skor Puncak Murni) :
  best_murni_cal_a55 / best_murni_cal_a55.pt (iou=0.38, TTA=True)
  -> E1 = 0.72118 | E2 (@0.24) = 0.71907 | mAP50 = 0.69406

Baseline Awal sebagai Pembanding :
  A_yolo11s_s42 / best.pt (iou=0.70, TTA=False)
  -> E1 = 0.68424 | E2 = 0.68351 | mAP50 = 0.69272

Kelas terlemah (F1 valid) :
  Brown spot (naik dari 0.3712 di baseline menjadi 0.4175 - 0.4206) dan Leaf scald (naik dari 0.5322 menjadi 0.5485).
  Kelas tertinggi: Narrow brown (0.9923), False smut (0.9351), Healthy (0.9011), Leaf roller (0.8788).

Kelandaian kurva F1 :
  Sangat landai di sekitar puncak (conf = 0.22 - 0.25), terbukti dari selisih E1 vs E2 yang hanya 0.00080 (0.72053 vs 0.71973) serta selisih determinisme 2x run = 0.00000.

Usulan :
  Menggunakan best_audit_safe.pt sebagai model rilis utama karena menembus E1 = 0.7205 (E2 = 0.7197) sekaligus menjamin selisih <= 1,31% di seluruh skenario audit juri (termasuk jika juri memakai parameter default iou=0.70 tanpa TTA).