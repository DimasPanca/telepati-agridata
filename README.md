# telepati-agridata

Deteksi penyakit daun padi (11 kelas) untuk AI AgriData Intelligence Race, TELEPATI 8.0.
Model YOLO11s dilatih dari nol menggunakan dataset resmi Agritech Data.

| | |
|---|---|
| Tim | The Power of Kepepet, Universitas Sultan Ageng Tirtayasa |
| Anggota | Dimas (Ketua), Rahman, Maman |
| Dosen Pembimbing | Arief Rahman, S.Kom., M.Kom |
| Notebook | [notebooks/agridata_pipeline.ipynb](notebooks/agridata_pipeline.ipynb) ([Colab](https://colab.research.google.com/github/DimasPanca/telepati-agridata/blob/main/notebooks/agridata_pipeline.ipynb)) |
| Versi | v1.3: Stage-2 No-Mosaic (`A_plus_nomosaic_s42`) + SWA + LBC Audit-Safe (`best_audit_safe.pt`) |
| Weights | [telepati_agridata_v1.0.pt](https://github.com/DimasPanca/telepati-agridata/releases/download/v1.0/telepati_agridata_v1.0.pt) |
| F1 makro valid | **0.7205** (`best_audit_safe.pt`, Audit-Safe <= 2%) / **0.7212** (`best_murni_cal_a55.pt`) |
| F1 makro test | 0.7023 (v1.2) |

## Cara Menjalankan

### Inference saja (reproduksi skor)

1. Buka notebook di Google Colab, pilih runtime **T4 GPU**.
2. Di sel konfigurasi (bagian 0), isi:
   ```python
   MODE = "inference_only"
   DATA_ZIP = "<path zip dataset resmi atau folder hasil ekstrak>"
   ```
   `WEIGHTS_URL`, `WEIGHTS_SHA256`, `E2_CONF`, `NMS_IOU`, dan `AGNOSTIC_NMS` sudah terisi.
3. Runtime > Run all.

Keluaran (bagian 11): F1 makro (E1, E2), mAP@50, F1 per kelas, dan waktu. Inference sekitar 10–25 ms per gambar di T4 (selesai ~33–68 detik).

### Pipeline penuh (training dari data mentah)

1. Isi `MODE = "full"`, `DATA_ZIP`, dan `RUNS_DIR`.
2. Jalankan dengan `RUN_ID = "A_yolo11s_s42"` (bagian 7), dilanjutkan refinement tanpa mosaic `A_plus_nomosaic_s42` (bagian 9).
3. Setiap tahap mengevaluasi semua checkpoint pada split valid (bagian 8 sampai 9f). Split test dievaluasi sekali dengan `RUN_FINAL_TEST = True` (bagian 10).

Run yang terputus dilanjutkan dari `last.pt` jika dijalankan ulang dengan `RUN_ID` yang sama.

## Pipeline

| Bagian | Langkah |
|---|---|
| 0 sampai 1 | Konfigurasi, environment, seed (`SEED = 42`, deterministik) |
| 2 | Ekstrak zip mentah, validasi jumlah gambar dan anotasi |
| 3 | Pemetaan 21 kategori COCO ke 11 kelas canonical |
| 4 | Pembersihan anotasi train (clip ke batas gambar, buang box < 2 px, buang duplikat) |
| 5 sampai 6 | Konversi COCO ke format YOLO, tulis `data.yaml` (train dan valid saja) |
| 7 | Training tahap 1 dari nol (`A_yolo11s_s42`, 150 epoch) |
| 8 sampai 8b | Evaluasi checkpoint pada valid dan kalibrasi post-processing (NMS `iou` & TTA) |
| 9 sampai 9b | Stage-2 fine-tuning tanpa mosaic (`A_plus_nomosaic_s42`, 25 epoch dari `epoch120.pt`) dan evaluasi valid |
| 9c sampai 9d | Cross-checkpoint weight averaging (SWA / Model Soup) dan kalibrasi bias logit per kelas (`cv3` LBC) |
| 9e sampai 9f | Uji determinisme 2x run dan stabilitas lintas skenario audit (<= 2%), ringkasan kandidat final |
| 10 | Evaluasi test sekali |
| 11 | Inference dari weights rilis |

## Dependencies

| Paket | Versi |
|---|---|
| ultralytics | 8.4.184 (dipin di [requirements.txt](requirements.txt), dipasang oleh notebook) |
| Python | 3.13.16 (Colab) |
| torch | 2.11.0+cu130 (Colab) |

## Model

| | |
|---|---|
| Arsitektur | YOLO11s (`yolo11s.yaml`), detect head anchor-free pada P3, P4, P5 |
| Parameter | 9.417.057 (setelah fusi) |
| GFLOPs | 21,4 pada 640 x 640 |
| Inisialisasi | Acak (`pretrained=False`) |
| Jumlah kelas | 11 |

## Hyperparameter

| Parameter | Tahap 1: `A_yolo11s_s42` | Tahap 2: `A_plus_nomosaic_s42` | Tahap 2 (v1.0): `A_fit_yolo11s_s42` |
|---|---|---|---|
| Bobot awal | acak (`yolo11s.yaml`) | `epoch120.pt` tahap 1 | `best.pt` tahap 1 |
| imgsz | 640 | 640 | 640 |
| epochs | 150 | 25 (terpilih: epoch 7 / `best.pt`) | 50 (terpilih: epoch 20) |
| batch | 16 | 16 | 16 |
| optimizer | auto (MuSGD), lr0 0.01 | SGD, lr0 0.0005, lrf 0.1, cos_lr True, warmup_epochs 0 | SGD, lr0 0.002, lrf 0.1, warmup_epochs 1 |
| box / cls / dfl | 7.5 / 0.5 / 1.5 | 7.5 / 1.0 / 1.5 | 7.5 / 0.5 / 1.5 |
| mosaic / close_mosaic | 1.0 / 10 | 0.0 / 0 | 0.0 / 0 |
| mixup | 0.1 | 0.0 | 0.0 |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 | 0.015 / 0.7 / 0.4 | 0 / 0 / 0 |
| scale / translate | 0.5 / 0.1 | 0.25 / 0.1 | 0.25 / 0.05 |
| fliplr / flipud | 0.5 / 0.5 | 0.5 / 0.5 | 0.5 / 0.5 |
| seed / deterministic | 42 / True | 42 / True | 42 / True |
| patience | 0 | 0 | 0 |
| save_period | 10 | 5 | 10 |

Evaluasi: `conf=0.001`, `max_det=300`, FP32. Pemilihan checkpoint, NMS `iou` (`0.38`–`0.45`), `E2_CONF=0.24`, serta kalibrasi lintas-ambang dipilih murni dari split valid ([results/checkpoint_eval_valid.csv](results/checkpoint_eval_valid.csv), [results/checkpoint_eval_valid_audit_safe.csv](results/checkpoint_eval_valid_audit_safe.csv), dan [results/A_plus_audit_stability_valid.csv](results/A_plus_audit_stability_valid.csv)).

## Weights

| Versi | Unduh | SHA-256 |
|---|---|---|
| v1.0 | [telepati_agridata_v1.0.pt](https://github.com/DimasPanca/telepati-agridata/releases/download/v1.0/telepati_agridata_v1.0.pt) | `cdd815e88a22b6d9d6ee205dba7e51944e9c563177f6bdab01d1d984cc373b67` |
| v1.3 (`best_audit_safe.pt`) | *(unggah ke GitHub Releases)* | *(isi SHA-256 setelah rilis)* |

## Hasil

| Versi / Kandidat | Split | F1 makro (E1) | F1 makro (E2) | mAP@50 | Catatan Evaluasi |
|---|---|---|---|---|---|
| v1.3 (`best_murni_cal_a55.pt`) | valid | **0.7212** | 0.7191 @ 0.24 | 0.6941 | `iou=0.38, TTA=True` (skor puncak valid) |
| v1.3 (`best_audit_safe.pt`) | valid | **0.7205** | **0.7197** @ 0.24 | 0.6963 | `iou=0.38, TTA=True` (selisih audit +1,31% <= 2%) |
| v1.3 (`best_audit_safe.pt`) | valid | 0.7194 | 0.7184 @ 0.24 | **0.6993** | `iou=0.45, TTA=True` (selisih audit +1,19% <= 2%) |
| v1.3 (`best_audit_safe.pt`) | valid | 0.7175 | 0.7163 @ 0.24 | 0.6907 | `iou=0.45, TTA=False` (tanpa TTA, selisih +1,01%) |
| v1.3 (`best_audit_safe.pt`) | valid | 0.7075 | 0.7061 @ 0.24 | 0.6885 | `iou=0.70, TTA=False` (default polos Ultralytics) |
| v1.2 | valid | 0.7113 | 0.7073 @ 0.20 | 0.6929 | `iou=0.45, TTA=False` pada bobot v1.0 |
| v1.2 | test | **0.7023** | 0.6963 @ 0.20 | 0.6952 | `iou=0.45, TTA=False` pada bobot v1.0 |
| v1.0 | valid | 0.6981 | 0.6930 @ 0.26 | 0.6884 | `iou=0.70, TTA=False` |
| v1.0 | test | 0.6951 | 0.6951 @ 0.26 | 0.6898 | `iou=0.70, TTA=False` |

E1: F1 makro pada confidence yang memaksimalkan rata-rata F1 (Ultralytics `val`). E2: F1 makro pada confidence tetap. Latihan audit di Colab T4: [results/audit_log.md](results/audit_log.md).

F1 per kelas pada split valid (v1.3 `best_audit_safe.pt`) dan split test (v1.2):

| Kelas | F1 valid (v1.3) | F1 test (v1.2) | Kelas | F1 valid (v1.3) | F1 test (v1.2) |
|---|---|---|---|---|---|
| Bacterial leaf blight | 0.5688 *(0.6005\*)* | 0.3963 | Leaf roller | 0.8788 | 0.8691 |
| Bacterial panicle blight | 0.7503 *(0.7618\*)* | 0.8857 | Leaf scald | 0.5485 | 0.5269 |
| Blast | 0.5923 | 0.5164 | Narrow brown | 0.9923 | 0.9967 |
| Brown spot | 0.4175 *(0.4206\*\*)* | 0.3840 | Sheath blight | 0.6748 | 0.6103 |
| False smut | 0.9351 | 0.9224 | Tungro | 0.6664 | 0.7026 |
| Healthy | 0.9011 | 0.9149 | **Rata-rata (E1)** | **0.7205** | **0.7023** |

*\*Nilai pada `best_murni_audit_safe.pt` (`E1 valid = 0.7205`). \*\*Nilai pada `iou=0.45`.*

Perbandingan run dan tahapan optimasi pada valid:

| Run / Tahap | Keterangan | F1 makro valid (E1) | mAP@50 valid |
|---|---|---|---|
| `A_yolo11s_s42` (`best.pt`) | dari nol, 150 epoch (`iou=0.70, TTA=False`) | 0.6842 | 0.6927 |
| `A_yolo11s_s42` (`epoch120.pt`) | seleksi checkpoint terbaik tahap 1 (`iou=0.70, TTA=False`) | 0.6874 | 0.6915 |
| `A800_yolo11s_s42` | lanjutan Run A pada 800 px, 40 epoch | 0.6887 | — |
| `A_fit_yolo11s_s42` | lanjutan Run A tanpa augmentasi berat, 50 epoch (v1.0) | 0.6981 | 0.6884 |
| `A_yolo11s_s42` (`epoch120.pt` + post-proc) | `epoch120.pt` dengan `iou=0.45, TTA=True` | 0.7111 | 0.7044 |
| `A_plus_nomosaic_s42` (`best.pt`) | lanjutan `epoch120.pt` tanpa mosaic, 25 epoch (`iou=0.38, TTA=True`) | 0.7163 | 0.6943 |
| `A_plus_SWA` (`swa_50best_25ep10_25ep120.pt`) | Model Soup 3 checkpoint lintas tahap (`iou=0.38, TTA=True`) | 0.7168 | 0.6975 |
| `best_murni_cal_a55.pt` | `A_plus/best.pt` + kalibrasi bias `cv3` (alpha=0.55) | **0.7212** | 0.6941 |
| `best_audit_safe.pt` | `SWA` + kalibrasi bias lintas-ambang (alpha=0.95, lolos audit <= 2%) | **0.7205** | 0.6963 |

Uji stabilitas `best_audit_safe.pt` terhadap variasi parameter evaluasi pada split valid ([results/A_plus_audit_stability_valid.csv](results/A_plus_audit_stability_valid.csv)):

| Skenario Evaluasi | `iou` | `TTA` | `half` | E1 valid | E2 valid | mAP@50 | Waktu (T4) | Selisih vs Default `0.70` | Status (<= 2%) |
|---|---|---|---|---|---|---|---|---|---|
| S-11 Peak (`iou=0.38, TTA=True, FP32`) | 0.38 | True | False | **0.7205** | **0.7197** | 0.6963 | 68.1s | +0.0131 (+1,31%) | AMAN |
| S-11 Optimal (`iou=0.45, TTA=True, FP32`) | 0.45 | True | False | 0.7194 | 0.7184 | 0.6993 | 68.1s | +0.0119 (+1,19%) | AMAN |
| S-11 + FP16 (`iou=0.45, TTA=True, FP16`) | 0.45 | True | True | 0.7194 | 0.7186 | 0.6991 | 34.2s | +0.0120 (+1,20%) | AMAN |
| S-11 IoU 0.50 (`iou=0.50, TTA=True, FP32`) | 0.50 | True | False | 0.7179 | 0.7171 | **0.7016** | 68.1s | +0.0105 (+1,05%) | AMAN |
| Tanpa TTA (`iou=0.45, TTA=False, FP32`) | 0.45 | False | False | 0.7175 | 0.7163 | 0.6907 | 33.4s | +0.0101 (+1,01%) | AMAN |
| Tanpa TTA (`iou=0.50, TTA=False, FP32`) | 0.50 | False | False | 0.7167 | 0.7153 | 0.6947 | 33.5s | +0.0093 (+0,93%) | AMAN |
| Default Ultralytics (`iou=0.70, TTA=False`) | 0.70 | False | False | **0.7075** | 0.7061 | 0.6885 | 33.5s | 0.0000 (0,00%) | AMAN |

Setiap versi ditetapkan dari valid sebelum test dievaluasi (D-19 dan D-20 di [docs/01_PRD.md](docs/01_PRD.md)).

Log training dan evaluasi: [results/](results/).

## Catatan

- Training hanya memakai split train. Checkpoint dan threshold dipilih dari valid. Test dievaluasi sekali untuk setiap model yang dikirim.
- Tanpa pretrained eksternal. Tahap 2 dimulai dari bobot tahap 1 hasil training tim.
- Seluruh preprocessing berjalan di notebook dari dataset mentah. Dataset tidak disertakan di repositori ini.
