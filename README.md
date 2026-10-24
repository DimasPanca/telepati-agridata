# telepati-agridata

Deteksi penyakit daun padi (11 kelas) untuk AI AgriData Intelligence Race, TELEPATI 8.0.
Model YOLO11s dilatih dari nol menggunakan dataset resmi Agritech Data.

| | |
|---|---|
| Tim | The Power of Kepepet, Universitas Sultan Ageng Tirtayasa |
| Anggota | Dimas (Ketua), Rahman, Maman |
| Dosen Pembimbing | Arief Rahman, S.Kom., M.Kom |
| Notebook | [notebooks/agridata_pipeline.ipynb](notebooks/agridata_pipeline.ipynb) ([Colab](https://colab.research.google.com/github/DimasPanca/telepati-agridata/blob/main/notebooks/agridata_pipeline.ipynb)) |
| Versi | v1.2: weights v1.0 dengan NMS IoU 0.45 per kelas |
| Weights | [telepati_agridata_v1.0.pt](https://github.com/DimasPanca/telepati-agridata/releases/download/v1.0/telepati_agridata_v1.0.pt) |
| F1 makro test | 0.7023 |

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

Keluaran (bagian 11): F1 makro (E1, E2), mAP@50, F1 per kelas, dan waktu. Inference sekitar 10 ms per gambar di T4.

### Pipeline penuh (training dari data mentah)

1. Isi `MODE = "full"`, `DATA_ZIP`, dan `RUNS_DIR`.
2. Jalankan dengan `RUN_ID = "A_yolo11s_s42"`, lalu `RUN_ID = "A_fit_yolo11s_s42"`.
3. Setiap run mengevaluasi semua checkpoint pada split valid (bagian 8). Split test dievaluasi sekali dengan `RUN_FINAL_TEST = True` (bagian 10).

Run yang terputus dilanjutkan dari `last.pt` jika dijalankan ulang dengan `RUN_ID` yang sama.

## Pipeline

| Bagian | Langkah |
|---|---|
| 0 sampai 1 | Konfigurasi, environment, seed |
| 2 | Ekstrak zip mentah, validasi jumlah gambar dan anotasi |
| 3 | Pemetaan 21 kategori COCO ke 11 kelas canonical |
| 4 | Pembersihan anotasi train (clip ke batas gambar, buang box < 2 px, buang duplikat) |
| 5 sampai 6 | Konversi COCO ke format YOLO, tulis `data.yaml` (train dan valid saja) |
| 7 | Training (tabel `RUNS`) |
| 8 sampai 9 | Evaluasi checkpoint pada valid, pilih model final, setelan NMS, dan `E2_CONF` |
| 9b | Opsional: checkpoint averaging dan kalibrasi bias per kelas |
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

| Parameter | Tahap 1: `A_yolo11s_s42` | Tahap 2: `A_fit_yolo11s_s42` |
|---|---|---|
| Bobot awal | acak | `best.pt` tahap 1 |
| imgsz | 640 | 640 |
| epochs | 150 | 50 (terpilih: epoch 20) |
| batch | 16 | 16 |
| optimizer | auto (MuSGD), lr0 0.01 | SGD, lr0 0.002, lrf 0.1, warmup_epochs 1 |
| mosaic / close_mosaic | 1.0 / 10 | 0.0 / 0 |
| mixup | 0.1 | 0.0 |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 | 0 / 0 / 0 |
| scale / translate | 0.5 / 0.1 | 0.25 / 0.05 |
| fliplr / flipud | 0.5 / 0.5 | 0.5 / 0.5 |
| seed / deterministic | 42 / True | 42 / True |
| patience | 0 | 0 |
| save_period | 10 | 10 |

Evaluasi: `conf=0.001`, `max_det=300`, FP32. Checkpoint diurutkan dengan NMS default Ultralytics (`iou=0.7`). Evaluasi akhir, test, dan inference memakai NMS `iou=0.45` per kelas dan `E2_CONF=0.20`, keduanya dipilih dari valid (bagian 9, [results/nms_selection_valid.csv](results/nms_selection_valid.csv)).

## Weights

| Versi | Unduh | SHA-256 |
|---|---|---|
| v1.0 | [telepati_agridata_v1.0.pt](https://github.com/DimasPanca/telepati-agridata/releases/download/v1.0/telepati_agridata_v1.0.pt) | `cdd815e88a22b6d9d6ee205dba7e51944e9c563177f6bdab01d1d984cc373b67` |

v1.2 memakai file weights v1.0 tanpa perubahan. Yang berubah hanya setelan NMS dan `E2_CONF` di notebook.

## Hasil

| Versi | Split | F1 makro (E1) | F1 makro (E2) | mAP@50 |
|---|---|---|---|---|
| v1.2 | valid | 0.7113 | 0.7073 @ 0.20 | 0.6929 |
| v1.2 | test | **0.7023** | 0.6963 @ 0.20 | 0.6952 |
| v1.0 | valid | 0.6981 | 0.6930 @ 0.26 | 0.6884 |
| v1.0 | test | 0.6951 | 0.6951 @ 0.26 | 0.6898 |

E1: F1 makro pada confidence yang memaksimalkan rata-rata F1 (Ultralytics `val`). E2: F1 makro pada confidence tetap. Latihan audit di Colab T4: [results/audit_log.md](results/audit_log.md).

F1 per kelas (test, v1.2):

| Kelas | F1 | Kelas | F1 |
|---|---|---|---|
| Bacterial leaf blight | 0.3963 | Leaf roller | 0.8691 |
| Bacterial panicle blight | 0.8857 | Leaf scald | 0.5269 |
| Blast | 0.5164 | Narrow brown | 0.9967 |
| Brown spot | 0.3840 | Sheath blight | 0.6103 |
| False smut | 0.9224 | Tungro | 0.7026 |
| Healthy | 0.9149 | | |

Perbandingan run pada valid:

| Run | Keterangan | F1 makro valid |
|---|---|---|
| A_yolo11s_s42 | dari nol, 150 epoch | 0.6835 (`best.pt`) |
| A800_yolo11s_s42 | lanjutan Run A pada 800 px, 40 epoch | 0.6887 |
| A_fit_yolo11s_s42 | lanjutan Run A tanpa augmentasi berat, 50 epoch | 0.6981 (epoch 20), dirilis sebagai v1.0 |

Setelan setelah training, dibandingkan pada bobot v1.0:

| Versi | Perubahan | E1 valid | E1 test |
|---|---|---|---|
| v1.0 | NMS default `iou=0.7` | 0.6981 | 0.6951 |
| v1.1 | kalibrasi bias per kelas di head, tidak dirilis | 0.7033 | 0.6936 |
| v1.2 | NMS `iou=0.45` per kelas | 0.7113 | 0.7023 |

Setiap versi ditetapkan dari valid sebelum test dievaluasi (D-19 dan D-20 di [docs/01_PRD.md](docs/01_PRD.md)).

Log training: [results/](results/).

## Catatan

- Training hanya memakai split train. Checkpoint dan threshold dipilih dari valid. Test dievaluasi sekali untuk setiap model yang dikirim.
- Tanpa pretrained eksternal. Tahap 2 dimulai dari bobot tahap 1 hasil training tim.
- Seluruh preprocessing berjalan di notebook dari dataset mentah. Dataset tidak disertakan di repositori ini.
