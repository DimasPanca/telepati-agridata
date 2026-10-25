# telepati-agridata

Deteksi penyakit daun padi (11 kelas) untuk AI AgriData Intelligence Race, TELEPATI 8.0.
Model YOLO11s dilatih dari nol menggunakan dataset resmi Agritech Data.

| | |
|---|---|
| Tim | The Power of Kepepet, Universitas Sultan Ageng Tirtayasa |
| Anggota | Dimas (Ketua), Rachman, Winata |
| Dosen Pembimbing | Arief Rahman, S.Kom., M.Kom |
| Notebook | [notebooks/agridata_pipeline.ipynb](notebooks/agridata_pipeline.ipynb) ([Colab](https://colab.research.google.com/github/DimasPanca/telepati-agridata/blob/main/notebooks/agridata_pipeline.ipynb)) |
| Versi | v1.3: weights v1.3 dengan TTA dan NMS IoU 0.45 per kelas |
| Weights | [telepati_agridata_v1.3.pt](https://github.com/DimasPanca/telepati-agridata/releases/download/v1.3/telepati_agridata_v1.3.pt) |
| F1 makro test | 0.7123 |

## Cara Menjalankan

### Inference saja (reproduksi skor)

1. Buka notebook di Google Colab, pilih runtime **T4 GPU**.
2. Di sel konfigurasi (bagian 0), isi:
   ```python
   MODE = "inference_only"
   DATA_ZIP = "<path zip dataset resmi atau folder hasil ekstrak>"
   ```
   `WEIGHTS_URL`, `WEIGHTS_SHA256`, `E2_CONF`, `NMS_IOU`, `AGNOSTIC_NMS`, dan `TTA` sudah terisi.
3. Runtime > Run all.

Keluaran (bagian 11): F1 makro (E1, E2), mAP@50, F1 per kelas, dan waktu. Dengan TTA, inference sekitar 30 ms per gambar di T4 (1.059 gambar test).

### Pipeline penuh (training dari data mentah)

1. Isi `MODE = "full"`, `DATA_ZIP`, dan `RUNS_DIR`.
2. Jalankan dengan `RUN_ID = "A_yolo11s_s42"`, lalu `RUN_ID = "A_plus_nomosaic_s42"`.
3. Setiap run mengevaluasi semua checkpoint pada split valid (bagian 8).
4. Dengan `RUN_ID = "A_plus_nomosaic_s42"`, isi `BUILD_V13 = True` untuk menyusun bobot v1.3 (bagian 9c).
5. Split test dievaluasi sekali dengan `RUN_FINAL_TEST = True` (bagian 10).

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
| 8 sampai 9 | Evaluasi checkpoint pada valid, pilih model final, setelan TTA, NMS, dan `E2_CONF` dengan uji silang dua paruh |
| 9b | Opsional: model soup lintas run dan kalibrasi bias per kelas |
| 9c | Bobot v1.3: rata-rata bobot tiga checkpoint, lalu geser bias kelas di head |
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

Bobot v1.3 dibangun dalam tiga tahap. Tahap 1 dan 2 adalah training; tahap 3 hanya mengolah bobot hasil training tanpa data tambahan.

| Parameter | Tahap 1: `A_yolo11s_s42` | Tahap 2: `A_plus_nomosaic_s42` |
|---|---|---|
| Bobot awal | acak | `epoch120.pt` tahap 1 |
| imgsz | 640 | 640 |
| epochs | 150 | 25 |
| batch | 16 | 16 |
| optimizer | auto (MuSGD), lr0 0.01, lrf 0.01, warmup_epochs 3 | SGD, lr0 0.0005, lrf 0.1, cos_lr, warmup_epochs 0, warmup_bias_lr 0 |
| mosaic / close_mosaic | 1.0 / 10 | 0.0 / 0 |
| mixup | 0.1 | 0.0 |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 | 0.015 / 0.7 / 0.4 |
| scale / translate | 0.5 / 0.1 | 0.25 / 0.1 |
| fliplr / flipud | 0.5 / 0.5 | 0.5 / 0.5 |
| loss cls / box / dfl | 0.5 / 7.5 / 1.5 | 1.0 / 7.5 / 1.5 |
| seed / deterministic | 42 / True | 42 / True |
| patience | 0 | 0 |
| save_period | 10 | 5 |

Tahap 3 (bagian 9c):

| Langkah | Setelan |
|---|---|
| Rata-rata bobot | 50% `A_plus_nomosaic_s42/best.pt`, 25% `A_plus_nomosaic_s42/epoch10.pt`, 25% `A_yolo11s_s42/epoch120.pt` |
| Geser bias kelas di head (`cv3`) | 0.95 x (logit(0.24) - logit(conf puncak F1 kelas di valid, NMS 0.7)), dibatasi -1.35 sampai 1.35 |

Evaluasi: `conf=0.001`, `max_det=300`, FP32. Checkpoint diurutkan dengan NMS default Ultralytics (`iou=0.7`). Evaluasi akhir, test, dan inference v1.3 memakai TTA bawaan Ultralytics (`augment=True`), NMS `iou=0.45` per kelas, dan `E2_CONF=0.25`, semuanya dipilih dari valid (bagian 9, [results/v13_selection_valid.csv](results/v13_selection_valid.csv)).

## Weights

| Versi | Unduh | SHA-256 |
|---|---|---|
| v1.3 | [telepati_agridata_v1.3.pt](https://github.com/DimasPanca/telepati-agridata/releases/download/v1.3/telepati_agridata_v1.3.pt) | `04d2f640c7370988415a8594c8a7658e4dee86267c6a0cfb3cc9bc45936b1e7a` |
| v1.0 | [telepati_agridata_v1.0.pt](https://github.com/DimasPanca/telepati-agridata/releases/download/v1.0/telepati_agridata_v1.0.pt) | `cdd815e88a22b6d9d6ee205dba7e51944e9c563177f6bdab01d1d984cc373b67` |

v1.2 memakai file weights v1.0 tanpa perubahan. Yang berubah hanya setelan NMS dan `E2_CONF` di notebook.

## Hasil

| Versi | Split | F1 makro (E1) | F1 makro (E2) | mAP@50 |
|---|---|---|---|---|
| v1.3 | test | **0.7123** | 0.7053 @ 0.25 | 0.7038 |
| v1.2 | valid | 0.7113 | 0.7073 @ 0.20 | 0.6929 |
| v1.2 | test | 0.7023 | 0.6963 @ 0.20 | 0.6952 |
| v1.0 | valid | 0.6981 | 0.6930 @ 0.26 | 0.6884 |
| v1.0 | test | 0.6951 | 0.6951 @ 0.26 | 0.6898 |

E1: F1 makro pada confidence yang memaksimalkan rata-rata F1 (Ultralytics `val`). E2: F1 makro pada confidence tetap. Latihan audit di Colab T4: [results/audit_log.md](results/audit_log.md).

F1 per kelas (test):

| Kelas | v1.3 | v1.2 |
|---|---|---|
| Bacterial leaf blight | 0.4878 | 0.3963 |
| Bacterial panicle blight | 0.8540 | 0.8857 |
| Blast | 0.5282 | 0.5164 |
| Brown spot | 0.3981 | 0.3840 |
| False smut | 0.9316 | 0.9224 |
| Healthy | 0.9308 | 0.9149 |
| Leaf roller | 0.8688 | 0.8691 |
| Leaf scald | 0.5291 | 0.5269 |
| Narrow brown | 0.9942 | 0.9967 |
| Sheath blight | 0.6123 | 0.6103 |
| Tungro | 0.7005 | 0.7026 |

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

Pemilihan v1.3 memakai F1 uji silang dua paruh di valid: confidence dipilih di satu paruh lalu F1 dihitung di paruh lainnya, 5 kali bergantian, dengan potongan yang sama untuk semua kandidat.

| Bobot dan setelan | F1 uji silang valid | Selisih vs v1.2 |
|---|---|---|
| v1.2: bobot v1.0, NMS 0.45 per kelas | 0.7065 | |
| bobot v1.0 + TTA, NMS 0.45 per kelas | 0.7029 | -0.0036 |
| bobot v1.3, NMS 0.45 per kelas | 0.7061 | -0.0003 |
| **v1.3: bobot v1.3 + TTA, NMS 0.45 per kelas** | **0.7115** | **+0.0050** |

Bias kelas v1.3 digeser berdasarkan valid yang sama, sehingga angka valid v1.3 optimistis. Tabel lengkap: [results/v13_selection_valid.csv](results/v13_selection_valid.csv).

Setiap versi ditetapkan dari valid sebelum test dievaluasi (D-19, D-20, dan D-21 di [docs/01_PRD.md](docs/01_PRD.md)).

Log training: [results/](results/).

## Catatan

- Training hanya memakai split train. Checkpoint dan threshold dipilih dari valid. Test dievaluasi sekali untuk setiap model yang dikirim.
- Tanpa pretrained eksternal. Tahap 2 dimulai dari bobot tahap 1 hasil training tim, dan tahap 3 hanya mengolah bobot tim sendiri.
- Satu model: TTA bawaan Ultralytics memproses gambar yang sama pada tiga skala dengan satu file weights.
- Seluruh preprocessing berjalan di notebook dari dataset mentah. Dataset tidak disertakan di repositori ini.
