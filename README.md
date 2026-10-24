# telepati-agridata

Model object detection penyakit padi (11 kelas) untuk **AI AgriData Intelligence Race — TELEPATI 8.0** oleh tim **The Power of Kepepet**, dilatih dari nol pada dataset resmi Agritech Data.

| | |
|---|---|
| Tim | The Power of Kepepet — Universitas Sultan Ageng Tirtayasa |
| Anggota | Dimas (Ketua), Rahman, Maman |
| Dosen Pembimbing | *(nama)* |
| Notebook | [`notebooks/agridata_pipeline.ipynb`](notebooks/agridata_pipeline.ipynb) · [Open in Colab](https://colab.research.google.com/github/DimasPanca/telepati-agridata/blob/main/notebooks/agridata_pipeline.ipynb) |
| Model weights | [`telepati_agridata_v1.0.pt`](https://github.com/DimasPanca/telepati-agridata/releases/download/v1.0/telepati_agridata_v1.0.pt) (GitHub Release `v1.0`) |
| Skor mandiri | **F1 makro 0,6951** pada split test resmi (Colab T4) |
| Dokumen tim | [`docs/`](docs/) — mulai dari `docs/README_TIM.txt` |

---

## 1. Ringkasan

- **Tugas:** object detection 11 kelas kondisi tanaman padi (10 penyakit + Healthy) pada citra 640×640.
- **Data:** dataset resmi Agritech Data. 21 kategori COCO dipetakan ke 11 kelas canonical di dalam notebook; tiga supercategory tanpa bounding box tidak dipakai.
- **Model:** YOLO11s (Ultralytics), dibangun dari `yolo11s.yaml` dan **dilatih dari nol tanpa pretrained**, lalu di-fine-tune dari bobot tim sendiri.
- **Evaluasi:** F1-Score makro pada split test resmi.
- **Platform:** Google Colab, GPU T4.

---

## 2. How to Run

### 2.1 Reproduksi skor (inference-saja, untuk audit)

1. Buka notebook di Google Colab: [Open in Colab](https://colab.research.google.com/github/DimasPanca/telepati-agridata/blob/main/notebooks/agridata_pipeline.ipynb).
2. Runtime → Change runtime type → **T4 GPU**.
3. Siapkan zip dataset resmi apa adanya (format COCO, split `train/`, `valid/`, `test/`; folder pembungkus dan file `__MACOSX` ditangani otomatis), lalu isi path-nya di `DATA_ZIP` pada sel konfigurasi (bagian 0). `DATA_ZIP` juga boleh berisi folder hasil ekstrak.
4. Set `MODE = "inference_only"`. `WEIGHTS_URL`, `WEIGHTS_SHA256`, dan `E2_CONF` sudah terisi.
5. Runtime → Run all. Notebook akan:
   - memasang `ultralytics==8.4.184`,
   - mengekstrak dan memvalidasi dataset, lalu memetakan label ke 11 kelas,
   - mengunduh weights dari direct link Release dan memeriksa SHA-256,
   - menjalankan inference pada split test resmi dan mencetak F1 makro (E1 dan E2), mAP@50, F1 per kelas, serta waktu.

Inference di T4 sekitar 10 ms per gambar (1.059 gambar test). Menjalankan ulang menghasilkan angka yang identik.

### 2.2 Reproduksi penuh (training dari data mentah)

1. Langkah 1–3 seperti di atas, lalu set `MODE = "full"`.
2. Jalankan berurutan dengan `RUN_ID` dari tabel `RUNS` (bagian 7 notebook):
   1. `A_yolo11s_s42` — training dari nol (150 epoch).
   2. `A_fit_yolo11s_s42` — fine-tuning dari `best.pt` Run A (50 epoch). Membutuhkan folder Run A di `RUNS_DIR`.
3. Setiap run otomatis mengevaluasi semua checkpoint di valid (bagian 8). Model final dipilih dari valid (bagian 9), lalu test dievaluasi sekali dengan `RUN_FINAL_TEST = True` (bagian 10).
4. Jika sesi Colab terputus, jalankan ulang dengan `RUN_ID` yang sama: training dilanjutkan dari `last.pt`.

---

## 3. Dependencies

| Paket | Versi |
|---|---|
| ultralytics | 8.4.184 (dipin di [`requirements.txt`](requirements.txt), dipasang oleh notebook) |
| Python | 3.13.16 (Colab) |
| torch | 2.11.0+cu130 (Colab) |
| numpy, pandas, PyYAML | versi bawaan Colab |

Versi Python dan torch adalah versi Colab saat latihan audit dan dicetak ulang di bagian 1 notebook setiap kali dijalankan.

---

## 4. Arsitektur model

| | |
|---|---|
| Arsitektur | YOLO11s (`yolo11s.yaml`), head deteksi anchor-free dengan 3 skala (P3–P5) |
| Parameter | 9.417.057 (setelah fusi lapisan) |
| Komputasi | 21,4 GFLOPs pada 640×640 |
| Inisialisasi | Acak (`pretrained=False`), tanpa bobot eksternal |
| Kelas | 11 |

Alasan pemilihan: keluarga detektor transformer (DETR) bergantung pada backbone pretrained dan konvergen lambat jika dilatih dari nol, sedangkan YOLO berbasis CNN konvergen cepat dari nol dan pasti termuat di Colab. Dari eksperimen yang dibandingkan di valid (bagian 7), YOLO11s dengan tahap fine-tuning tanpa augmentasi berat memberi F1 makro valid tertinggi.

---

## 5. Hyperparameter

Training dua tahap. Seluruh tahap hanya memakai split train.

| Parameter | Tahap 1: Run A (dari nol) | Tahap 2: A_fit (fine-tuning) |
|---|---|---|
| Bobot awal | acak (`yolo11s.yaml`) | `best.pt` Run A |
| `imgsz` | 640 | 640 |
| Epoch | 150 (`patience=0`) | 50 (checkpoint terpilih: epoch 20) |
| Batch | 16 | 16 |
| Optimizer | auto (MuSGD, `lr0=0.01`) | SGD, `lr0=0.002`, `lrf=0.1`, `warmup_epochs=1` |
| Mosaic / mixup | 1.0 (mati 10 epoch terakhir) / 0.1 | 0 / 0 |
| HSV (h, s, v) | 0.015, 0.7, 0.4 | 0, 0, 0 |
| Skala / translasi | 0.5 / 0.1 | 0.25 / 0.05 |
| Flip horizontal / vertikal | 0.5 / 0.5 | 0.5 / 0.5 |
| Seed / deterministic | 42 / True | 42 / True |
| AMP | True | True |
| Checkpoint berkala | tiap 10 epoch | tiap 10 epoch |

Evaluasi dan inference: `conf=0.001`, NMS `iou=0.7`, `max_det=300`, FP32, `batch=16`. `E2_CONF = 0.26` dipilih dari valid.

Pembersihan data (train saja): koordinat dipotong ke batas gambar, box dengan sisi < 2 px dibuang (22 box), box identik dalam satu gambar disisakan satu (7 box). Label valid dan test hanya dipetakan ke 11 kelas.

---

## 6. Model weights

| Versi | Direct link | SHA-256 |
|---|---|---|
| v1.0 | [telepati_agridata_v1.0.pt](https://github.com/DimasPanca/telepati-agridata/releases/download/v1.0/telepati_agridata_v1.0.pt) | `cdd815e88a22b6d9d6ee205dba7e51944e9c563177f6bdab01d1d984cc373b67` |

---

## 7. Hasil

Skor mandiri diambil dari jalur inference-saja di Colab T4 (lihat [`results/audit_log.md`](results/audit_log.md)).

| Split | F1 makro (E1) | F1 makro @ conf 0,26 (E2) | mAP@50 |
|---|---|---|---|
| valid | 0,6981 | 0,6930 | 0,6884 |
| **test** | **0,6951** | **0,6951** | **0,6898** |

F1 per kelas pada test:

| Kelas | F1 | Kelas | F1 |
|---|---|---|---|
| Bacterial leaf blight | 0,4315 | Leaf roller | 0,8613 |
| Bacterial panicle blight | 0,8547 | Leaf scald | 0,5059 |
| Blast | 0,5167 | Narrow brown | 0,9986 |
| Brown spot | 0,3431 | Sheath blight | 0,6065 |
| False smut | 0,9133 | Tungro | 0,7042 |
| Healthy | 0,9104 | | |

Ringkasan eksperimen (F1 makro valid, checkpoint terbaik tiap run):

| Run | Deskripsi | F1 makro valid |
|---|---|---|
| `A_yolo11s_s42` | YOLO11s dari nol, 150 epoch | 0,6835 (`best.pt`) |
| `A800_yolo11s_s42` | Lanjutan Run A pada 800 px, 40 epoch | 0,6887 |
| **`A_fit_yolo11s_s42`** | **Lanjutan Run A tanpa augmentasi berat, 50 epoch** | **0,6981 (epoch 20) — dipakai untuk v1.0** |

Log training dan evaluasi tiap run ada di [`results/`](results/).

---

## 8. Kepatuhan aturan

- Training hanya memakai split train. Pemilihan checkpoint, hyperparameter, dan `E2_CONF` memakai valid. Test dievaluasi satu kali untuk model yang dikirim.
- Tidak ada pretrained model eksternal. Tahap fine-tuning memakai bobot hasil training tim sendiri.
- Seluruh pemrosesan data dimulai dari dataset mentah di dalam notebook.
- Random seed tetap di setiap tahap.
- Dataset resmi tidak disertakan di repositori ini.
