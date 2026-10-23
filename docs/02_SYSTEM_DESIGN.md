---
doc_id: SYSTEM_DESIGN
judul: "System Design — Pipeline Object Detection Penyakit Padi"
versi: 1.1
tanggal: 2026-10-23
status: AKTIF
pemilik: Dimas (Arsitek)
bergantung_pada: [01_PRD.md]
dokumen_terkait: [03_DATA_CARD.md, 04_EXPERIMENT_PLAN.md, 05_AUDIT_RUNBOOK.md]
implementasi: notebooks/agridata_pipeline.ipynb
---

# System Design — Pipeline Object Detection Penyakit Padi

> **Untuk manusia dan AI.** Dokumen ini menjelaskan *bagaimana* sistem disusun. *Apa* dan *kenapa* ada di `01_PRD.md`. Setiap tahap pipeline punya ID `S-xx` yang sama dengan nomor bagian di notebook. Aturan lomba dirujuk dengan `R-xx` dan keputusan dengan `D-xx` dari PRD.

---

## 1. Gambaran besar

```text
[Dataset mentah resmi (zip)]
        │  disimpan di Google Drive masing-masing anggota (privat)
        ▼
[Google Colab T4] ── notebooks/agridata_pipeline.ipynb
   S-00 Konfigurasi (MODE, RUN_ID, path, seed)
   S-01 Lingkungan: pasang ultralytics 8.4.184, mount Drive, kunci seed
   S-02 Dataset mentah: ekstrak (zip resmi atau zip tim), cek kelengkapan, validasi jumlah
   S-03 Pemetaan 21 → 11 kelas
   S-04 Pembersihan anotasi (train saja)
   S-05 Konversi COCO → YOLO (+ repeat factor sampling untuk Run C)
   S-06 data.yaml (train + valid saja)
   S-07 Training dari nol (tabel RUNS: E0, A, B, C)
   S-08 Evaluasi checkpoint di valid (E1, mAP@50) + fungsi E2
   S-09 Pemilihan model final dan E2_CONF (valid saja)
   S-10 Evaluasi test SEKALI
   S-11 Inference dari weights rilis (jalur audit)
        │
        ├── checkpoint sementara → Google Drive (RUNS_DIR/RUN_ID)
        ├── weights final → GitHub Releases (R-13, D-08)
        └── results.csv, args.yaml, tabel evaluasi, notebook ber-output → repo GitHub
```

Prinsip desain:

1. **Satu notebook, dua mode.** `MODE = "full"` menjalankan S-01 sampai S-10. `MODE = "inference_only"` menjalankan S-01 sampai S-03 lalu S-11. Semua fungsi tetap didefinisikan di kedua mode; hanya *aksi* yang dijaga dengan `if MODE == ...`.
2. **Satu sumber kebenaran per hal.** Pemetaan kelas, aturan pembersihan, fungsi evaluasi, dan parameter inference masing-masing didefinisikan sekali lalu dipakai ulang.
3. **Test terisolasi.** Test tidak pernah masuk `data.yaml` training. Test hanya disentuh di S-10 dan S-11 (R-20).
4. **Semua parameter eksplisit.** Tidak ada nilai penting yang bergantung pada default library tanpa ditulis.

---

## 2. Lingkungan

### 2.1 Google Colab (utama, D-01)

| Hal | Desain |
|---|---|
| Runtime | GPU T4. Sel S-01 memasang `ultralytics==8.4.184` (D-13). PyTorch memakai versi bawaan Colab dan versinya dicetak. |
| Akun | Setiap anggota memakai akun Google sendiri. Kuota GPU dihitung per akun, jadi tiga run bisa berjalan paralel. |
| Data | Zip dataset di Drive → diekstrak ke `/content/work/raw` (metadata macOS dilewati, folder split dicari otomatis). **Tidak** melatih langsung dari Drive. |
| Penyimpanan kerja | Run ditulis langsung ke `RUNS_DIR/RUN_ID` di Drive, supaya checkpoint selamat kalau sesi putus. |
| Batas sesi | Sesi gratis maksimal sekitar 12 jam dan bisa putus saat idle. Jalankan ulang notebook dengan `RUN_ID` yang sama → otomatis resume dari `last.pt`. |
| Fitur AI Colab | Tidak dipakai untuk mengolah dataset (R-11). |

### 2.2 Struktur folder Google Drive (per anggota)

```text
MyDrive/telepati/
├── data/
│   └── telepati-agridata-raw.zip        # 613.048.556 byte, SHA-256 di 03_DATA_CARD.md
└── runs/
    ├── <RUN_ID>/                        # contoh: A_yolo11s_s42
    │   ├── weights/epoch10.pt … last.pt, best.pt
    │   ├── results.csv, args.yaml, plot training
    │   └── checkpoint_eval_valid.csv    # hasil S-08
    └── final_test_scores.json           # hasil S-10 (hanya di akun Dimas)
```

### 2.3 Struktur runtime Colab (sementara)

```text
/content/work/
├── raw/              # hasil ekstrak zip: train/ valid/ test/ (COCO)
├── yolo/
│   ├── images/{train,valid,test}/   # hard link ke raw/ (atau salinan)
│   └── labels/{train,valid,test}/   # label YOLO
├── data.yaml         # train + valid (S-06)
├── data_test.yaml    # khusus S-10 / S-11
└── weights/          # weights rilis yang diunduh di S-11
```

### 2.4 Kaggle (cadangan)

Dipakai hanya kalau Colab tidak memberi GPU. Ganti `DATA_ZIP` ke `/kaggle/input/...`, `RUNS_DIR` dan `WORK_DIR` ke `/kaggle/working/...`. Tidak ada path lain yang perlu diubah.

---

## 3. Struktur repo

```text
telepati-agridata/
├── README.md                     # untuk juri (R-15)
├── requirements.txt              # versi dipin (NFR-03)
├── .gitignore                    # dataset, zip, weights, runs/, kredensial
├── notebooks/
│   └── agridata_pipeline.ipynb   # satu-satunya notebook resmi (R-07, R-16)
├── results/
│   ├── <RUN_ID>/results.csv, args.yaml, checkpoint_eval_valid.csv
│   ├── checkpoint_eval_valid.csv  # gabungan semua run (S-08)
│   └── audit_log.md               # catatan latihan audit
└── docs/                          # README_TIM.txt, 01_PRD.md … 06_GIT_WORKFLOW.md
```

Konfigurasi run tidak disimpan sebagai file terpisah; semuanya ada di tabel `RUNS` di bagian 7 notebook (D-15).

---

## 4. Peta notebook

| Bagian notebook | Tahap | Isi utama | `full` | `inference_only` |
|---|---|---|---|---|
| 0 | S-00 | `MODE`, `RUN_ID`, path, `WEIGHTS_URL`, `WEIGHTS_SHA256`, `E2_CONF`, `SEED`, `IMGSZ`, `RUN_FINAL_TEST` | ✓ | ✓ |
| 1 | S-01 | Pasang ultralytics, mount Drive, `set_seed`, cetak versi | ✓ | ✓ |
| 2 | S-02 | `sha256sum`, `find_dataset_root`, `is_complete`, `prepare_dataset`, `load_coco`, validasi `EXPECTED_COUNTS` | ✓ | ✓ |
| 3 | S-03 | `CLASS_NAMES`, `RAW_TO_CANONICAL`, `SUPERCATEGORIES`, `to_frames`, tabel jumlah box | ✓ | ✓ |
| 4 | S-04 | `clip_boxes`, `clean_train_boxes` | ✓ | definisi saja |
| 5 | S-05 | `export_split`, `repeat_factors` | ✓ | definisi saja |
| 6 | S-06 | `write_data_yaml` | ✓ | definisi saja |
| 7 | S-07 | `TRAIN_ARGS`, `RUNS`, `train` (dengan resume) | ✓ | — |
| 8 | S-08 | `evaluate_e1`, `evaluate_run`, fungsi E2 (`ground_truth`, `predict_split`, `match_predictions`, `macro_f1_at`, `best_conf`) | ✓ | definisi saja |
| 9 | S-09 | `SELECTED_WEIGHTS`, hitung `E2_CONF` dan kelandaian kurva | ✓ | — |
| 10 | S-10 | `prepare_test_data`, `score_test` (dijaga `RUN_FINAL_TEST`) | ✓ | definisi saja |
| 11 | S-11 | `download_weights`, skor test dari weights rilis, waktu total | — | ✓ |

Setiap bagian diawali sel Markdown yang menjelaskan tujuan, keluaran, dan aturan yang dipenuhi (R-16).

---

## 5. Desain tahap data

### S-02 Dataset mentah
- Masukan: `telepati-agridata-raw.zip` berisi `train/`, `valid/`, `test/`, masing-masing gambar `.jpg` dan `_annotations.coco.json`.
- Menerima zip resmi panitia (split dibungkus satu folder, ada `__MACOSX`, `._*`, `.DS_Store`) maupun zip tim (split langsung di akar). File metadata macOS dilewati saat ekstraksi; folder berisi `train/_annotations.coco.json` dicari otomatis dan menjadi `RAW_DIR`.
- Ekstraksi dianggap selesai hanya jika semua gambar yang tercantum di file COCO ada. Ekstraksi yang terputus diulang dari awal.
- SHA-256 dibandingkan dengan zip referensi tim hanya sebagai informasi, karena juri memakai zip resmi yang checksum-nya berbeda. Validitas isi dipastikan lewat jumlah gambar dan anotasi terhadap `03_DATA_CARD.md` §2.

### S-03 Pemetaan kelas
- `RAW_TO_CANONICAL` berbasis **nama** kategori raw. Kategori yang tidak dikenal menghentikan notebook dengan error.
- Anotasi supercategory dilewati (saat ini 0).
- Validasi: jumlah box setelah pemetaan sama dengan jumlah anotasi COCO per split.

### S-04 Pembersihan (D-14)

| Aturan | Train | Valid / Test |
|---|---|---|
| Koordinat di luar batas gambar → dipotong | ✓ | ✓ |
| Sisi box < 2 px → dibuang | ✓ (22 box) | — |
| Box identik dalam satu gambar → disisakan satu | ✓ (7 box) | — |
| Box seluas gambar penuh | dipertahankan | dipertahankan |
| Gambar tanpa anotasi | dipertahankan sebagai latar | dipertahankan |

Alasan valid/test tidak dibersihkan: skor harus dihitung terhadap label yang sama dengan yang dipakai juri.

### S-05 Konversi COCO → YOLO
- COCO `[x, y, w, h]` piksel → YOLO `class cx cy w h` ternormalisasi, 6 desimal.
- Gambar ditautkan dengan hard link (fallback: salin) ke `yolo/images/<split>`.
- Folder split dihapus dan dibuat ulang setiap kali supaya tidak ada sisa dari run lain.
- Test hanya dikonversi di S-10/S-11 dan ditulis ke `data_test.yaml` terpisah.

### Repeat factor sampling (Run C)
- Frekuensi kelas `f_c` = proporsi gambar train yang memuat kelas `c`.
- Faktor kelas `r_c = max(1, sqrt(t / f_c))`, dengan `t = 0.10`.
- Faktor gambar = faktor kelas terbesar di gambar itu, dibulatkan secara acak dengan RNG ber-seed.
- Gambar diulang sebagai file baru (`<nama>__rep<k>.jpg`) dengan label yang sama. Hanya data train yang dipakai.
- Dengan `t = 0.10`, kelas dengan frekuensi gambar terendah (Narrow brown 2,2%, Bacterial leaf blight 2,3%) mendapat faktor sekitar 2,1.

### Kebijakan split

| Split | Dipakai untuk | Dilarang |
|---|---|---|
| train | Training | — |
| valid | Seleksi checkpoint dan `E2_CONF` | Training, pseudo-label |
| test | S-10 (sekali) dan S-11 | Training, seleksi apa pun, pencocokan gambar dengan train |

---

## 6. Desain training (S-07)

| Parameter | Nilai | Catatan |
|---|---|---|
| Framework | ultralytics 8.4.184 | D-13 |
| Model | `yolo11s.yaml` atau `yolo26s.yaml` | Dibangun dari `.yaml`, bukan `.pt` (R-10) |
| `pretrained` | False | R-10 |
| `imgsz` | 640 | D-04 |
| `epochs` | 150 (disesuaikan dari E0) | `patience=0` → semua run berjalan penuh |
| `batch` | 16 | |
| `seed` / `deterministic` | 42 / True | R-08 |
| `workers` | 2 | vCPU Colab terbatas |
| `cache` | False | RAM Colab tidak cukup untuk cache |
| Augmentasi | mosaic 1.0 (mati 10 epoch terakhir), mixup 0.1, fliplr 0.5, flipud 0.5, HSV default | Copy-paste tidak dipakai (butuh label segmentasi) |
| `save_period` | 10 | Checkpoint berkala untuk S-08 |
| Output | `project=RUNS_DIR`, `name=RUN_ID`, `exist_ok=True` | Langsung ke Drive |
| Resume | Ada `last.pt` → `train(resume=True)`; kalau training sudah selesai, dilewati | |

Catatan AMP: Ultralytics mengunduh model kecil hanya untuk *memeriksa* dukungan mixed precision. Bobot itu tidak dipakai untuk inisialisasi. Penjelasan ini ada di Markdown bagian 1 notebook.

---

## 7. Desain evaluasi (S-08, S-10)

| Versi | Definisi | Kegunaan |
|---|---|---|
| **E1** | Rata-rata F1 per kelas dari `model.val()`. Ultralytics mengambil F1 pada satu confidence yang memaksimalkan rata-rata F1 semua kelas, IoU 0,5. | [DUGAAN] paling mirip cara juri. Metrik utama seleksi. |
| **E2** | F1 makro 11 kelas pada confidence tetap `E2_CONF`. Prediksi per kelas diurutkan dari confidence tertinggi lalu dicocokkan greedy ke label dengan IoU ≥ 0,5. | Cadangan bila juri memakai confidence tetap. |
| mAP@50 | Dari `model.val()` | Pembanding |

Parameter terkunci: `imgsz=640`, `batch=16`, `conf=0.001`, `iou=0.7` (NMS), `max_det=300`, `quantize=32` (FP32).

Pencocokan E2 dihitung sekali pada confidence 0,001. Karena prediksi diproses dari confidence tertinggi, memotong daftar pada threshold mana pun menghasilkan pencocokan yang sama, sehingga semua threshold bisa dievaluasi dari satu kali inference.

Keluaran S-08 per checkpoint: `run_id, checkpoint, E1, mAP50, F1 per kelas` → `RUNS_DIR/RUN_ID/checkpoint_eval_valid.csv`.

---

## 8. Pemilihan model (S-09)

1. Kandidat: semua checkpoint dari Run A, B, C (gabungan CSV S-08).
2. Urutkan berdasarkan **E1 di valid**.
3. Isi `SELECTED_WEIGHTS` dengan kandidat teratas, jalankan bagian 9: notebook mencetak `E2_CONF` terbaik di valid dan nilai F1 di ±0,05 dan ±0,10 dari titik itu.
4. Pilih kandidat dengan kurva landai dan tanpa kelas yang runtuh.
5. Keputusan oleh Dimas berdasarkan laporan Rahman. Test tidak dilihat (R-20b).

---

## 9. Jalur inference-saja (S-11)

| Langkah | Detail |
|---|---|
| 1 | Bagian 0–3: konfigurasi, pasang ultralytics, ekstrak data, pemetaan kelas |
| 2 | Unduh weights dari `WEIGHTS_URL` (direct link GitHub Release), cocokkan `WEIGHTS_SHA256` |
| 3 | Konversi label test (hanya pemetaan + pemotongan ke batas gambar) |
| 4 | Hitung E1 dan mAP@50 dengan `val()`, lalu E2 pada `E2_CONF` |
| 5 | Cetak skor, F1 per kelas, dan waktu total |

Syarat lolos: runtime Colab T4 bersih, selesai < 10 menit, dua kali berturut-turut menghasilkan angka identik.

---

## 10. Artefak dan penamaan

| Artefak | Format nama | Disimpan di |
|---|---|---|
| Run | `<E0|A|B|C>_<model>[_bal]_s<seed>` | Drive `runs/` |
| Checkpoint | `epoch<N>.pt`, `last.pt`, `best.pt` | Drive |
| Weights final | `telepati_agridata_v<X.Y>.pt` | GitHub Release `v<X.Y>` |
| Log run | `results.csv`, `args.yaml` | `results/<RUN_ID>/` di repo |
| Evaluasi valid | `checkpoint_eval_valid.csv` | Drive run + `results/` di repo |
| Skor test | `final_test_scores.json` | Drive + README |
| Catatan audit | `audit_log.md` | `results/` di repo |

---

## 11. Urutan prioritas (Dimas)

1. Bagian 0–7 lolos uji `E0_smoke` di Colab → tag `v0.1-core` → Rahman mulai Run A.
2. Sambil Run A berjalan: periksa Run B (YOLO26s) dan C mulai normal di epoch awal.
3. 24 Okt: bagian 8–11 dipakai untuk seleksi, evaluasi test, dan latihan audit.

---

## 12. Antarmuka antar peran

| Dari | Ke | Yang diserahkan | Format |
|---|---|---|---|
| Dimas | Rahman, Maman | Notebook inti | Tag Git `v0.1-core` |
| Rahman, Maman | Dimas | Log run dan evaluasi valid run masing-masing | `results/<RUN_ID>/` lewat pull request |
| Rahman | Dimas | Tabel gabungan dan laporan per kelas | `results/checkpoint_eval_valid.csv` + ringkasan di pull request |
| Dimas | Maman | Weights final, `E2_CONF` | GitHub Release + nilai di README |
| Maman | Dimas | Hasil latihan audit | `results/audit_log.md` |
