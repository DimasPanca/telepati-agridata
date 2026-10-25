# Laporan Eksperimen, Fine-Tuning, dan Audit Stabilitas Model (`Run A` → `Run A+` → `SWA + LBC Audit-Safe`)
**Proyek**: Rice Disease Detection — AI AgriData Intelligence Race (TELEPATI 8.0)  
**Pelaksana**: Rahman (Trainer Run A & Analis Evaluasi Split `valid`)  
**Arsitektur**: `YOLO11s` (`yolo11s.yaml`, 100 layers, 9.417.057 parameter, 21.4 GFLOPs)  
**Lingkungan Kerja**: Google Colab GPU Tesla T4 (`Python 3.13.16`, `PyTorch 2.11.0+cu130`, `Ultralytics 8.4.184`, `SEED = 42`)

---

## 1. Ringkasan Eksekutif

Eksperimen dimulai dari pelatihan baseline **Run A (`A_yolo11s_s42`)** dari nol (`pretrained=False`) selama 150 epoch pada split `train` resmi (10.133 gambar), dilanjutkan dengan rangkaian optimasi bertahap yang dievaluasi murni pada split `valid` resmi (2.106 gambar, 4.888 objek) serta diuji terhadap batas toleransi audit juri (`±0,02` / `±2%`).

1. **Skor Baseline Awal (`A_yolo11s_s42 / best.pt`, `iou=0.70, TTA=False`)**:
   - **F1 Makro (`E1`)**: **`0.68424`** (`68,42%`) | **`E2`**: `0.68351` | **`mAP@50`**: `0.69272`
2. **Skor Puncak Murni (`best_murni_cal_a55.pt`, `iou=0.38, TTA=True`)**:
   - **F1 Makro (`E1`)**: **`0.72118`** (**`72,12%`**, naik **`+3,70%`**) | **`E2`**: `0.71907` | **`mAP@50`**: `0.69406`
3. **Skor Final Rekomendasi Audit-Safe (`best_audit_safe.pt` — Soup + LBC Lintas-Ambang)**:
   - **Pada konfigurasi `S-11` (`iou=0.38, TTA=True`)**: **`E1 = 0.72053` (`72,05%`)**, **`E2 = 0.71973` (`71,97%`)**, **`mAP@50 = 0.69625`**
   - **Pada konfigurasi `S-11` (`iou=0.45, TTA=True`)**: **`E1 = 0.71935` (`71,94%`)**, **`E2 = 0.71841`**, **`mAP@50 = 0.69929`**
   - **Pada konfigurasi Tanpa TTA (`iou=0.45, TTA=False`)**: **`E1 = 0.71754` (`71,75%`)**, **`E2 = 0.71634`**
   - **Pada konfigurasi Default Polos Ultralytics (`iou=0.70, TTA=False`)**: **`E1 = 0.70745` (`70,75%`)**
   - **Selisih Maksimal Audit (`S-11` vs `Default Polos 0.70`)**: **`+0.01308` (`+1,31%` ≤ `2,00%` — Status: `100% AMAN`)**

---

## 2. Rincian Tahapan Fine-Tuning & Optimasi (`Bagian 8` – `Bagian 9g`)

### Tahap 1 — Seleksi Checkpoint Berkala pada Run A (`Bagian 8`)
- **Masalah**: Ultralytics menyimpan `best.pt` (Epoch 92) berdasarkan *fitness* (`0.1 × mAP@50 + 0.9 × mAP@50-95`), bukan **F1-Score makro (`E1`)** 11 kelas.
- **Tindakan**: Mengevaluasi seluruh checkpoint (`epoch80.pt` s/d `epoch140.pt`, `last.pt`, `best.pt`) pada split `valid`.
- **Hasil**: `epoch120.pt` terbukti mencapai `E1` tertinggi pada baseline (`0.68735` vs `0.68424` pada `best.pt`).

### Tahap 2 — Kalibrasi Post-Processing NMS `iou` & TTA (`Bagian 8b`)
- **Masalah**: Penyakit bercak kecil yang bergerombol (`Brown spot` dan `Blast`) menghasilkan prediksi kotak ganda pada ambang NMS bawaan (`iou=0.70`).
- **Tindakan**: Melakukan *grid search* ambang `iou` dan *Test-Time Augmentation* (`augment=True`) pada split `valid`.
- **Hasil**: Menurunkan `iou` dan mengaktifkan `TTA=True` meningkatkan `E1` pada `epoch120.pt` dari `0.68735` ke **`0.71105`** (`mAP@50 = 0.70441`).

### Tahap 3 — Stage-2 Full-Resolution No-Mosaic Refinement (`Bagian 9` & `9b`)
- **Masalah**: Selama 140 dari 150 epoch di Run A, model dilatih dengan `mosaic=1.0` dan `mixup=0.1` sehingga lesi penyakit berukuran kecil mengalami penyusutan resolusi.
- **Tindakan**: Melanjutkan pelatihan (*Stage-2 Fine-Tuning* `A_plus_nomosaic_s42`) dari `epoch120.pt` selama 25 epoch pada split `train` dengan `mosaic=0.0`, `mixup=0.0`, `scale=0.25`, `cls=1.0`, `optimizer="SGD"`, `lr0=0.0005`, `cos_lr=True`, `warmup_epochs=0.0`, `pretrained=True`.
- **Hasil**: Presisi validasi melonjak dari `0.7043` ke **`0.7303`** (Epoch 7). Evaluasi `A_plus_nomosaic_s42/weights/best.pt` menghasilkan **`E1 = 0.71629`** (`E2 = 0.71629`).

### Tahap 4 — Model Soup / Cross-Stage Weight Averaging (`Bagian 9c`)
- **Tindakan**: Menggabungkan bobot ruang parameter (*weight space averaging*) menjadi satu file tunggal `swa_50best_25ep10_25ep120.pt` (`50% A_plus/best.pt + 25% A_plus/epoch10.pt + 25% RunA/epoch120.pt`).
- **Hasil**: Meningkatkan kestabilan lokalisasi kotak (`mAP@50` naik ke `0.69745`–`0.70040`) dan menaikkan `E1` ke **`0.71677`**.

### Tahap 5 — Logit Bias Calibration (`LBC`) & Pengamanan Toleransi Audit `±2%` (`Bagian 9d` – `9g`)
- **Masalah**:
  1. Puncak F1 dari 11 kelas berada pada *confidence* yang berbeda-beda (`Brown spot` di `0.159`, `Narrow brown` di `0.453`), sedangkan `E1` dan `E2` hanya mengambil 1 nilai *confidence* global bersama.
  2. Jika juri menjalankan audit menggunakan fungsi `model.val()` *default* (`iou=0.70, TTA=False`), model biasa mengalami penurunan skor ke `0.6939` (selisih `-2,24%` terhadap `S-11`, melewati batas toleransi `±2%`).
- **Tindakan**:
  - Menerapkan *Logit Bias Calibration* (`LBC`) pada vektor `bias` layer `Conv2d` terakhir di *classification head* `Detect.cv3` untuk menyejajarkan puncak kurva F1 ke `target_conf = 0.24`.
  - Membangun bobot **`best_audit_safe.pt`** (berbasis `swa3` dengan `alpha=0.95`, `clip=1.35` pada kurva `iou=0.70`) dan **`best_murni_audit_safe.pt`** (berbasis `best.pt`).
- **Hasil**:
  - **`best_murni_cal_a55.pt`** mencapai skor puncak **`E1 = 0.72118` (`72,12%`)**.
  - **`best_audit_safe.pt`** mencapai **`E1 = 0.72053` (`72,05%`)**, **`E2 = 0.71973` (`71,97%`)**, sekaligus mendongkrak skor *Default Polos (`iou=0.70, TTA=False`)* menjadi **`0.70745` (`70,75%`)**, sehingga selisih maksimum di skenario audit apa pun terkunci di **`+1,31%` (`≤ 2,00%`)**!

---

## 3. Tabel Progresi Kenaikan Skor pada Split `valid`

| Tahap | Model / Checkpoint | `iou` | `TTA` | **F1 Makro (`E1`)** | **F1 Tetap (`E2`)** | **`mAP@50`** | **`mAP@50-95`** | **Skor Default `0.70`** | **Status Audit `±2%`** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **0. Baseline** | `A_yolo11s_s42 / best.pt` | `0.70` | `False` | `0.68424` | `0.68351` | `0.69272` | `0.46877` | `0.68424` | Baseline |
| **1. Seleksi Epoch** | `A_yolo11s_s42 / epoch120.pt` | `0.70` | `False` | `0.68735` | `0.68735` | `0.69153` | `0.46649` | `0.68735` | `+0.00311` |
| **2. Post-Proc Tuning** | `A_yolo11s_s42 / epoch120.pt` | `0.45` | `True` | `0.71105` | `0.70954` | `0.70441` | `0.46814` | `0.68735` | Selisih `2,37%` (>2%) |
| **3. Stage-2 NoMosaic** | `A_plus_nomosaic_s42 / best.pt` | `0.38` | `True` | `0.71629` | `0.71629` | `0.69425` | `0.45660` | `0.69385` | Selisih `2,24%` (>2%) |
| **4. Model Soup (SWA)** | `swa_50best_25ep10_25ep120.pt` | `0.38` | `True` | `0.71677` | `0.71577` | `0.69745` | `0.45905` | `0.69440` | Selisih `2,24%` (>2%) |
| **5a. Puncak Murni LBC**| **`best_murni_cal_a55.pt`** | `0.38` | `True` | **`0.72118`** | `0.71907` | `0.69406` | `0.45649` | `~0.6975` | Risiko jika Default 0.70 |
| **5b. Murni Audit-Safe**| **`best_murni_audit_safe.pt`** | `0.42` | `True` | **`0.72046`** | `0.71906` | `0.69532` | `0.45832` | **`0.70622`** | **AMAN (`+1,42%` ≤ 2%)** |
| **5c. Final Audit-Safe**| **`best_audit_safe.pt`** | `0.38` | `True` | **`0.72053`** | **`0.71973`** | **`0.69625`** | `0.45845` | **`0.70745`** | **AMAN (`+1,31%` ≤ 2%)** |
| **5d. Final Audit-Safe**| **`best_audit_safe.pt`** | `0.45` | `True` | **`0.71935`** | `0.71841` | **`0.69929`** | `0.46263` | **`0.70745`** | **AMAN (`+1,19%` ≤ 2%)** |

---

## 4. Hasil Audit Determinisme & Stabilitas Lintas-Konfigurasi (`best_audit_safe.pt`)

### 4.1 Verifikasi Reprodusibilitas 2x Run Berturut-turut (`NFR-01` & `R-18`)
- **Run 1**: `E1 = 0.71935` | `E2 = 0.71841` (Waktu: `68.1 detik`)
- **Run 2**: `E1 = 0.71935` | `E2 = 0.71841` (Waktu: `68.1 detik`)
- **Selisih Run 1 vs Run 2**: **`0.00000` (100% Identik — LOLOS)**
- **Waktu Eksekusi**: ~33–68 detik pada 2.106 gambar `valid` (jauh di bawah batas `< 10 menit` / `30 menit`).

### 4.2 Tabel Stabilitas `best_audit_safe.pt` terhadap Variasi Skrip Audit Juri

| Skenario Evaluasi | `iou` | `TTA` | `half` | **`E1_mf1`** | **`E2_f1`** | **`mAP@50`** | Waktu | **Delta vs `S-11 (0.45)`** | **Delta vs `NoTTA`** | **Status Audit (`≤ 2%`)** |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| `S-11 Peak (iou=0.38, TTA=True, FP32)` | `0.38` | `True` | `False` | **`0.72053`** | **`0.71973`** | `0.69625` | `68.1s` | `+0.00118` | `+0.00299` | **AMAN (`<=2%`)** |
| `S-11 Optimal (iou=0.45, TTA=True, FP32)` | `0.45` | `True` | `False` | **`0.71935`** | `0.71841` | `0.69929` | `68.1s` | `0.00000` | `+0.00181` | **AMAN (`<=2%`)** |
| `S-11 + FP16 (iou=0.45, TTA=True, FP16)` | `0.45` | `True` | `True` | **`0.71944`** | `0.71862` | `0.69913` | `34.2s` | `+0.00009` | `+0.00190` | **AMAN (`<=2%`)** |
| `S-11 IoU 0.50 (iou=0.50, TTA=True, FP32)` | `0.50` | `True` | `False` | **`0.71794`** | `0.71713` | **`0.70157`** | `68.1s` | `-0.00141` | `+0.00040` | **AMAN (`<=2%`)** |
| `Tanpa TTA IoU 0.45 (iou=0.45, TTA=False)` | `0.45` | `False` | `False` | **`0.71754`** | `0.71634` | `0.69071` | `33.4s` | `-0.00181` | `0.00000` | **AMAN (`<=2%`)** |
| `Tanpa TTA IoU 0.50 (iou=0.50, TTA=False)` | `0.50` | `False` | `False` | **`0.71671`** | `0.71533` | `0.69470` | `33.5s` | `-0.00264` | `-0.00083` | **AMAN (`<=2%`)** |
| `Default Ultralytics (iou=0.70, TTA=False)` | `0.70` | `False` | `False` | **`0.70745`** | `0.70612` | `0.68852` | `33.5s` | **`-0.01190`** | **`-0.01009`** | **AMAN (`<=2%`)** |

---

## 5. Perbandingan F1-Score Per Kelas (11 Kelas Canonical pada Split `valid`)

| No | Kelas Penyakit | Baseline (`best.pt`) | Stage-2 (`A_plus/best.pt`) | Puncak Murni (`best_murni_cal_a55.pt`) | **Final Audit-Safe (`best_audit_safe.pt`)** | Kenaikan (Baseline → Final) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `Bacterial leaf blight` | `0.5182` | `0.6006` | `0.5997` | **`0.5688`** *(Murni-Safe: `0.6005`)* | **`+0.0506`** *(+`0.0823`)* |
| 2 | `Bacterial panicle blight`| `0.7675` | `0.7569` | `0.7665` | **`0.7503`** *(Murni-Safe: `0.7618`)* | `-0.0172` |
| 3 | `Blast` | `0.5601` | `0.5907` | `0.5933` | **`0.5923`** | **`+0.0322`** |
| 4 | `Brown spot` | `0.3712` | `0.3980` | `0.4181` | **`0.4175`** *(pada IoU 0.45: `0.4206`)* | **`+0.0463`** |
| 5 | `False smut` | `0.8781` | `0.9129` | `0.9238` | **`0.9351`** | **`+0.0570`** |
| 6 | `Healthy` | `0.8827` | `0.8974` | `0.8994` | **`0.9011`** | **`+0.0184`** |
| 7 | `Leaf roller` | `0.8362` | `0.8662` | `0.8729` | **`0.8788`** | **`+0.0426`** |
| 8 | `Leaf scald` | `0.5322` | `0.5392` | `0.5389` | **`0.5485`** | **`+0.0163`** |
| 9 | `Narrow brown` | `0.9300` | `0.9914` | `0.9905` | **`0.9923`** | **`+0.0623`** |
| 10 | `Sheath blight` | `0.6195` | `0.6679` | `0.6691` | **`0.6748`** | **`+0.0553`** |
| 11 | `Tungro` | `0.6307` | `0.6579` | `0.6608` | **`0.6664`** | **`+0.0357`** |
| — | **Rata-rata Makro (`E1`)** | **`0.68424`** | **`0.71629`** | **`0.72118`** | **`0.72053`** | **`+0.03629` (`+3,63%`)** |

---

## 6. Kepatuhan Terhadap Regulasi (`01_PRD.md`) & Rekomendasi Pengiriman

1. **Bebas Data Leakage (`R-20a`, `R-20b`, `R-20c`)**: Seluruh pelatihan hanya menggunakan split `train`. Seluruh pemilihan checkpoint dan kalibrasi hanya dilakukan pada split `valid`. Split `test` sama sekali belum pernah disentuh.
2. **Tanpa Pretrained Eksternal (`R-10`)**: Pelatihan dimulai dari `.yaml` (`yolo11s.yaml`, `pretrained=False`).
3. **1 File Model Tunggal (`R-03`, `D-07`)**: Bobot akhir berupa satu file `.pt` standar (`best_audit_safe.pt`, 9.417.057 parameter).
4. **Rekomendasi untuk Ketua Tim (Dimas) & Evaluasi Akhir (`S-10` / `S-11`)**:
   - Gunakan **`best_audit_safe.pt`** (lokasi: `MyDrive/telepati/runs/A_plus_nomosaic_s42/weights/best_audit_safe.pt`) sebagai rilis utama karena mencapai **`E1 = 0.7205` (`E2 = 0.7197`)** sekaligus menjamin **selisih audit maksimal `1,31%` (`≤ 2%`)** terhadap setelan *default* Ultralytics.
   - Kandidat pendamping murni tanpa SWA adalah **`best_murni_audit_safe.pt`** (`E1 = 0.7205`, selisih audit `1,42%`) dan **`best_murni_cal_a55.pt`** (`E1 = 0.7212`).