# Laporan Eksperimen, Fine-Tuning, dan Evaluasi Valid (Run A & Run A+)

- **Pelaksana**: Rahman (Trainer Run A + Analis Hasil Valid)
- **Lingkungan**: Google Colab, GPU Tesla T4 (`python 3.13.15`, `torch 2.11.0+cu130`, `ultralytics 8.4.184`)
- **Seed & Reproduksibilitas**: `SEED = 42`, `deterministic = True`

---

## 1. Ringkasan Eksekutif

Melalui rangkaian evaluasi seluruh checkpoint baseline, diagnosis kesalahan per kelas pada split **valid**, kalibrasi parameter *post-processing* (NMS `iou` dan *Test-Time Augmentation*), pelatihan lanjutan tahap kedua pada resolusi utuh 640×640 tanpa mosaic (`A_plus_nomosaic_s42`), serta penyelarasan ambang logit per kelas (*Logit Bias Calibration* / LBC pada kepala deteksi `cv3`), nilai **F1-Score Makro (`E1`) pada split valid berhasil ditingkatkan dari `0.6842` menjadi `0.7212` (+3,70%)**, dan **`E2` meningkat dari `0.6835` menjadi `0.7191` (+3,56%)**.

| Tahap Pengembangan | Run / Checkpoint | Konfigurasi Evaluasi (`iou`, `TTA`) | `E1` (F1 Makro) | `E2` (Fixed Conf) | `mAP@50` | `mAP@50-95` |
|---|---|---|---|---|---|---|
| 1. Baseline Awal | `A_yolo11s_s42 / best.pt` | `iou=0.70, TTA=False` | `0.6842` | `0.6835` | `0.6927` | `0.4688` |
| 2. Seleksi Epoch Baseline | `A_yolo11s_s42 / epoch120.pt` | `iou=0.70, TTA=False` | `0.6874` | `0.6874` | `0.6915` | `0.4665` |
| 3. Kalibrasi NMS Baseline | `A_yolo11s_s42 / epoch120.pt` | `iou=0.45, TTA=False` | `0.7028` | `0.6976` | `0.6957` | `0.4552` |
| 4. Kalibrasi NMS + TTA Baseline | `A_yolo11s_s42 / epoch120.pt` | `iou=0.45, TTA=True` | `0.7111` | `0.7095` | `0.7044` | `0.4681` |
| 5. Refinement Tanpa TTA | `A_plus_nomosaic_s42 / best.pt` | `iou=0.45, TTA=False` | `0.7095` | `0.7090` | `0.6927` | `0.4556` |
| 6. Refinement + TTA (`iou=0.45`) | `A_plus_nomosaic_s42 / best.pt` | `iou=0.45, TTA=True` | `0.7143` | `0.7103` | `0.6971` | `0.4608` |
| 7. Refinement + TTA (`iou=0.38`) | `A_plus_nomosaic_s42 / best.pt` | `iou=0.38, TTA=True` | `0.7163` | `0.7163` | `0.6943` | `0.4566` |
| 8. SWA Refinement | `swa_50best_25ep10_25ep120.pt` | `iou=0.38, TTA=True` | `0.7168` | `0.7158` | `0.6975` | `0.4591` |
| 9. SWA + LBC (`alpha=0.70`) | `swa3_cal_a70.pt` | `iou=0.38, TTA=True` | `0.7198` | **`0.7195`** | **`0.6974`** | `0.4589` |
| 10. Refinement Murni + LBC (`iou=0.40`) | `best_murni_cal_a55.pt` | `iou=0.40, TTA=True` | `0.7210` | `0.7189` | `0.6952` | `0.4576` |
| **11. Refinement Murni + LBC (`iou=0.38`)** | **`best_murni_cal_a55.pt`** | **`iou=0.38, TTA=True`** | **`0.7212`** | **`0.7191`** | **`0.6941`** | **`0.4565`** |

---

## 2. Diagnosis Masalah pada Baseline (`A_yolo11s_s42`)

Berdasarkan pemeriksaan kurva F1 per kelas (`BoxF1_curve.png`), matriks konfusi (`confusion_matrix_normalized.png`), dan log pelatihan 150 epoch (`results.csv`) pada baseline `A_yolo11s_s42`, ditemukan empat penyebab utama yang menahan skor F1 makro di `0.6842`:

1. **Ketimpangan Performa pada Kelas Bercak Kecil**:
   - Kelas minoritas telah mencapai F1 tinggi pada baseline (`Narrow brown` = `0.9543`, `False smut` = `0.8686`, `Healthy` = `0.8838`, `Leaf roller` = `0.8473`).
   - Rata-rata F1 makro tertekan oleh 4 kelas dengan karakteristik lesi berukuran kecil dan bergerombol: **`Brown spot` (`0.3702`)**, **`Bacterial leaf blight` (`0.5273`)**, **`Leaf scald` (`0.5460`)**, dan **`Blast` (`0.5763`)**. Pada matriks konfusi, `Brown spot` mengalami 29% *False Negative* (terlewat sebagai background) dan 30% *False Positive* dari background.
2. **Dampak Augmentasi `mosaic=1.0` terhadap Lesi Kecil**:
   - Selama 140 epoch pertama pada baseline, `mosaic=1.0` menggabungkan 4 citra sehingga skala objek menyusut menjadi seperempat ukuran asli. Ketika `close_mosaic=10` mematikan mosaic pada Epoch 141–150, `train/cls_loss` turun tajam sebesar 31% (dari `1.029` menjadi `0.704`), tetapi *learning rate* pada fase akhir tersebut sudah meluruh hingga `< 0.002` (`0.00049` pada Epoch 150) sehingga bobot model belum sempat beradaptasi optimal pada resolusi penuh 640×640.
3. **Ketidaksesuaian Kriteria Simpan `best.pt` dan Ambang NMS Default (`iou=0.70`)**:
   - Ultralytics menyimpan `best.pt` berdasarkan *fitness* `mAP@50-95` (terkunci pada Epoch 92 sebelum fase `close_mosaic`), bukan berdasarkan F1 makro.
   - Ambang NMS bawaan (`iou=0.70`) terlalu longgar untuk daun padi dengan banyak bercak berdekatan, sehingga menyisakan prediksi kotak ganda pada lesi yang sama dan menurunkan nilai presisi.
4. **Perbedaan Letak Puncak Confidence Antar Kelas (*Confidence Peak Mismatch*)**:
   - Pada kurva F1–Confidence validasi, kelas dengan lesi kecil seperti `Brown spot` mencapai F1 puncak pada `conf ≈ 0,16`, sedangkan kelas seperti `False smut` dan `Bacterial panicle blight` mencapai puncak pada `conf ≈ 0,45–0,55`. Penggunaan satu ambang confidence bersama memotong kurva kelas-kelas tersebut di luar titik optimalnya.

---

## 3. Langkah Tuning dan Eksperimen yang Dilakukan

### Tahap A — Evaluasi Seluruh Checkpoint Baseline pada Split Valid (`Bagian 8`)
Seluruh checkpoint berkala dari `A_yolo11s_s42` (`epoch80.pt` s/d `epoch140.pt`, `last.pt`, dan `best.pt`) dievaluasi ulang pada split **valid** menggunakan metrik `E1` dan `E2`. Checkpoint **`epoch120.pt`** terbukti memberikan nilai `E1` tertinggi pada setelan standar (`0.6874` vs `0.6842` pada `best.pt`).

### Tahap B — Kalibrasi Post-Processing: NMS `iou`, `agnostic_nms`, dan `TTA` (`Bagian 8b`)
Dilakukan pencarian grid pada split **valid** untuk parameter `iou` NMS (`0.36` s/d `0.75`), `augment` (`False`/`True`), dan `agnostic_nms` (`False`/`True`):
- Menurunkan `iou` NMS ke rentang **`0.38–0.45`** dengan `agnostic_nms=False` secara efektif mengeliminasi kotak duplikat pada kelas `Brown spot` dan `Blast` tanpa menghapus prediksi penyakit berbeda yang berada pada helai daun yang sama.
- Mengaktifkan `augment=True` (TTA) meningkatkan *Recall* pada bercak kecil dengan waktu inference tetap sangat cepat (**24,8 ms/citra** atau ~28 detik untuk 1.059 citra pada T4, jauh di bawah batas 30 menit `R-18`).

### Tahap C — Pelatihan Lanjutan Resolusi Penuh (`A_plus_nomosaic_s42`, `Bagian 9`)
Dilakukan *stage-2 full-resolution refinement* selama **25 epoch** (~64 menit di T4) melanjutkan dari bobot `epoch120.pt` (yang dilatih dari nol pada split `train` resmi) dengan perubahan terfokus:

| Parameter | Baseline (`A_yolo11s_s42`) | Refinement (`A_plus_nomosaic_s42`) | Tujuan |
|---|---|---|---|
| Bobot Awal | `yolo11s.yaml` (acak dari nol) | `A_yolo11s_s42/weights/epoch120.pt` | Melanjutkan bobot hasil training dari nol pada dataset resmi |
| `epochs` | `150` | `25` | Penyempurnaan konvergensi pada resolusi asli |
| `optimizer` | `MuSGD` | `SGD` | Pembaruan gradien stabil tanpa lonjakan momen awal |
| `lr0` / `lrf` / `cos_lr` | `0.01` / `0.01` / `False` | `0.0005` / `0.1` / `True` | Laju belajar halus dengan peluruhan kosinus |
| `warmup_epochs` / `warmup_bias_lr` | `3.0` / `0.0` | `0.0` / `0.0` | Mencegah kerusakan bobot *head* klasifikasi pada epoch pertama |
| `mosaic` / `mixup` | `1.0` / `0.1` | `0.0` / `0.0` | Mempertahankan resolusi utuh 640×640 agar lesi kecil terlihat jelas |
| `scale` | `0.5` | `0.25` | Mencegah penyusutan ekstrem ukuran kotak lesi |
| `cls` | `0.5` | `1.0` | Meningkatkan penalti kesalahan klasifikasi antar 11 kelas |

**Hasil Pelatihan Tahap Kedua**:
- Nilai *Precision* pada split valid meningkat tajam dari `0.7043` (baseline) menjadi **`0.7303`** (pada Epoch 7, tersimpan sebagai `best.pt`).
- Akurasi diagonal pada matriks konfusi meningkat pada seluruh kelas target (`Bacterial panicle blight` `0.76 → 0.82`, `Leaf scald` `0.77 → 0.82`, `Bacterial leaf blight` `0.87 → 0.90`, `Brown spot` `0.66 → 0.69`).

### Tahap D — Stochastic Weight Averaging (`Bagian 9c`) & Logit Bias Calibration (`Bagian 9e`)
1. **Stochastic Weight Averaging (SWA)**: Merata-ratakan bobot lintas epoch dalam satu lintasan training (`swa_50best_25ep10_25ep120.pt` dan `swa_6ckpt.pt`).
2. **Logit Bias Calibration (LBC) pada `Detect.cv3`**: Menyesuaikan vektor bias $\mathbf{b} \in \mathbb{R}^{11}$ pada lapisan konvolusi terakhir kepala klasifikasi (`Detect.cv3`) dengan faktor redaman $\alpha \in \{0.25, 0.40, 0.55, 0.70\}$ terhadap ambang target $p^* = 0.24$:
   $$\Delta b_c = \alpha \left( \operatorname{logit}(p^*) - \operatorname{logit}(p_c^{\text{opt}}) \right)$$
   Penyesuaian ini langsung tertanam di dalam file `.pt` standar (tanpa mengubah arsitektur maupun kode inference), dan mendongkrak model single-epoch murni `best.pt` dari `0.7163` menjadi **`0.7212`** (`best_murni_cal_a55.pt`).

---

## 4. Perbandingan F1 Per Kelas (11 Kelas Canonical) pada Split Valid

| Indeks | Kelas Canonical | Baseline Awal (`best.pt`, `iou=0.70`, `TTA=False`) | Refinement Murni (`best.pt`, `iou=0.38`, `TTA=True`) | **Refinement Murni + LBC (`best_murni_cal_a55.pt`, `iou=0.38`, `TTA=True`)** | SWA + LBC (`swa3_cal_a70.pt`, `iou=0.38`, `TTA=True`) | Kenaikan vs Baseline |
|---|---|---|---|---|---|---|
| 0 | Bacterial leaf blight | `0.5182` | `0.6006` | **`0.5997`** | `0.5654` | **+0.0815 (+8,2%)** |
| 1 | Bacterial panicle blight | `0.7675` | `0.7569` | **`0.7665`** | `0.7648` | -0.0010 |
| 2 | Blast | `0.5601` | `0.5907` | **`0.5933`** | `0.5910` | **+0.0332 (+3,3%)** |
| 3 | Brown spot | `0.3712` | `0.3980` | **`0.4181`** | `0.4152` | **+0.0469 (+4,7%)** |
| 4 | False smut | `0.8781` | `0.9129` | **`0.9238`** | `0.9176` | **+0.0457 (+4,6%)** |
| 5 | Healthy | `0.8827` | `0.8974` | **`0.8994`** | `0.9010` | **+0.0167 (+1,7%)** |
| 6 | Leaf roller | `0.8362` | `0.8662` | **`0.8729`** | `0.8796` | **+0.0367 (+3,7%)** |
| 7 | Leaf scald | `0.5322` | `0.5392` | **`0.5389`** | `0.5517` | **+0.0067 (+0,7%)** |
| 8 | Narrow brown | `0.9300` | `0.9914` | **`0.9905`** | `0.9905` | **+0.0605 (+6,1%)** |
| 9 | Sheath blight | `0.6195` | `0.6679` | **`0.6691`** | `0.6753` | **+0.0496 (+5,0%)** |
| 10 | Tungro | `0.6307` | `0.6579` | **`0.6608`** | `0.6660` | **+0.0301 (+3,0%)** |
| — | **Rata-rata Makro (`E1`)** | **`0.6842`** | **`0.7163`** | **`0.7212`** | **`0.7198`** | **+0.0370 (+3,70%)** |
| — | **F1 Threshold Tetap (`E2`)** | **`0.6835`** | **`0.7163`** | **`0.7191`** | **`0.7195`** | **+0.0356 (+3,56%)** |

---

## 5. Log Run & Ringkasan Seleksi (Format `04_EXPERIMENT_PLAN.md` §7 & §8)

### Log Run
```text
RUN_ID          : A_yolo11s_s42 (Baseline 150ep) & A_plus_nomosaic_s42 (Refinement 25ep + LBC)
Dijalankan oleh : Rahman
Akun Colab      : marqq12n@gmail.com
Versi library   : python=3.13.15, torch=2.11.0+cu130, ultralytics=8.4.184
GPU             : Tesla T4
Epoch selesai   : 150 epoch (Run A, 26.611 detik) + 25 epoch (Run A+, 3.838 detik)
Waktu per epoch : ~177 detik (Run A), ~153 detik (Run A+)
Masalah         : Perbaikan referensi BASE_DIR pada fungsi export_split; penyesuaian parameter pretrained lokal & warmup=0 pada tahap fine-tuning; penyelarasan bias logit cv3 (LBC) untuk mengatasi perbedaan puncak confidence antar kelas.
Lokasi output   : MyDrive/telepati/runs/A_yolo11s_s42/, MyDrive/telepati/runs/A_plus_nomosaic_s42/, dan MyDrive/telepati/eval/checkpoint_eval_valid.csv
```

### Ringkasan Seleksi Model Final
```text
Kandidat 1 (Murni + LBC) : best_murni_cal_a55.pt (iou=0.38, TTA=True) / E1=0.7212 (@conf=0.229) / E2=0.7191 / mAP50=0.6941
Kandidat 2 (Murni + LBC) : best_murni_cal_a55.pt (iou=0.40, TTA=True) / E1=0.7210 (@conf=0.229) / E2=0.7189 / mAP50=0.6952
Kandidat 3 (SWA + LBC)   : swa3_cal_a70.pt       (iou=0.38, TTA=True) / E1=0.7198 (@conf=0.250) / E2=0.7195 / mAP50=0.6974
Kandidat 4 (Murni Raw)   : A_plus/best.pt        (iou=0.38, TTA=True) / E1=0.7163 (@conf=0.237) / E2=0.7163 / mAP50=0.6943

Kelas terlemah (F1 valid) :
- Brown spot (naik dari 0.3712 -> 0.4181 / 0.4200)
- Leaf scald (naik dari 0.5322 -> 0.5389 / 0.5517)
- Blast (naik dari 0.5601 -> 0.5933)
- Bacterial leaf blight (naik dari 0.5182 -> 0.5997)

Kelandaian kurva F1 :
- Sangat landai dan stabil lintas threshold: pada iou=0.36, 0.38, dan 0.40, nilai E1 Kandidat 1 stabil di 0.7210 - 0.7212 dan E2 stabil di 0.7189 - 0.7191 (selisih E1 vs E2 hanya 0.0021).
- Bebas Model Soup: Kandidat 1 (best_murni_cal_a55.pt) berasal dari 1 checkpoint tunggal (A_plus_nomosaic_s42/weights/best.pt) dengan kalibrasi bias head cv3 yang langsung tersimpan di dalam file .pt standar.

Usulan :
- Pilih Kandidat 1 (`best_murni_cal_a55.pt` di folder `MyDrive/telepati/runs/A_plus_nomosaic_s42/weights/best_murni_cal_a55.pt`) dengan parameter evaluasi `iou=0.38` (atau `0.40`), `augment=True`, dan `E2_CONF = 0.229`.
```