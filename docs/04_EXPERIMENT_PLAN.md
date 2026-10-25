---
doc_id: EXPERIMENT_PLAN
judul: "Rencana Eksperimen — Run A, B, C"
versi: 1.0
tanggal: 2026-10-23
status: AKTIF
pemilik: Dimas (desain), Rahman dan Maman (eksekusi)
bergantung_pada: [01_PRD.md, 02_SYSTEM_DESIGN.md]
---

# Rencana Eksperimen

> **Untuk manusia dan AI.** Dokumen ini mengatur *apa* yang dilatih, *siapa* yang menjalankan, dan *bagaimana* hasilnya dibandingkan. Setiap run punya ID tetap. Isi konfigurasi hanya diubah oleh Dimas (D-11).

---

## 1. Prinsip

1. **Satu perubahan per run.** Semua run memakai data, seed, resolusi, jumlah epoch, dan augmentasi yang sama, kecuali satu hal yang diuji.
2. **Paralel di akun berbeda.** Setiap run di akun Google Colab yang berbeda supaya kuota GPU tidak berebut.
3. **Ukur sebelum berkomitmen.** Jumlah epoch ditetapkan dari uji waktu (E0), bukan ditebak.
4. **Seleksi hanya di valid.** Test tidak dilihat sampai model final dipilih (R-20b).
5. **Semua dicatat.** Setiap run menyimpan `results.csv`, `args.yaml`, dan checkpoint berkala.

---

## 2. Daftar run

| ID | Model | Perbedaan yang diuji | Pertanyaan | Dijalankan oleh | Akun |
|---|---|---|---|---|---|
| `E0_smoke` | YOLO11s (`.yaml`) | Uji alur: 2 epoch, 5% data | Pipeline jalan? Berapa menit per epoch? | Dimas | Dimas |
| **`A_yolo11s_s42`** | YOLO11s (`.yaml`) | Baseline | Seberapa bagus pilihan paling aman? | Rahman | Rahman |
| **`B_yolo26s_s42`** | YOLO26s (`.yaml`) | Arsitektur | Apakah fitur lesi kecil YOLO26 membantu dari nol? | Maman | Maman |
| **`C_yolo11s_bal_s42`** | YOLO11s (`.yaml`) | Repeat factor sampling, `t = 0.10` (kelas langka diulang sekitar 2×) | Apakah F1 makro naik lewat kelas langka? | Dimas | Dimas |
| **`A_fit_yolo11s_s42`** | YOLO11s, mulai dari `best.pt` Run A | 640, 50 epoch, SGD `lr0=0.002`, mosaic/mixup/variasi warna dimatikan | Apakah model yang lebih pas dengan data train menaikkan F1 makro valid? | Dimas | GPU yang tersedia |
| **`A800_yolo11s_s42`** | YOLO11s, mulai dari `best.pt` Run A | Resolusi 800, 40 epoch, SGD `lr0=0.002` | Apakah resolusi lebih tinggi memperbaiki lesi kecil (Brown spot, Leaf scald, Blast)? | Dimas | Dimas (shortcut folder Run A) |

`RUN_ID` di atas adalah kunci tabel `RUNS` di bagian 7 notebook. Run C dipindah ke akun Dimas supaya Rahman tidak perlu menunggu Run A selesai. Setelah hasil Run A keluar, akun Dimas dipakai untuk `A800_yolo11s_s42` (D-16); Run C dijalankan hanya jika ada GPU yang menganggur.

Kalau waktu tersisa setelah seleksi: **A+** = melanjutkan training run pemenang beberapa puluh epoch, dicek ulang di valid.

---

## 3. Setelan bersama

| Parameter | Nilai |
|---|---|
| Data training | Split train saja (10.133 gambar) |
| Data seleksi | Split valid |
| `imgsz` | 640 |
| `epochs` | Dari E0, target sekitar 150 |
| `batch` | 16 |
| `seed` | 42 |
| `deterministic` | True |
| `pretrained` | False |
| `save_period` | 10 |
| `workers` | 2 |
| Augmentasi | Sama untuk semua run (lihat `02_SYSTEM_DESIGN.md` §6) |

---

## 4. Anggaran waktu

Perkiraan kasar, **wajib dikalibrasi dengan E0**:

| Model | Perkiraan per epoch di T4 | 150 epoch |
|---|---|---|
| YOLO11s | sekitar 3–4 menit | sekitar 7,5–10 jam |
| YOLO26s | belum diketahui, diukur di E0/awal Run B | — |

- Colab gratis maksimal sekitar 12 jam per sesi dan bisa putus. Bagi run jadi **segmen di bawah sekitar 8 jam**, resume dari `last.pt` di Drive.
- Kalau E0 menunjukkan 150 epoch tidak muat sampai 24 Okt sore, turunkan epoch **untuk semua run** secara sama.

---

## 5. Protokol menjalankan run

1. Buka notebook dari repo dengan tag `v0.1-core` (atau versi yang ditentukan Dimas) di Colab.
2. Pastikan zip dataset ada di `MyDrive/telepati/data/telepati-agridata-raw.zip`.
3. Di sel konfigurasi (bagian 0): `MODE = "full"`, isi `RUN_ID` sesuai tabel §2. **Jangan ubah sel lain.**
4. Runtime → Change runtime type → **T4 GPU**, lalu Runtime → Run all.
5. Pastikan output run tertulis ke `MyDrive/telepati/runs/<RUN_ID>/`.
6. Kalau sesi putus: buka ulang, Run all lagi dengan `RUN_ID` yang sama. Notebook otomatis melanjutkan dari `last.pt`.
7. Setelah training selesai, bagian 8 otomatis mengevaluasi semua checkpoint run itu di valid dan menulis `checkpoint_eval_valid.csv` ke folder run.
8. Salin `results.csv`, `args.yaml`, dan `checkpoint_eval_valid.csv` ke `results/<RUN_ID>/` di repo lewat branch dan pull request.
9. Isi log run (§7).

Larangan selama run: mengubah konfigurasi di tengah run, menambah data valid/test, mengolah dataset lewat API LLM, melihat skor test.

---

## 6. Protokol evaluasi dan seleksi

1. Setiap trainer menjalankan bagian 8 untuk run-nya sendiri (otomatis setelah training). Fungsi evaluasinya sama untuk semua run.
2. Rahman menggabungkan CSV ketiga run ke `results/checkpoint_eval_valid.csv`: `run_id, checkpoint, E1, mAP50, F1 per kelas`.
3. Rahman menulis ringkasan di pull request: tiga kandidat teratas, kelas terlemah, dan kelandaian kurva F1.
4. Dimas memutuskan model final berdasarkan aturan di `02_SYSTEM_DESIGN.md` §8.
5. Setelah keputusan dikunci, test dievaluasi **sekali** (S-10).

Kriteria seleksi, berurutan:
1. E1 di valid tertinggi.
2. Kurva F1 landai di sekitar puncak, dan E2 dekat dengan E1.
3. Tidak ada kelas yang runtuh (F1 per kelas sangat rendah dibanding run lain).
4. Kalau dua kandidat setara: pilih yang lebih sederhana (YOLO11s), karena lebih pasti termuat saat audit.

---

## 7. Template log run

Salin ke deskripsi pull request setiap run.

```text
RUN_ID          :
Dijalankan oleh :
Akun Colab      :
Versi notebook  : (tag/commit)
RUN_ID (tabel RUNS) :
Versi library   : python=, torch=, ultralytics=
GPU             : T4
Mulai / selesai :
Jumlah segmen   :
Epoch selesai   :
Waktu per epoch :
Masalah         :
Lokasi output   : MyDrive/telepati/runs/<RUN_ID>/
Catatan         :
```

---

## 8. Template ringkasan seleksi

```text
Kandidat 1 : RUN_ID / epoch / E1 / E2 / mAP50
Kandidat 2 :
Kandidat 3 :
Kelas terlemah (F1 valid) :
Kelandaian kurva F1       :
Usulan                    :
Keputusan Dimas           :
```

---

## 9. Tahap penyempurnaan 25 Oktober (branch `dimstest`)

Titik awal adalah v1.2: bobot v1.0 (A_fit epoch 20) dengan NMS 0,45 per kelas. Skornya E1 valid 0,7113, F1 uji silang valid 0,7066, dan E1 test 0,7023.

Laporan Rahman (`A_plus_nomosaic_s42`) mencapai E1 valid 0,7212. Angka itu gabungan tiga hal: TTA, NMS 0,38, dan kalibrasi bias per kelas yang dihitung dan dinilai pada valid yang sama. Tanpa TTA dan kalibrasi, pada NMS 0,45, `A_plus/best.pt` mendapat 0,7095, setara dengan v1.0 (0,7113). Jadi selisihnya datang dari TTA dan kalibrasi in-sample, bukan dari bobot yang lebih baik. Kalibrasi in-sample sudah terbukti tidak bertahan di test (v1.1). Karena itu semua kandidat di tahap ini dibandingkan dengan **F1 uji silang dua paruh valid** dari bagian 9 notebook, bukan dengan E1 valid saja.

Efek TTA tergantung model. Pada bobot v1.0, TTA menurunkan F1 uji silang: 0,7065 menjadi 0,7029 (NMS 0,45 per kelas) dan 0,7051 (NMS 0,45 lintas kelas), dengan Brown spot turun 0,019 karena TTA Ultralytics memperkecil gambar. Jadi TTA tidak otomatis dipakai; grid di bagian 9 menilainya untuk setiap kandidat.

### Pembagian kerja

| Siapa | Yang dijalankan | Waktu T4 | Output |
|---|---|---|---|
| Rahman | `RUN_ID = "A_fit2_yolo11s_s42"`, `MODE = "full"` | sekitar 30 menit | `runs/A_fit2_yolo11s_s42/` |
| Maman | `RUN_ID = "A_fit800_yolo11s_s42"`, `MODE = "full"` | sekitar 70 menit | `runs/A_fit800_yolo11s_s42/` |
| Dimas | Bagian 9 dengan `SELECTED_WEIGHTS = RELEASE_WEIGHTS` (grid TTA x NMS untuk v1.0) | sekitar 10 menit | tabel `infer_table` |
| Dimas | Bagian 9b (`REFINE_WEIGHTS = True`) setelah bahan soup tersedia | sekitar 20 menit | `runs/soup/soup.pt`, `soup_summary.json` |
| Dimas | Bagian 9 dengan `SELECTED_WEIGHTS` = kandidat terbaik (soup atau `A_fit800/best.pt`) | sekitar 10 menit | `E2_CONF` dan setelan inference final |

Kedua run baru mengunduh bobot rilis v1.0 sendiri, jadi tidak butuh folder run lain. Untuk bagian 9b, bahan soup harus ada sebagai **file asli** (bukan shortcut) di Drive Dimas dengan struktur `runs/<RUN_ID>/weights/<file>.pt`:

- `A_plus_nomosaic_s42/weights/best.pt` (dari Rahman)
- `A_fit2_yolo11s_s42/weights/{best,last,epoch4,epoch8}.pt` (dari Rahman)
- `A_fit_yolo11s_s42/weights/{epoch10,epoch40}.pt` (sudah ada)

Rahman juga mengirim `args.yaml` dan `results.csv` run `A_plus_nomosaic_s42` lewat pull request ke `results/A_plus_nomosaic_s42/`, supaya entri `A_plus_nomosaic_s42` di tabel `RUNS` bisa dicocokkan dengan setelan aslinya.

### Aturan keputusan

1. Kandidat: v1.2 (tanpa TTA), v1.0 + TTA, soup (dengan setelan inference terbaiknya), dan `A_fit800/best.pt`.
2. Setiap kandidat dinilai di bagian 9 dengan grid TTA x NMS. Yang dibandingkan adalah baris terbaik tiap kandidat pada kolom `F1 uji silang`.
3. Kandidat baru dipakai hanya jika F1 uji silangnya minimal **+0,003** di atas v1.2 (0,7066). Kalau tidak ada yang lolos, v1.2 tetap final.
4. Keputusan dicatat sebagai D-21 di PRD dan di-push **sebelum** test dievaluasi. Test dievaluasi sekali, dilaporkan apa adanya, lalu latihan audit di Colab T4.
5. Kalibrasi bias per kelas (`CALIBRATE`) tidak dipakai untuk memilih model.
