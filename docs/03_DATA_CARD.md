---
doc_id: DATA_CARD
judul: "Data Card — Agritech Data (Dataset Resmi TELEPATI 8.0)"
versi: 1.0
tanggal: 2026-10-23
status: AKTIF
pemilik: Dimas
bergantung_pada: [01_PRD.md]
---

# Data Card — Agritech Data

> **Untuk manusia dan AI.** Semua angka di dokumen ini hasil pemeriksaan langsung terhadap dataset resmi yang diunduh dari Dashboard Peserta. Gunakan angka ini untuk memvalidasi notebook (S-02). Dataset **tidak boleh** diunggah ke repo atau tempat publik (NFR-05).

---

## 1. Identitas paket data

| Hal | Nilai |
|---|---|
| Sumber | Dashboard Peserta TELEPATI 8.0 (dataset resmi, R-06) |
| Format | COCO JSON ekspor Roboflow, 3 split |
| Ukuran gambar | Semua 640×640 |
| Paket kerja tim | `telepati-agridata-raw.zip` berisi `train/`, `valid/`, `test/` saja (tanpa `__MACOSX` dan `.DS_Store`) |
| Ukuran zip | 613.048.556 byte (584,6 MB), 13.301 file |
| SHA-256 zip | `797e47749f91f4403e778dd0150716da1cd94cddbecee2ec56cf5545c50cec50` |
| Keaslian | Ukuran dan CRC32 seluruh 13.301 file di zip sama dengan file hasil ekstrak dataset resmi |

Zip asli dari panitia tidak tersimpan. Zip kerja dibuat ulang dari isi dataset resmi tanpa perubahan apa pun.

---

## 2. Statistik per split

| Split | Gambar | Anotasi | Gambar tanpa anotasi | File COCO |
|---|---|---|---|---|
| train | 10.133 | 20.163 | 59 | `train/_annotations.coco.json` |
| valid | 2.106 | 4.888 | 13 | `valid/_annotations.coco.json` |
| test | 1.059 | 2.670 | 5 | `test/_annotations.coco.json` |
| **Total** | **13.298** | **27.721** | 77 | |

Ketiga file COCO memiliki daftar 21 kategori yang identik.

---

## 3. Pemetaan 21 kategori raw → 11 kelas canonical (R-09)

Urutan indeks canonical: alfabet, mengikuti halaman regulasi resmi.

| Indeks | Kelas canonical | Nama raw di COCO | ID raw |
|---|---|---|---|
| 0 | Bacterial leaf blight | `Bacterial leaf blight` | 15 |
| 1 | Bacterial panicle blight | `Bacterial panicle Blight` | 1 |
| 2 | Blast | `Blast`, `Leaf blast`, `Infected Blast` | 8, 17, 5 |
| 3 | Brown spot | `BrownSpot`, `Brown spot` | 9, 16 |
| 4 | False smut | `False-Smut` | 2 |
| 5 | Healthy | `Healthy Rice Leaf`, `Healthy Rice beads`, `Healthy`, `healthy` | 3, 4, 10, 20 |
| 6 | Leaf roller | `Leaf-roller` | 6 |
| 7 | Leaf scald | `Leaf Scald`, `Leaf scald` | 11, 18 |
| 8 | Narrow brown | `Narrow brown` | 19 |
| 9 | Sheath blight | `Sheath Blight` | 13 |
| 10 | Tungro | `Rice-Tungro` | 12 |
| — | *(dibuang, supercategory)* | `Leaf-blight`, `Rice-Leaf-Diseasee`, `paddy` | 0, 7, 14 |

- 18 nama raw bermakna + 3 supercategory = 21 kategori.
- Ketiga supercategory **tidak punya satu pun bounding box** di semua split.
- Pemetaan di notebook dibuat berbasis **nama**, lalu divalidasi terhadap ID ini.
- Ringkasan dari asisten AI lain yang beredar di awal salah menulis nama raw kelas Healthy. Tabel ini sudah dicocokkan langsung dengan file COCO.

---

## 4. Jumlah box per kelas canonical

| Kelas | Train | Valid | Test |
|---|---|---|---|
| Bacterial leaf blight | 476 | 125 | 59 |
| Bacterial panicle blight | 528 | 45 | 27 |
| Blast | 4.149 | 947 | 535 |
| Brown spot | 5.010 | 1.461 | 840 |
| False smut | 852 | 86 | 43 |
| Healthy | 2.374 | 424 | 184 |
| Leaf roller | 812 | 87 | 44 |
| Leaf scald | 1.438 | 390 | 222 |
| Narrow brown | 222 | 75 | 36 |
| Sheath blight | 1.762 | 488 | 269 |
| Tungro | 2.540 | 760 | 411 |
| **Total** | **20.163** | **4.888** | **2.670** |

Implikasi:
- Ketimpangan kelas di train sekitar 22 kali (Brown spot 5.010 vs Narrow brown 222).
- Metrik F1 **makro** memberi bobot sama ke setiap kelas, jadi kelas langka sangat menentukan skor.
- Test hanya punya 27 box Bacterial panicle blight dan 36 box Narrow brown, sehingga F1 kelas itu sensitif terhadap beberapa deteksi saja.

---

## 5. Karakteristik bounding box (train)

| Statistik luas box | Nilai (px²) |
|---|---|
| Minimum | 1 |
| Kuartil 25% | 1.332 (sekitar 36×36 px) |
| Median | 12.238 (sekitar 110×110 px) |
| Kuartil 75% | 108.942 |
| Maksimum | 409.600 (seluas gambar penuh) |

- Box per gambar: median 1, maksimum 178.
- Tidak ada box dengan lebar atau tinggi ≤ 0.
- Ada box sangat kecil (luas 1–3 px²) yang patut dicurigai sebagai label rusak (aturan S-04).
- Gaya label campuran: sebagian gambar berisi banyak box lesi kecil, sebagian hanya satu box seluas gambar (gaya klasifikasi).

---

## 6. Sumber campuran

Pola nama file menunjukkan dataset gabungan beberapa sumber: foto WhatsApp (`IMG-...-WA...`), foto ponsel (`20240915_...`), file bernama sesuai kelas (`BROWNSPOT...`, `TUNGRO...`), dan gambar yang sudah ber-augmentasi (`aug_...`). Semua telah melalui ekspor Roboflow (`..._jpg.rf.<hash>.jpg`).

---

## 7. Duplikasi (temuan pemeriksaan awal)

Pemeriksaan perceptual hash (pHash 64-bit, memperhitungkan flip dan rotasi 90°, ambang jarak Hamming ≤ 4, diverifikasi visual pada sampel):

| Perbandingan | Gambar yang punya kembaran | Persentase |
|---|---|---|
| valid → train | 1.111 / 2.106 | 52,8% |
| test → train | 564 / 1.059 | 53,3% |
| train di dalam train | 8.227 / 10.133 | 81,2% |

- Train hanya sekitar **4.878 grup unik** dari 10.133 gambar (banyak salinan hasil flip/rotasi/crop).
- Panitia menyatakan duplikat bawaan **bukan pelanggaran** selama training hanya pada train, dan dataset tidak akan direvisi (R-20).

**[ATURAN] Batas penggunaan temuan ini:**
- Statistik di atas hanya informasi latar. Pemeriksaan itu dilakukan di luar repo dan tidak dipakai untuk melatih atau memilih apa pun.
- Notebook **tidak boleh** berisi pencocokan gambar test↔train atau valid↔train, dan tidak boleh memakai label test selain untuk menghitung skor (R-20c, D-06).
- Pengelompokan gambar kembar **di dalam train** (train↔train) boleh dipakai, misalnya untuk holdout diagnostik.
