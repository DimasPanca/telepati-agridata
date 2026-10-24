---
doc_id: PRD
judul: "Product Requirements Document — AI AgriData Intelligence Race (TELEPATI 8.0)"
versi: 1.0
tanggal: 2026-10-23
status: AKTIF (sumber kebenaran tunggal tim)
pemilik: Dimas (Arsitek)
dibaca_oleh: Dimas, Rahman, Maman
dokumen_terkait: [02_SYSTEM_DESIGN.md, 03_DATA_CARD.md, 04_EXPERIMENT_PLAN.md, 05_AUDIT_RUNBOOK.md, 06_GIT_WORKFLOW.md]
platform_utama: Google Colab (GPU T4)
batas_submission: 2026-10-25 23:59 WIB
---

# PRD — AI AgriData Intelligence Race (TELEPATI 8.0)

> **Cara membaca dokumen ini (untuk manusia dan AI).**
> - Setiap aturan punya ID `R-xx`, setiap kebutuhan punya ID `REQ-xx`, setiap keputusan punya ID `D-xx`. Rujuk ID ini di commit, pull request, dan diskusi.
> - Penanda: **[ATURAN]** = ketentuan resmi lomba, tidak bisa ditawar. **[KEPUTUSAN]** = sudah dikunci tim. **[TERBUKA]** = belum pasti, pakai default yang tertulis. **[DUGAAN]** = asumsi tim yang belum dikonfirmasi panitia.
> - Kalau dokumen lain bertentangan dengan PRD, **PRD yang berlaku**. Kalau PRD bertentangan dengan aturan resmi panitia, **aturan panitia yang berlaku**, lalu PRD diperbaiki.

---

## 1. Ringkasan

Tim membangun **satu model object detection** yang mendeteksi **11 kelas kondisi daun/tanaman padi** (10 penyakit + sehat) pada citra 640×640, **dilatih dari nol** (tanpa pretrained), dengan seluruh alur dari data mentah sampai inference ditulis di **satu Jupyter Notebook** yang reproducible, dikumpulkan lewat **satu repo GitHub publik**, dan diaudit juri dengan **inference di Google Colab T4**.

Skor yang dinilai: **F1-Score makro pada split test resmi**.

---

## 2. Konteks dan studi kasus

- Tema TELEPATI 8.0: *"AgriTech: Growing the Golden Future"*. Penyelenggara: HIMATEL POLBAN.
- Studi kasus: **Arif**, petani muda, memantau ribuan tanaman padi secara manual. Ia ingin sistem *Smart Agriculture* berbasis kamera (drone/perangkat pemantau) yang otomatis mengenali kondisi tanaman.
- Tantangan resmi: membangun model AI berbasis **object detection** dengan akurasi, presisi, dan keandalan tinggi, siap menjadi komponen utama sistem Smart Agriculture.

---

## 3. Tujuan dan ukuran sukses

| ID | Tujuan | Ukuran sukses |
|---|---|---|
| G-1 | Lolos verifikasi audit (status **VERIFIED**) | Skor audit juri berselisih ≤ 0,02 dari skor mandiri. Target internal: selisih < 0,005. |
| G-2 | Skor setinggi mungkin | F1 makro test resmi setinggi mungkin. Target internal ditetapkan setelah baseline (lihat `04_EXPERIMENT_PLAN.md`). |
| G-3 | Patuh aturan 100% | Tidak ada pelanggaran `R-xx` (terutama leakage `R-20`, pretrained `R-10`, LLM `R-11`). |
| G-4 | Tepat waktu | Submission terverifikasi masuk sebelum **25 Okt 2026 18:00 WIB** (batas resmi 23:59 WIB). |
| G-5 | Siap babak final (jika Top 3) | Model dan demo siap dipresentasikan 7 Nov 2026 di POLBAN. |

Non-tujuan babak penyisihan: aplikasi/dashboard, inference tiling untuk citra drone besar, skor keparahan per petak. Semua ini ditunda ke persiapan final (lihat §11).

---

## 4. Aturan resmi lomba (lengkap)

Sumber: halaman regulasi resmi TELEPATI 8.0 dan jawaban tertulis panitia (diterima 23 Okt 2026). Ringkasan dari asisten AI lain yang beredar di awal **bukan** sumber resmi dan sebagian datanya usang.

### 4.1 Peserta dan pendaftaran

| ID | [ATURAN] | Status tim |
|---|---|---|
| R-01 | Tim 2–3 mahasiswa aktif D3/D4/S1 dari perguruan tinggi yang sama. | 3 orang: Dimas, Rahman, Maman (UNTIRTA). |
| R-02 | Wajib 1 Ketua Tim dan 1 Dosen Pembimbing. Dosen tidak dihitung anggota. | Ketua: Dimas *(konfirmasi)*. Dosen: *(isi nama)*. |
| R-03 | Setiap tim hanya boleh mengirim **1 model AI**. | Tidak ada ensemble (lihat D-07). |
| R-04 | Semua anggota mencantumkan username/tautan GitHub. | Repo: `github.com/DimasPanca/telepati-agridata`. Username Rahman dan Maman: *(isi)*. |
| R-05 | Biaya Rp135.000/tim. Follow IG `@telepatipolban`. Unggah KTM aktif (PDF/JPG/PNG, maks 2 MB). Unggah twibbon di IG pribadi, mention `@telepatipolban`, tag minimal 3 teman. Grup WA diakses lewat website setelah registrasi. | Setiap anggota cek dan centang di `05_AUDIT_RUNBOOK.md` §5. |

### 4.2 Jadwal resmi

| Tanggal | Kegiatan |
|---|---|
| 1 Agu – **25 Okt 2026 23:59 WIB** | Pendaftaran dan submission karya (diperpanjang) |
| 26 Okt – 1 Nov 2026 | Audit run dan penilaian juri |
| 2 Nov 2026 | Pengumuman Top 3 finalis (website dan Instagram) |
| 5 Nov 2026 | Technical meeting finalis (daring) |
| 7 Nov 2026 | Presentasi final, demonstrasi model, dan awarding (luring, POLBAN) |

### 4.3 Teknis dataset dan pengembangan

| ID | [ATURAN] |
|---|---|
| R-06 | Wajib memakai **Dataset Resmi (Agritech Data)** dari panitia. |
| R-07 | Seluruh alur (cleaning, preprocessing, augmentation) **wajib** berupa kode Python di Jupyter Notebook (`.ipynb`), dimulai dari **dataset mentah**. Dilarang mengunggah data bersih tanpa jejak kode pemrosesan. |
| R-08 | Wajib menyertakan **random seed** (`random_state`) di setiap kode pemrosesan dan pelatihan. |
| R-09 | Wajib memetakan 21 kategori COCO bawaan menjadi **11 kelas canonical** di alur preprocessing. Tiga supercategory (`Leaf-blight`, `Rice-Leaf-Diseasee`, `paddy`) bukan kelas dan tidak dipakai. Detail pemetaan: `03_DATA_CARD.md` §3. |
| R-10 | **Dilarang** memakai pretrained model eksternal di luar dataset resmi. |
| R-11 | **Pengolahan dataset lewat API LLM dilarang keras.** Halaman regulasi menyebut LLM/asisten hanya untuk debugging sintaksis dan dokumentasi; panitia kemudian meluruskan bahwa yang dilarang adalah pengolahan dataset lewat API LLM, sedangkan penggunaan AI untuk membangun kode diperbolehkan (pelurusan diterima tim, 23 Okt 2026; simpan bukti tertulisnya). |

### 4.4 Repo dan berkas pengumpulan

| ID | [ATURAN] |
|---|---|
| R-12 | Seluruh pengumpulan terpusat di **1 repo GitHub publik** tim. |
| R-13 | Model weights (`.pt`, `.onnx`, `.h5`, `.pkl`) wajib diunggah lewat **GitHub Releases atau Git LFS** di repo tim, dengan **direct link** ke file rilis. **Dilarang lewat Google Drive.** |
| R-14 | Riwayat commit **berkala dan bertahap** oleh akun GitHub anggota terdaftar. Single commit berisiko **REJECTED/diskualifikasi**. |
| R-15 | `README.md` wajib memuat: **How to Run**, **daftar dependencies**, **arsitektur model**, **hyperparameter**, **tautan model weights**. |
| R-16 | Berkas babak penyisihan: (1) trained model weights, (2) Jupyter Notebook end-to-end (preprocessing, training, inference) + dokumentasi Markdown, (3) repo GitHub publik dengan riwayat commit bertahap dan README, (4) **Surat Pernyataan Orisinalitas**: scan PDF, bermeterai Rp10.000, ditandatangani Ketua Tim dan Dosen Pembimbing, diunggah lewat form website. |

### 4.5 Penilaian dan audit

| ID | [ATURAN] |
|---|---|
| R-17 | Skor mandiri dan audit dihitung pada **split test resmi**. Metrik: **F1-Score makro**, tanpa threshold confidence khusus (jawaban panitia). Halaman regulasi juga menyebut mAP@50 (lihat O-01). |
| R-18 | Audit dijalankan di **Google Colab (GPU T4)**. Skor audit diambil dari **inference memakai weights yang dikumpulkan**, tanpa melatih ulang. Notebook training tetap wajib dilampirkan dan **akan diperiksa**. Model dilatih ulang hanya jika ada indikasi pelanggaran. Toleransi selisih skor **±0,02**. Inference **maksimal 30 menit**. Seed tetap. |
| R-19 | Status **VERIFIED**: hasil audit konsisten dengan skor mandiri, masuk Official Leaderboard. Status **REJECTED**: kode error, data leakage, file rusak, tidak reproducible, atau selisih skor melebihi toleransi. |
| R-20 | **Leakage (REJECTED):** (a) melatih model memakai data valid/test, termasuk pseudo-label; (b) memilih model atau hyperparameter berdasarkan skor test (gunakan valid); (c) mengambil label test lewat pencocokan gambar dengan train. Duplikat bawaan panitia **bukan** pelanggaran selama melatih hanya pada split train. Dataset tidak direvisi. |
| R-21 | Re-submission diperbolehkan selama periode pendaftaran. Submission terbaru yang terverifikasi menimpa yang lama, tanpa pemotongan nilai. Tim yang masih REJECTED sampai batas akhir tidak masuk Official Leaderboard. |

---

## 5. Kebutuhan (requirements)

### 5.1 Kebutuhan fungsional

| ID | Kebutuhan | Pemilik | Rujukan |
|---|---|---|---|
| REQ-01 | Notebook membaca dataset **mentah** (zip resmi) dari satu variabel path, tanpa langkah manual. | Dimas | R-06, R-07 |
| REQ-02 | Notebook memetakan 21 kategori raw ke 11 kelas canonical dengan dictionary eksplisit, lalu memvalidasi bahwa tidak ada anotasi tak terpetakan. | Dimas | R-09 |
| REQ-03 | Notebook membersihkan anotasi (box rusak/terlalu kecil) dengan aturan tertulis dan mencatat jumlah yang dibuang. | Dimas | R-07 |
| REQ-04 | Notebook mengonversi COCO ke format YOLO hanya untuk train dan valid sebagai data latih/seleksi; test hanya dipakai di sel evaluasi akhir. | Dimas | R-20 |
| REQ-05 | Training dari nol: arsitektur dibangun dari file `.yaml`, `pretrained=False`, tanpa memuat bobot `.pt` eksternal. | Dimas (desain), Rahman/Maman (eksekusi) | R-10 |
| REQ-06 | Seed tetap dan mode deterministik di semua tahap (split, augmentasi, training, inference). | Semua | R-08, R-18 |
| REQ-07 | Fungsi evaluasi tunggal menghitung **F1 makro 11 kelas** (dua versi, lihat D-09) plus mAP@50 sebagai pembanding. | Dimas | R-17 |
| REQ-08 | Pemilihan checkpoint hanya berdasarkan **valid**. | Rahman (analisis), Dimas (keputusan) | R-20(b) |
| REQ-09 | Jalur **inference-saja**: unduh weights dari direct link Release, inference pada test resmi, cetak skor. Selesai < 10 menit di Colab T4 (batas resmi 30 menit). | Dimas (bangun), Maman (uji) | R-18 |
| REQ-10 | Notebook final tersimpan **dengan output** (log, kurva, tabel) sebagai bukti training. | Maman, Dimas | R-18 |

### 5.2 Kebutuhan non-fungsional

| ID | Kebutuhan |
|---|---|
| NFR-01 | Reproducible: menjalankan jalur inference dua kali menghasilkan skor identik. |
| NFR-02 | Portabel: tidak ada path lokal laptop, path Kaggle, atau path Drive pribadi yang tertinggal di jalur inference. |
| NFR-03 | Versi library dipin di `requirements.txt` dan README. |
| NFR-04 | Tidak ada kredensial (token Kaggle/GitHub) di notebook atau repo. |
| NFR-05 | Dataset resmi tidak pernah diunggah ke repo atau ke tempat publik. |
| NFR-06 | Weights final dirilis di GitHub Releases, bukan Google Drive (Drive hanya tempat kerja sementara). |

### 5.3 Deliverables dan definisi selesai

| # | Deliverable | Definisi selesai |
|---|---|---|
| DL-1 | Model weights (`best.pt`) | Ada di GitHub Release `v1.x`, direct link bisa diunduh tanpa login, SHA-256 tercatat di README. |
| DL-2 | Notebook `notebooks/agridata_pipeline.ipynb` | Berjalan end-to-end dari data mentah, berisi Markdown penjelasan setiap tahap, tersimpan dengan output, lolos latihan audit. |
| DL-3 | Repo GitHub publik | Publik saat submit, commit bertahap dari ketiga akun, README lengkap sesuai R-15, `.gitignore` mencegah dataset/weights/kredensial ikut. |
| DL-4 | Surat Pernyataan Orisinalitas | PDF scan, meterai Rp10.000, tanda tangan Ketua dan Dosen Pembimbing, terunggah di form website. |
| DL-5 | Skor mandiri | Diambil dari latihan audit Colab T4 (bukan dari log training), dicatat di README dengan versi library dan parameter inference. |

---

## 6. Keputusan yang sudah dikunci

| ID | [KEPUTUSAN] | Alasan |
|---|---|---|
| D-01 | **Google Colab (T4) adalah platform utama** untuk training dan audit. Kaggle hanya cadangan kalau Colab tidak memberi GPU. | Audit juri berjalan di Colab T4, jadi lingkungan kerja = lingkungan audit. |
| D-02 | Keluarga model: **YOLO dari Ultralytics, dibangun dari `.yaml`, dilatih dari nol.** | Detektor transformer (DETR) bergantung pada backbone pretrained. CNN YOLO lebih cocok dari nol dan pasti termuat di Colab. |
| D-03 | Tiga run paralel, satu perubahan per run: **A** = YOLO11s baseline, **B** = YOLO26s, **C** = YOLO11s + penyeimbangan kelas. | Efisien, hasil bisa dibandingkan adil. Detail: `04_EXPERIMENT_PLAN.md`. |
| D-04 | Resolusi 640, seed `42`, `deterministic=True`. | Ukuran asli data 640×640. Reproducibility. |
| D-05 | Training **hanya** pada split train (10.133 gambar). Valid untuk seleksi. Test dievaluasi **sekali** di akhir. | R-20. |
| D-06 | Tidak ada pencocokan gambar test↔train di notebook atau inference. Pengelompokan gambar kembar hanya di dalam train. | R-20(c). |
| D-07 | Tidak ada ensemble, pseudo-label, self-supervised, atau TTA berat. | R-03, R-18, R-20(a). |
| D-08 | Weights final lewat **GitHub Releases** (bukan Git LFS). | Direct link sederhana, tanpa kuota LFS. |
| D-09 | Skor dilaporkan dalam dua versi: **E1** = `mf1` Ultralytics `val()` (F1 pada confidence optimal bersama, IoU 0,5) dan **E2** = F1 makro pada confidence tetap yang ditulis eksplisit. | Menutup risiko definisi metrik juri (O-01). |
| D-10 | Skor mandiri = angka dari **latihan audit** di runtime Colab T4 yang bersih. | Paling mirip cara juri. |
| D-11 | Dimas adalah satu-satunya pengubah kode inti, desain, dan tabel `RUNS`. Rahman dan Maman mengirim log dan hasil evaluasi lewat branch dan pull request. | Menghindari "banyak tangan" pada inti. |
| D-12 | Model cadangan: run pertama yang lolos latihan audit langsung disubmit; model lebih baik (berdasarkan **valid**) di-resubmit sebelum batas. | R-21, jaminan selalu ada yang terkirim. |
| D-13 | Versi library dipin: **ultralytics 8.4.184** (mendukung YOLO11 dan YOLO26). PyTorch memakai versi bawaan Colab dan dicatat di setiap run. | Lingkungan training = lingkungan audit. |
| D-14 | Pembersihan hanya untuk **train**: box dipotong ke batas gambar, sisi < 2 px dibuang, box identik dalam satu gambar disisakan satu. Label valid dan test hanya dipetakan ke 11 kelas dan dipotong ke batas gambar. | Skor valid/test harus dihitung terhadap label yang sama dengan juri. |
| D-20 | Kandidat v1.2 ditetapkan sebelum evaluasi test: bobot v1.0 tanpa perubahan (file dan SHA-256 sama), dengan NMS per kelas IoU 0,45 (sebelumnya default 0,7) dan `E2_CONF` 0,20 untuk evaluasi akhir, test, dan inference. Dasar keputusan, valid saja: uji silang dua paruh di bagian 9 memberi F1 makro 0,6909 → 0,7066, dan pada analisis terpisah kenaikannya positif di 10 dari 10 penilaian (terkecil +0,012); E1 0,6981 → 0,7113; mAP@50 0,6884 → 0,6929. NMS lintas kelas 0,45 memberi F1 setara (uji silang 0,7077, E1 0,7115) tetapi mAP@50 turun ke 0,6769, sehingga dipilih NMS per kelas. Kalibrasi v1.1 di atas NMS baru tidak menambah skor (−0,0002) dan inference 800 px lebih rendah (0,6905), sehingga keduanya tidak dipakai. Analisis ini dilakukan setelah hasil test v1.1 diketahui; karena itu pemilihan hanya memakai valid dan aturan ini dicatat sebelum test v1.2. Hasil test v1.2 dilaporkan apa adanya dan tidak dipakai untuk memilih versi. Jika v1.2 gagal secara teknis, dipakai v1.0 yang sudah lolos latihan audit. | R-20b. Satu parameter inference yang berlaku sama untuk semua kelas, sehingga risiko overfit ke valid lebih kecil dibanding 11 offset v1.1. Tabel lengkap: `results/nms_selection_valid.csv`. |
| D-19 | Kandidat v1.1 ditetapkan sebelum evaluasi test: bobot v1.0 (A_fit epoch20) dengan kalibrasi bias per kelas (target conf 0,26, penyusutan n/(n+50)), tanpa model soup. Dasar keputusan: uji silang dua paruh valid memberi kenaikan F1 makro +0,0053 dan +0,0075 (rata-rata +0,0064); model soup hanya +0,0003. Hasil test v1.1 dilaporkan apa adanya dan tidak dipakai untuk memilih antara v1.0 dan v1.1. v1.0 tetap dipakai hanya jika v1.1 gagal secara teknis (latihan audit tidak lolos atau melewati batas waktu). | R-20b |
| D-18 | Bagian 9b notebook: model soup greedy (rata-rata bobot EMA checkpoint satu run, diterima hanya jika E1 valid naik) dan kalibrasi bias kelas pada head (`logit(t_bersama) − logit(t_kelas)`, ditarik ke nol dengan `n/(n+50)`). Rilis baru (v1.1) hanya jika E1 valid naik ≥ 0,008 dari checkpoint dasar. | Tetap satu model; keputusan hanya dari valid (R-03, R-20b). Kalibrasi tersimpan di weights sehingga tidak bergantung pada setelan inference juri. |
| D-17 | Run lanjutan `A_fit_yolo11s_s42`: fine-tuning dari `best.pt` Run A di 640 dengan mosaic, mixup, dan variasi warna dimatikan (flip, skala 0,25, dan pergeseran 0,05 tetap), 50 epoch, SGD `lr0=0.002`. Ditambahkan setelah A800 hanya +0,005 di valid dan evaluasi Run A di 2.000 gambar train menunjukkan F1 makro 0,744 (underfit). | Training tetap hanya pada split train (R-20); bobot awal hasil training tim sendiri (R-10). |
| D-16 | Run lanjutan `A800_yolo11s_s42`: fine-tuning dari `best.pt` Run A di resolusi 800, menggantikan Run C di akun Dimas. Ditambahkan setelah Run A menunjukkan kelas langka sudah kuat (Narrow brown ±0,98) dan kelemahan ada di lesi kecil (Brown spot ±0,40, Leaf scald ±0,54, Blast ±0,56). | Bobot awal hasil training tim sendiri dari nol, bukan pretrained eksternal (R-10 tetap terpenuhi). |
| D-15 | Seluruh kode ada di notebook; konfigurasi run berupa tabel `RUNS` di bagian 7 notebook (tidak ada folder `configs/`). Trainer hanya mengubah `RUN_ID` di sel konfigurasi. | R-07, dan notebook bisa langsung dijalankan dari Colab tanpa clone repo privat. |

---

## 7. Peran, wewenang, dan RACI

| Orang | Peran | Wewenang |
|---|---|---|
| **Dimas** | Arsitek (fundamental + computer vision), Ketua Tim *(konfirmasi)* | Satu-satunya pengubah kode inti dan desain. Keputusan model final. Merge pull request. Rilis weights. |
| **Rahman** | Trainer + analis hasil | Menjalankan Run A. Menggabungkan evaluasi checkpoint semua run di valid. Laporan F1 per kelas. Mengusulkan pemenang. |
| **Maman** | Trainer + penjaga reproduksibilitas | Menjalankan Run B. Memastikan setiap run sah (seed, setelan, log). Latihan audit di Colab T4. `requirements.txt`. |

RACI (R = mengerjakan, A = memutuskan, C = dimintai masukan, I = diberi tahu):

| Pekerjaan | Dimas | Rahman | Maman |
|---|---|---|---|
| Inti pipeline (data, mapping, cleaning, konversi) | R/A | I | I |
| Desain arsitektur dan konfigurasi run | R/A | C | C |
| Fungsi evaluasi F1 makro | R/A | C | I |
| Menjalankan Run A | C | R | I |
| Menjalankan Run C | R | I | I |
| Menjalankan Run B | C | I | R |
| Evaluasi checkpoint di valid | A | R | C |
| Pemilihan model final | A | R | C |
| Jalur inference-saja | R/A | I | C |
| Latihan audit Colab T4 | A | I | R |
| Rilis weights dan README | R/A | C | C |
| Surat orisinalitas dan submit | R/A (Ketua) | I | I |

---

## 8. Jadwal kerja dan titik keputusan

| Waktu | Dimas | Rahman | Maman | Titik keputusan |
|---|---|---|---|---|
| 23 Okt pagi–siang | Selesaikan notebook inti, jalankan `E0_smoke` di Colab | Siapkan Colab + Drive, unggah zip dataset | Siapkan Colab + Drive, unggah zip dataset, cek `requirements.txt` | Inti lolos uji alur → tag `v0.1-core` |
| 23 Okt sore–malam | Mulai Run C, pantau B dan C di epoch awal | Mulai Run A | Mulai Run B | Ketiga run berjalan |
| 24 Okt | Jalur inference-saja, notebook final | Evaluasi checkpoint di valid, laporan per kelas | Lanjutkan segmen run, latihan audit model cadangan | Pilih model final (valid) |
| 24 Okt malam | Rilis `v1.0`, submit cadangan | — | Latihan audit model final | Skor mandiri dikunci |
| 25 Okt | README final, re-submit jika ada model lebih baik di valid | Bagian hasil eksperimen di README | Latihan audit ulang | **Submit final sebelum 18:00 WIB** |

---

## 9. Risiko dan rencana cadangan

| ID | Risiko | Dampak | Mitigasi |
|---|---|---|---|
| RK-01 | Colab gratis tidak memberi GPU atau memutus sesi | Run terlambat | Checkpoint ke Drive tiap epoch, resume. Tiap anggota pakai akun Google sendiri. Cadangan: Kaggle (`CHECKLIST_SETUP_KAGGLE_COLAB.md` di folder proyek) atau Colab pay-as-you-go (sekitar $9,99/100 unit, data pihak ketiga). |
| RK-02 | YOLO26 tidak didukung versi ultralytics yang dipin atau tidak stabil dari nol | Run B gagal | Run B hanya penantang. Run A tetap baseline. |
| RK-03 | Definisi F1 juri berbeda dari kita | Selisih skor > 0,02 → REJECTED | D-09 (dua versi skor), pertanyaan O-01, pilih checkpoint dengan kurva F1 landai. |
| RK-04 | Skor audit meleset karena versi library | REJECTED | Pin versi, latihan audit di runtime bersih, uji juga dengan ultralytics terbaru. |
| RK-05 | Leakage tidak sengaja | REJECTED | D-05, D-06, D-07, checklist kepatuhan di `05_AUDIT_RUNBOOK.md`. |
| RK-06 | Kelas langka (Narrow brown, Bacterial panicle blight) lemah | F1 makro turun | Run C (penyeimbangan kelas), analisis per kelas. |
| RK-07 | Dimas jadi titik macet | Training mulai terlambat | Inti dikerjakan berurutan prioritas (`02_SYSTEM_DESIGN.md` §11). |
| RK-08 | Commit hanya dari satu akun | Dicurigai saat pemeriksaan | Setiap anggota commit pekerjaan nyatanya sendiri (`06_GIT_WORKFLOW.md`). |
| RK-09 | Surat orisinalitas terlambat (meterai, tanda tangan dosen) | Submission tidak lengkap | Urus 23 Okt. |

---

## 10. Pertanyaan terbuka

| ID | [TERBUKA] | Default sementara |
|---|---|---|
| O-01 | Implementasi F1 makro juri (mf1 Ultralytics atau confidence tetap), IoU pencocokan, apakah mAP@50 masih dipakai. | Laporkan E1 dan E2 (D-09). [DUGAAN] E1 = cara juri. |
| O-02 | Audit memakai sel inference notebook kita atau skrip juri; format dan versi library yang didukung. | Pakai arsitektur standar ultralytics, `.pt`, versi dipin, uji juga dengan versi terbaru. |
| O-03 | Toleransi ±0,02 absolut atau relatif. | [DUGAAN] absolut. Target internal < 0,005. |
| O-04 | Re-submission setelah REJECTED pada masa audit. | Submit cadangan sedini mungkin (D-12). |
| O-05 | Urutan ID kelas wajib atau tidak. | Urutan alfabet seperti di situs (`03_DATA_CARD.md` §3). |
| O-06 | ~~Batas penggunaan LLM untuk membangun kode.~~ **Terjawab** 23 Okt: diperbolehkan; yang dilarang hanya pengolahan dataset lewat API LLM (R-11). | — |
| O-07 | Pretraining mandiri pada train. | Tidak dipakai. |

---

## 11. Persiapan babak final (setelah 2 Nov, jika Top 3)

- Inference tiling untuk citra drone beresolusi besar.
- Skor keparahan per petak (rasio area lesi) dan daftar prioritas penanganan, sesuai kasus Arif.
- Ekspor ONNX dan ukur latensi CPU sebagai bukti siap edge.
- Demo sederhana untuk presentasi 7 Nov.

---

## 12. Glosarium

| Istilah | Arti |
|---|---|
| Kelas canonical | 11 kelas resmi hasil pemetaan dari 21 kategori raw |
| Skor mandiri | Skor yang dilaporkan tim, harus sama dengan hasil audit juri dalam toleransi |
| F1 makro | Rata-rata F1 dari 11 kelas dengan bobot sama |
| E1 / E2 | Dua versi F1 makro yang kita laporkan (D-09) |
| Latihan audit | Meniru audit juri di runtime Colab T4 yang bersih (`05_AUDIT_RUNBOOK.md`) |
| Run | Satu eksperimen training dengan konfigurasi tetap (A, B, atau C) |
| Gambar kembar | Gambar nyaris identik (flip/rotasi/crop) di dataset resmi |
| Leakage | Penggunaan data valid/test yang dilarang R-20 |
