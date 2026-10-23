---
doc_id: AUDIT_RUNBOOK
judul: "Runbook Latihan Audit (Colab T4) dan Checklist Submission"
versi: 1.0
tanggal: 2026-10-23
status: AKTIF
pemilik: Maman (latihan audit), Dimas (submission)
bergantung_pada: [01_PRD.md, 02_SYSTEM_DESIGN.md]
---

# Runbook Latihan Audit dan Checklist Submission

> **Untuk manusia dan AI.** Juri mengaudit dengan inference memakai weights kita di Google Colab T4, tanpa training ulang, maksimal 30 menit, toleransi ±0,02 (R-18). Runbook ini meniru proses itu. **Skor mandiri yang dilaporkan adalah angka dari latihan ini** (D-10).

---

## 1. Prasyarat

- [ ] Repo berisi notebook final, `requirements.txt` (ultralytics 8.4.184), dan README.
- [ ] `WEIGHTS_URL`, `WEIGHTS_SHA256`, dan `E2_CONF` sudah diisi di sel konfigurasi notebook.
- [ ] Weights ada di GitHub Release dengan direct link yang bisa diunduh **tanpa login**.
- [ ] SHA-256 weights tercatat di README.
- [ ] Model final sudah dipilih berdasarkan **valid** (bukan test).

---

## 2. Langkah latihan audit

- [ ] Buka runtime Colab **baru dan bersih** (Runtime → Disconnect and delete runtime lebih dulu). Pilih **T4 GPU**.
- [ ] Buka notebook dari repo GitHub (link "Open in Colab"), bukan salinan pribadi.
- [ ] Set `MODE = "inference_only"`.
- [ ] Jalankan sel setup: pasang `requirements.txt`. Catat lama pemasangan.
- [ ] Siapkan data mentah test lewat variabel path tunggal (dari Drive ke `/content`).
- [ ] Unduh weights dari **direct link Release**. Cocokkan SHA-256 dengan README.
- [ ] Jalankan S-11: inference test, cetak E1, E2, mAP@50, F1 per kelas.
- [ ] Catat **waktu total** dari muat model sampai skor tercetak. Target < 10 menit (batas resmi 30 menit).
- [ ] **Jalankan sekali lagi** di runtime yang sama. Angka harus identik.
- [ ] Ulangi di runtime bersih kedua dengan ultralytics **terbaru tanpa pin**. Pastikan weights termuat dan skor sama. Kalau berbeda, tulis versi wajib di README.
- [ ] Opsional: bandingkan `batch` 1 vs 16 dan FP32 vs FP16. Kalau selisih > 0,005, kunci parameter itu dan tulis di README.
- [ ] Pastikan tidak ada sel yang mencocokkan test dengan train atau memakai label test selain untuk menghitung skor.
- [ ] Isi log (§3) di `results/audit_log.md`, commit dari akun Maman.

Kriteria lolos: selisih antar-jalankan = 0, selisih antar-versi library < 0,005, waktu < 10 menit, tanpa error.

---

## 3. Template log latihan audit

```text
Tanggal / jam        :
Pelaksana            : Maman
Versi notebook       : (tag/commit)
Release weights      : v
SHA-256 weights      :
Versi library        : python=, torch=, ultralytics=
GPU                  : T4
Jalankan 1           : E1=     E2=     mAP50=
Jalankan 2           : E1=     E2=     mAP50=
Ultralytics terbaru  : versi=     E1=     E2=
Waktu total          :
Masalah              :
Status               : LOLOS / GAGAL
```

---

## 4. Checklist kepatuhan (centang sebelum submit)

- [ ] **R-07** Notebook berjalan end-to-end dari zip mentah, tanpa data olahan dari luar.
- [ ] **R-08** Seed `42` dan mode deterministik di setiap tahap.
- [ ] **R-09** Pemetaan 21 → 11 kelas ada di notebook dan divalidasi.
- [ ] **R-10** Model dibangun dari `.yaml`, `pretrained=False`, tidak ada bobot eksternal.
- [ ] **R-11** Tidak ada pengolahan dataset lewat API LLM. Bukti tertulis pelurusan panitia soal penggunaan AI tersimpan.
- [ ] **R-13** Weights di GitHub Release, direct link di README, tidak lewat Google Drive.
- [ ] **R-14** Commit bertahap dari ketiga akun anggota.
- [ ] **R-15** README memuat How to Run, dependencies, arsitektur, hyperparameter, link weights.
- [ ] **R-18** Notebook final tersimpan dengan output training sebagai bukti.
- [ ] **R-20a** Training hanya pada train. Tidak ada valid/test atau pseudo-label di training.
- [ ] **R-20b** Pemilihan model dan hyperparameter hanya dari valid. Test dievaluasi sekali.
- [ ] **R-20c** Tidak ada pencocokan gambar test dengan train.
- [ ] **NFR-04** Tidak ada token atau kredensial di notebook dan repo.
- [ ] **NFR-05** Dataset tidak ada di repo.

---

## 5. Checklist administrasi dan submission

### Administrasi (setiap anggota)
- [ ] Biaya pendaftaran Rp135.000 lunas.
- [ ] KTM aktif terunggah (PDF/JPG/PNG, maks 2 MB).
- [ ] Follow Instagram `@telepatipolban`.
- [ ] Twibbon diunggah di Instagram pribadi, mention `@telepatipolban`, tag minimal 3 teman.
- [ ] Username GitHub terdaftar di dashboard.
- [ ] Masuk grup WhatsApp peserta.

### Berkas (R-16)
- [ ] **Surat Pernyataan Orisinalitas**: PDF scan, meterai Rp10.000, tanda tangan Ketua dan Dosen Pembimbing.
- [ ] Repo **publik**, dicek dari browser yang tidak login.
- [ ] Link weights bisa diunduh tanpa login.
- [ ] Skor mandiri dari latihan audit tercatat di README.

### Pengiriman
- [ ] **Submission cadangan** dikirim begitu model pertama lolos latihan audit (D-12).
- [ ] Submission final dikirim lewat form website **sebelum 25 Okt 2026 18:00 WIB** (batas resmi 23:59).
- [ ] Bukti pengiriman (tangkapan layar) disimpan.

---

## 6. Kalau latihan audit gagal

| Gejala | Tindakan |
|---|---|
| Weights tidak termuat | Cek versi ultralytics, pin versi yang dipakai saat training |
| Skor berbeda antar-jalankan | Cek seed, `deterministic`, `half=False`, `batch` tetap |
| Skor berbeda dari Kaggle/sesi lain | Laporkan angka dari latihan audit Colab sebagai skor mandiri |
| Waktu > 10 menit | Kurangi overhead setup, jangan pakai TTA |
| Link weights minta login | Repo belum publik atau link bukan link Release |
