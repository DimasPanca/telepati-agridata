---
doc_id: GIT_WORKFLOW
judul: "Alur Kerja Git dan GitHub Tim"
versi: 1.0
tanggal: 2026-10-23
status: AKTIF
pemilik: Dimas
bergantung_pada: [01_PRD.md]
---

# Alur Kerja Git dan GitHub

> **Untuk manusia dan AI.** Guidebook mewajibkan satu repo publik dengan riwayat commit **bertahap dari akun anggota terdaftar** (R-12, R-14). Commit tunggal berisiko REJECTED. Dokumen ini memastikan riwayat commit rapi, nyata, dan bisa dilacak.

Repo: `https://github.com/DimasPanca/telepati-agridata`

---

## 1. Persiapan satu kali (setiap anggota)

1. Akun GitHub sesuai username yang didaftarkan ke panitia.
2. Terima undangan collaborator dari Dimas (Settings → Collaborators).
3. Pasang Git, lalu atur identitas. **Email harus email akun GitHub**, kalau tidak commit tidak tercatat atas nama akun:

```bash
git config --global user.name "Nama Lengkap"
```

```bash
git config --global user.email "email-akun-github@contoh.com"
```

4. Clone repo:

```bash
git clone https://github.com/DimasPanca/telepati-agridata.git
```

---

## 2. Aturan cabang

| Branch | Pemilik | Isi |
|---|---|---|
| `main` | Dimas | Notebook, tabel `RUNS`, dokumen, hasil yang sudah ditinjau, log Run C |
| `run-a` | Rahman | Log Run A, tabel gabungan evaluasi valid |
| `run-b`, `audit` | Maman | Log Run B, catatan latihan audit |

- Hanya Dimas yang mengubah notebook, termasuk tabel `RUNS` (D-11, D-15).
- Rahman dan Maman bekerja di branch masing-masing, lalu membuat **pull request** ke `main`. Dimas meninjau dan melakukan merge.

---

## 3. Alur harian

```bash
git pull
```

```bash
git checkout -b run-a
```

Commit kecil setiap satu bagian selesai, lalu:

```bash
git push -u origin run-a
```

Buat pull request di GitHub, tulis ringkasan memakai template di `04_EXPERIMENT_PLAN.md` §7 atau §8.

---

## 4. Aturan commit

- **Satu commit = satu perubahan yang jelas.** Contoh: `Tambah pemetaan 21 ke 11 kelas`, `Log Run A segmen 1`, `Catatan latihan audit v1.0`.
- **Commit pekerjaan nyata dari akun sendiri.** Jangan membuat commit kosong atau commit atas nama orang lain.
- **Commit dari Colab** lewat File → Save a copy in GitHub tercatat atas nama akun GitHub yang login.
- **Jangan menumpuk** semua perubahan lalu commit sekali di hari terakhir.
- Pesan commit dalam bahasa Indonesia, kalimat perintah singkat.

---

## 5. Yang tidak boleh masuk repo

Sudah dicegah oleh `.gitignore`, tetapi tetap periksa sebelum commit:

| Jangan di-commit | Alasan |
|---|---|
| Dataset, zip, folder `Telepati 8.0 Datasets/` | NFR-05 |
| `__MACOSX/`, `.DS_Store` | Sampah sistem |
| `runs/`, `*.pt`, `*.onnx`, `*.h5`, `*.pkl` | Weights lewat Release (D-08) |
| `kaggle.json`, token apa pun | NFR-04 |

Yang **boleh** dari hasil training: `results.csv`, `args.yaml`, tabel evaluasi, catatan audit.

---

## 6. Tag dan rilis

| Tag | Arti | Dibuat oleh |
|---|---|---|
| `v0.1-core` | Inti pipeline lolos uji alur, siap dipakai trainer | Dimas |
| `v1.0` | Model cadangan pertama lolos latihan audit | Dimas |
| `v1.1`, `v1.2` | Model lebih baik berdasarkan valid | Dimas |

Membuat Release weights:
1. GitHub → Releases → Create a new release.
2. Tag sesuai tabel, judul, catatan: run, epoch, E1/E2 valid, SHA-256.
3. Lampirkan `telepati_agridata_v<X.Y>.pt` di Attach binaries, lalu Publish.
4. Salin direct link:

```text
https://github.com/DimasPanca/telepati-agridata/releases/download/v1.0/telepati_agridata_v1.0.pt
```

5. Uji dari browser tanpa login. Repo harus sudah publik.

---

## 7. Visibilitas repo

- Repo boleh privat selama pengembangan.
- **Wajib publik** paling lambat saat submit, dan tetap publik selama audit (26 Okt – 1 Nov) dan babak final.
- Riwayat commit tetap utuh saat repo diubah ke publik.
