================================================================================
 README TIM - PANDUAN MEMBACA DOKUMEN
 AI AgriData Intelligence Race - TELEPATI 8.0
 Diperbarui: 23 Oktober 2026
 Batas submission: 25 Oktober 2026, 23:59 WIB (target tim: 18:00 WIB)
================================================================================

File ini untuk TIM (Dimas, Rahman, Maman). Bukan untuk juri.
README.md di root repo adalah untuk juri.

--------------------------------------------------------------------------------
 1. URUTAN MEMBACA (WAJIB, SEMUA ANGGOTA)
--------------------------------------------------------------------------------

  Langkah  File                        Waktu    Kenapa
  -------  --------------------------  -------  --------------------------------
  1        docs/README_TIM.txt         5 mnt    Peta dokumen (file ini)
  2        docs/01_PRD.md              20 mnt   Aturan lomba, tujuan, keputusan,
                                                peran. SUMBER KEBENARAN UTAMA.
  3        docs/03_DATA_CARD.md        10 mnt   Fakta dataset, pemetaan 21->11
  4        Dokumen sesuai peran (lihat bagian 2)

Kalau ada yang bertentangan:
  aturan resmi panitia  >  01_PRD.md  >  dokumen lain

--------------------------------------------------------------------------------
 2. BACAAN SESUAI PERAN
--------------------------------------------------------------------------------

 DIMAS (Arsitek: fundamental + computer vision, Ketua Tim)
   - 02_SYSTEM_DESIGN.md   seluruhnya, terutama bagian 4, 5, 6, 7, 11
   - 04_EXPERIMENT_PLAN.md seluruhnya
   - 06_GIT_WORKFLOW.md    bagian 2 dan 6 (merge, tag, rilis)
   - 05_AUDIT_RUNBOOK.md   bagian 4 dan 5 (kepatuhan, submission)

 RAHMAN (Trainer Run A + analis hasil valid semua run)
   - 04_EXPERIMENT_PLAN.md bagian 2, 3, 5, 6, 7, 8
   - 02_SYSTEM_DESIGN.md   bagian 2 (Colab dan Drive), 7, 8
   - 06_GIT_WORKFLOW.md    seluruhnya

 MAMAN (Trainer Run B + penjaga reproduksibilitas + latihan audit)
   - 05_AUDIT_RUNBOOK.md   seluruhnya
   - 04_EXPERIMENT_PLAN.md bagian 2, 3, 5, 7
   - 02_SYSTEM_DESIGN.md   bagian 2, 9
   - 06_GIT_WORKFLOW.md    seluruhnya

--------------------------------------------------------------------------------
 3. DAFTAR DOKUMEN
--------------------------------------------------------------------------------

  01_PRD.md             Apa yang dibuat dan kenapa. Semua aturan lomba (R-xx),
                        kebutuhan (REQ-xx), keputusan (D-xx), peran, jadwal,
                        risiko, pertanyaan terbuka.
  02_SYSTEM_DESIGN.md   Bagaimana pipeline disusun: Colab, Drive, struktur
                        notebook (S-00 s/d S-11), data, training, evaluasi,
                        pemilihan model, jalur inference-saja.
  03_DATA_CARD.md       Fakta dataset: jumlah per split, pemetaan kelas,
                        jumlah box per kelas, duplikasi, SHA-256 zip.
  04_EXPERIMENT_PLAN.md Run A, B, C: siapa, apa, kapan, cara mencatat.
  05_AUDIT_RUNBOOK.md   Latihan audit di Colab T4, checklist kepatuhan,
                        checklist administrasi dan submission.
  06_GIT_WORKFLOW.md    Aturan branch, commit, pull request, tag, rilis.

--------------------------------------------------------------------------------
 4. HAL YANG HARUS DIPAHAMI SEMUA ORANG (VERSI SINGKAT)
--------------------------------------------------------------------------------

  [1] Training HANYA pakai split train. Valid untuk memilih model.
      Test dievaluasi SEKALI di akhir. Melanggar ini = REJECTED.
  [2] Jangan pernah mencocokkan gambar test dengan train.
  [3] Model dibangun dari file .yaml, pretrained=False. Tidak ada bobot luar.
  [4] Seed 42 di semua tahap.
  [5] Platform utama Google Colab T4 (sama dengan lingkungan audit juri).
      Kaggle hanya cadangan.
  [6] Checkpoint disimpan ke Google Drive tiap epoch. Weights FINAL hanya
      lewat GitHub Releases, tidak lewat Google Drive.
  [7] Dataset tidak boleh diolah lewat API LLM (dilarang keras).
  [8] Commit pekerjaan nyata dari akun GitHub sendiri, bertahap.
  [9] Dataset tidak boleh masuk repo atau dibagikan publik.
  [10] Notebook (termasuk tabel RUNS) hanya diubah Dimas. Trainer cukup
       mengisi RUN_ID di sel konfigurasi. Log dan hasil lewat pull request.

--------------------------------------------------------------------------------
 5. LANGKAH PERTAMA HARI INI (23 OKT)
--------------------------------------------------------------------------------

 SEMUA
   [ ] Baca bagian 1 dan 2 file ini, lalu 01_PRD.md
   [ ] Terima undangan collaborator GitHub, atur git config (06_GIT_WORKFLOW)
   [ ] Cek administrasi lomba (05_AUDIT_RUNBOOK.md bagian 5)

 DIMAS
   [ ] Jalankan notebooks/agridata_pipeline.ipynb dengan RUN_ID = "E0_smoke"
   [ ] Kalau lolos: tag v0.1-core, kabari Rahman dan Maman
   [ ] Mulai Run C (RUN_ID = "C_yolo11s_bal_s42")
   [ ] Urus surat orisinalitas (meterai, tanda tangan dosen)

 RAHMAN
   [ ] Unggah telepati-agridata-raw.zip ke MyDrive/telepati/data/
   [ ] Siapkan Colab T4 di akun sendiri
   [ ] Setelah v0.1-core: mulai Run A (RUN_ID = "A_yolo11s_s42")

 MAMAN
   [ ] Unggah telepati-agridata-raw.zip ke MyDrive/telepati/data/
   [ ] Siapkan Colab T4 di akun sendiri
   [ ] Setelah v0.1-core: mulai Run B (RUN_ID = "B_yolo26s_s42")

 Cara menjalankan run (semua trainer):
   1. Buka notebook dari GitHub di Colab, pilih T4 GPU
   2. Di sel bagian 0: MODE = "full", RUN_ID = <run kamu>
   3. Runtime -> Run all
   4. Sesi putus? Run all lagi dengan RUN_ID yang sama (otomatis resume)

 Zip dataset siap-upload (dari Dimas):
   telepati-agridata-raw.zip, 613.048.556 byte
   SHA-256: 797e47749f91f4403e778dd0150716da1cd94cddbecee2ec56cf5545c50cec50
   Bagikan secara privat (Drive pribadi ke email anggota), jangan publik.

--------------------------------------------------------------------------------
 6. KALAU BINGUNG
--------------------------------------------------------------------------------

  - Soal aturan lomba          -> 01_PRD.md bagian 4
  - Soal "boleh tidak pakai X" -> 01_PRD.md bagian 4.5 (R-20) dan bagian 6
  - Soal cara menjalankan run  -> 04_EXPERIMENT_PLAN.md bagian 5
  - Soal sesi Colab putus      -> 02_SYSTEM_DESIGN.md bagian 2, resume dari
                                  last.pt
  - Soal git                   -> 06_GIT_WORKFLOW.md
  - Masih bingung              -> tanya Dimas sebelum mengubah apa pun

================================================================================
