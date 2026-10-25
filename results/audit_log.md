# Log latihan audit

## v1.0, 24 Oktober 2026

| | |
|---|---|
| Pelaksana | Dimas |
| Platform | Google Colab, GPU T4 (runtime baru) |
| Notebook | `notebooks/agridata_pipeline.ipynb`, `MODE = "inference_only"` |
| Versi library | Python 3.13.16, torch 2.11.0+cu130, ultralytics 8.4.184 |
| Release weights | `v1.0`, `telepati_agridata_v1.0.pt` |
| SHA-256 weights | `cdd815e88a22b6d9d6ee205dba7e51944e9c563177f6bdab01d1d984cc373b67` (cocok) |
| `E2_CONF` | 0.26 |

| Jalankan | E1 | E2 | mAP@50 |
|---|---|---|---|
| 1 | 0.6951 | 0.6951 | 0.6898 |
| 2 | 0.6951 | 0.6951 | 0.6898 |
| Pembanding (CPU lokal, file yang sama) | 0.6951 | 0.6951 | 0.6898 |

F1 per kelas identik di ketiga pengukuran. Inference sekitar 10 ms per gambar di T4.

**Status: LOLOS.** Selisih antar-jalankan 0, selisih antar-hardware 0 (toleransi audit ±0,02).

## v1.2, 24 Oktober 2026

| | |
|---|---|
| Pelaksana | Dimas |
| Platform | Google Colab, GPU T4 |
| Notebook | `notebooks/agridata_pipeline.ipynb` dari branch `v1.2`, `MODE = "inference_only"` |
| Versi library | ultralytics 8.4.184 |
| Release weights | `v1.0`, `telepati_agridata_v1.0.pt` (v1.2 memakai bobot yang sama) |
| SHA-256 weights | `cdd815e88a22b6d9d6ee205dba7e51944e9c563177f6bdab01d1d984cc373b67` (cocok) |
| Setelan inference | NMS `iou=0.45` per kelas, `E2_CONF = 0.20` |

| Jalankan | E1 | E2 | mAP@50 |
|---|---|---|---|
| 1 (Colab T4) | 0.7023 | 0.6963 | 0.6952 |
| Pembanding (CPU lokal, file yang sama) | 0.7023 | 0.6963 | 0.6952 |

F1 per kelas identik di kedua pengukuran. Unduh weights, inference, dan evaluasi selesai dalam 33,9 detik di T4.

**Status: LOLOS.** Selisih antar-hardware 0 (toleransi audit ±0,02).

## v1.3, 25 Oktober 2026

| | |
|---|---|
| Pelaksana | Dimas |
| Platform | Google Colab, GPU T4 |
| Notebook | `notebooks/agridata_pipeline.ipynb` dari tag `v1.3`, `MODE = "inference_only"` |
| Versi library | ultralytics 8.4.184 |
| Release weights | `v1.3`, `telepati_agridata_v1.3.pt` (19.214.713 byte) |
| SHA-256 weights | `04d2f640c7370988415a8594c8a7658e4dee86267c6a0cfb3cc9bc45936b1e7a` (cocok, diperiksa notebook dan diunduh ulang dari link Release) |
| Setelan inference | TTA (`augment=True`), NMS `iou=0.45` per kelas, `E2_CONF = 0.25` |

| Jalankan | E1 | E2 | mAP@50 |
|---|---|---|---|
| 1 (Colab T4) | 0.7123 | 0.7053 | 0.7038 |

| Kelas | F1 | Kelas | F1 |
|---|---|---|---|
| Bacterial leaf blight | 0.4878 | Leaf roller | 0.8688 |
| Bacterial panicle blight | 0.8540 | Leaf scald | 0.5291 |
| Blast | 0.5282 | Narrow brown | 0.9942 |
| Brown spot | 0.3981 | Sheath blight | 0.6123 |
| False smut | 0.9316 | Tungro | 0.7005 |
| Healthy | 0.9308 | | |

Unduh weights, inference dengan TTA, dan evaluasi selesai dalam 64,8 detik di T4 (23,9 ms inference per gambar). Ini evaluasi test pertama dan satu-satunya untuk v1.3, dilakukan setelah keputusan D-21 di-push.

**Status: LOLOS.** Notebook berjalan dari weights rilis tanpa training, checksum cocok, waktu jauh di bawah batas 30 menit.
