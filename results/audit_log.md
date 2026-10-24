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
