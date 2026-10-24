# telepati-agridata

Rice leaf disease detection (11 classes) for AI AgriData Intelligence Race, TELEPATI 8.0.
YOLO11s trained from scratch on the official Agritech Data dataset.

| | |
|---|---|
| Team | The Power of Kepepet, Universitas Sultan Ageng Tirtayasa |
| Members | Dimas (lead), Rahman, Maman |
| Advisor | Arief Rahman, S.Kom., M.Kom |
| Notebook | [notebooks/agridata_pipeline.ipynb](notebooks/agridata_pipeline.ipynb) ([Colab](https://colab.research.google.com/github/DimasPanca/telepati-agridata/blob/main/notebooks/agridata_pipeline.ipynb)) |
| Weights | [telepati_agridata_v1.0.pt](https://github.com/DimasPanca/telepati-agridata/releases/download/v1.0/telepati_agridata_v1.0.pt) |
| Test macro F1 | 0.6951 |

## How to Run

### Inference only (reproduce the reported score)

1. Open the notebook in Google Colab and select a **T4 GPU** runtime.
2. In the config cell (section 0), set:
   ```python
   MODE = "inference_only"
   DATA_ZIP = "<path to the official dataset zip or extracted folder>"
   ```
   `WEIGHTS_URL`, `WEIGHTS_SHA256`, and `E2_CONF` are already set.
3. Runtime > Run all.

Output (section 11): macro F1 (E1, E2), mAP@50, per-class F1, and runtime. Inference takes about 10 ms per image on T4.

### Full pipeline (train from raw data)

1. Set `MODE = "full"` and `DATA_ZIP` / `RUNS_DIR`.
2. Run with `RUN_ID = "A_yolo11s_s42"`, then `RUN_ID = "A_fit_yolo11s_s42"`.
3. Each run evaluates all checkpoints on the valid split (section 8). The test split is evaluated once with `RUN_FINAL_TEST = True` (section 10).

Interrupted runs resume from `last.pt` when re-run with the same `RUN_ID`.

## Pipeline

| Section | Step |
|---|---|
| 0 to 1 | Config, environment, seed |
| 2 | Extract raw zip, validate image and annotation counts |
| 3 | Map 21 COCO categories to 11 canonical classes |
| 4 | Clean train annotations (clip to image, drop boxes < 2 px, drop duplicates) |
| 5 to 6 | Convert COCO to YOLO format, write `data.yaml` (train and valid only) |
| 7 | Train (`RUNS` table) |
| 8 to 9 | Evaluate checkpoints on valid, select final model and `E2_CONF` |
| 9b | Optional: checkpoint averaging and per-class bias calibration |
| 10 | Evaluate test once |
| 11 | Inference from released weights |

## Dependencies

| Package | Version |
|---|---|
| ultralytics | 8.4.184 (pinned in [requirements.txt](requirements.txt), installed by the notebook) |
| Python | 3.13.16 (Colab) |
| torch | 2.11.0+cu130 (Colab) |

## Model

| | |
|---|---|
| Architecture | YOLO11s (`yolo11s.yaml`), anchor-free detect head on P3, P4, P5 |
| Parameters | 9,417,057 (fused) |
| GFLOPs | 21.4 at 640 x 640 |
| Initialization | Random (`pretrained=False`) |
| Classes | 11 |

## Hyperparameters

| Parameter | Stage 1: `A_yolo11s_s42` | Stage 2: `A_fit_yolo11s_s42` |
|---|---|---|
| Init weights | random | Stage 1 `best.pt` |
| imgsz | 640 | 640 |
| epochs | 150 | 50 (selected: epoch 20) |
| batch | 16 | 16 |
| optimizer | auto (MuSGD), lr0 0.01 | SGD, lr0 0.002, lrf 0.1, warmup_epochs 1 |
| mosaic / close_mosaic | 1.0 / 10 | 0.0 / 0 |
| mixup | 0.1 | 0.0 |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 | 0 / 0 / 0 |
| scale / translate | 0.5 / 0.1 | 0.25 / 0.05 |
| fliplr / flipud | 0.5 / 0.5 | 0.5 / 0.5 |
| seed / deterministic | 42 / True | 42 / True |
| patience | 0 | 0 |
| save_period | 10 | 10 |

Evaluation: `conf=0.001`, NMS `iou=0.7`, `max_det=300`, FP32. `E2_CONF=0.26` selected on valid.

## Weights

| Version | Download | SHA-256 |
|---|---|---|
| v1.0 | [telepati_agridata_v1.0.pt](https://github.com/DimasPanca/telepati-agridata/releases/download/v1.0/telepati_agridata_v1.0.pt) | `cdd815e88a22b6d9d6ee205dba7e51944e9c563177f6bdab01d1d984cc373b67` |

## Results

| Split | Macro F1 (E1) | Macro F1 @ conf 0.26 (E2) | mAP@50 |
|---|---|---|---|
| valid | 0.6981 | 0.6930 | 0.6884 |
| test | 0.6951 | 0.6951 | 0.6898 |

E1: macro F1 at the confidence that maximizes mean F1 (Ultralytics `val`). E2: macro F1 at a fixed confidence. Test scores were reproduced on Colab T4, see [results/audit_log.md](results/audit_log.md).

Per-class F1 (test):

| Class | F1 | Class | F1 |
|---|---|---|---|
| Bacterial leaf blight | 0.4315 | Leaf roller | 0.8613 |
| Bacterial panicle blight | 0.8547 | Leaf scald | 0.5059 |
| Blast | 0.5167 | Narrow brown | 0.9986 |
| Brown spot | 0.3431 | Sheath blight | 0.6065 |
| False smut | 0.9133 | Tungro | 0.7042 |
| Healthy | 0.9104 | | |

Runs compared on valid:

| Run | Description | Valid macro F1 |
|---|---|---|
| A_yolo11s_s42 | from scratch, 150 epochs | 0.6835 (`best.pt`) |
| A800_yolo11s_s42 | Run A continued at 800 px, 40 epochs | 0.6887 |
| A_fit_yolo11s_s42 | Run A continued without heavy augmentation, 50 epochs | 0.6981 (epoch 20), released as v1.0 |

Training logs: [results/](results/).

## Notes

- Training uses the train split only. Checkpoints and thresholds are selected on valid. Test is evaluated once per submitted model.
- No external pretrained weights. Stage 2 starts from Stage 1 weights trained by the team.
- All preprocessing runs inside the notebook from the raw dataset. The dataset is not included in this repository.
