# Vulcan League AI Engine (VLAE)

**A command-line interface for training, evaluating, and deploying UNET-based ink detection models on carbonized papyrus scrolls from the [Vesuvius Challenge](https://scrollprize.org/).**

---

## Background

In 79 AD, the eruption of Mount Vesuvius carbonized a villa full of ancient papyrus scrolls, making them impossible to unroll by traditional means. The [Vesuvius Challenge](https://scrollprize.org/) is an open international effort to recover the text of these scrolls using CT scanning and machine learning; specifically, semantic segmentation models that detect ink on virtually unwrapped scroll surfaces.

Leading ink detection models (UNETs, UNETRs) require large commercial GPUs and significant compute, creating a barrier for community contributors. VLAE addresses this directly: it packages trained 3D convolutional models into a reproducible CLI tool that any contributor can run, regardless of hardware background.

This project was the subject of an AP Research study investigating the effects of compression and pruning on UNET efficiency for low-resource ink detection. Key finding: filter compression reduced FLOPs by ~73.6% with a ~3 times improvement in efficiency score (F-beta / FLOPs) compared to baseline.

---

## What VLAE Does

VLAE wraps the full model pipeline of data ingestion, model training, evaluation, and segment management into a single CLI application. Commands are triggered through `main.py` and routed to modular command files. The application auto-detects GPU availability (CUDA, Apple MPS, or CPU) and auto-installs dependencies on first run.

The model baked into is a **3D convolutional encoder + MLP decoder** architecture that operates on 48×64×64 voxel subvolumes extracted from CT scan TIFF stacks of Vesuvius scroll fragments. Three model variants are supported: baseline, compressed (filter reduction), and pruned (L1 unstructured, 30th percentile).

---

## Installation

**Requirements:** Python 3.7-3.9 (Avoid using later versions)

```bash
# Clone the repository
git clone https://github.com/rishitv26/VulcanLeagueAIEngine.git
cd VulcanLeagueAIEngine

# Run the application — dependencies install automatically on first launch
python main.py
```

On first run, VLAE detects your hardware (CUDA / MPS / CPU) and installs required packages automatically via `install.py` and `requirements.txt`.

**Dependencies** (auto-installed): PyTorch, NumPy, Pandas, Matplotlib, and Kaggle API client.

---

## Dataset

VLAE uses the **Vesuvius Challenge Ink Detection dataset**, available on [Kaggle](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection). It includes three labeled training fragments, each containing:

- Surface volume TIFF stacks (CT scans of carbonized scroll segments)
- Binary mask images indicating valid regions
- Ground-truth ink labels from infrared scanning

When using the `train` command, the data will autoinstall. However, if need be, it is possible to manually install the data from the Kaggle dataset and put it in the respective directory.

---

## CLI Commands

Run `python main.py` twice within the desired environment.

The first time, the program will successfully install all dependencies and quit
The second time, you will be greeted with a commandline interface as follows:

```
Loading...
...
Welcome to the VLAE (Vulcan League AI Engine) <version number>
type 'help' to see the list of commands.
type 'manual' for a basic tutorial on what to do.
[AI] Condition: 'baseline' | Filters: [16, 32, 64] | Device: <device>
[AI] FLOPs per forward pass: <FLOPs for your system>
>>> 
```

The following commands will be accessible from there:

| Command | Description |
|---|---|
| `clear/cls` | Clear the console output |
| `exit` | Stop the VLAE routine |
| `manual` | Basic instructions for running VLAE |
| `change-setting <setting> <new value>` | Change a setting variable manually |
| `get-setting <setting>` | Get the value of a given setting |
| `get-all-settings` | Get all changeable settings |
| `add-segment <dir>` | Copy a segment into the `test` subfolder for ink detection |
| `rm-segment <name>` | Delete a segment from the `test` subfolder |
| `train` | Train the model using `training_data` (must be comma-separated values of 1, 2, or 3; no repeats) |
| `eval` | Run the model on data from the `test` subfolder |

---

## Model Architecture

All three variants share the same base architecture: a **3D convolutional encoder** feeding into an **MLP decoder**.

**Encoder:** Three sequential Conv3D blocks (kernel size 3, stride 2, padding 1), each followed by Batch Normalization and ReLU activation. An adaptive average pooling layer collapses spatial dimensions to a single feature vector.

**Decoder:** Two fully connected hidden layers (128 units each, ReLU), followed by a single-neuron output layer producing a scalar ink probability in [0.0, 1.0].

**Training objective:** BCEWithLogitsLoss. **Optimizer:** SGD + OneCycleLR scheduler. **Metric:** F-beta score (β=0.5).

| Variant | Filter sizes | Notes |
|---|---|---|
| Baseline | [16, 32, 64] | Control model |
| Compressed | [8, 16, 32] | ~73.6% FLOPs reduction; ~3 times efficiency score improvement |
| Pruned | [16, 32, 64] + L1 pruning | 30th percentile weight removal + 10K fine-tune steps |

---


## Output Files

Each training and evaluation run produces:

- **`training_log.csv`** — Per-step log: condition label, fragment ID, batch loss, step index, batch F-beta score, elapsed time
- **`run_summary.csv`** — Per-run summary: total steps, training time, FLOPs per iteration, peak RAM, batch size, learning rate, timestamp
- **`output_fragment_X.png`** — Binary ink prediction image for evaluated fragment

---

## Research

This codebase was used in an AP Research study examining the effect of segmentation architecture optimizations (compression vs. pruning) on ink detection efficiency for the Vesuvius Challenge.

**Key results:**
- Compressed model achieved **~73.6% reduction in training FLOPs** and **~71% reduction in evaluation FLOPs** vs. baseline
- Compressed model's efficiency score (F-beta / FLOPs) was **~3 times higher** than baseline on both test fragments
- Pruning at 30th percentile increased total training FLOPs by ~17% with no measurable efficiency gain, suggesting conservative pruning serves weight regularization rather than computational reduction

**Conclusion:** Filter compression is the preferred method for making ink detection accessible to low-resource community contributors.

### Baseline Model Results:
<img width="640" height="275" alt="image_0 4_0" src="https://github.com/user-attachments/assets/7319c4bd-978e-4f9b-8307-97d03f383bfe" />

### Compressed Model Results:
<img width="640" height="275" alt="image_0 4_0 (1)" src="https://github.com/user-attachments/assets/ea4c3d30-44fc-4c32-86cb-c4dd6a7623e8" />

### Pruned Model Results:
<img width="640" height="275" alt="image_0 4_0 (2)" src="https://github.com/user-attachments/assets/82214964-4dc9-4c40-a140-d191477b0e97" />
<br><br>

> Note that although the compressed model had a worse F-beta, the calculated effeciency score (F-beta / FLOPs) was 3 times better than baseline and pruned.


---

## Useful References

- [Vesuvius Challenge](https://scrollprize.org/)
- [Volume Cartographer](https://github.com/educelab/volume-cartographer)
- [VolumeAnnotate](https://github.com/MosheLevy20/VolumeAnnotate)
- [Kaggle Ink Detection Dataset](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection)

---
