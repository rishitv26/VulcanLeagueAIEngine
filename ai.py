#
# Credit: https://www.kaggle.com/code/danielhavir/vesuvius-challenge-example-submission
#
############################################## Imports:
import os
import gc
import csv
import tracemalloc
from pathlib import Path
from typing import List, Tuple
import warnings
import util
import shutil
import opendatasets as od
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import PIL.Image as Image
from sklearn.metrics import fbeta_score
from sklearn.exceptions import UndefinedMetricWarning
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.utils.data as thd
from tqdm import tqdm

from config import Config

################################################## Logger:
class Logger:
    """
    Handles all CSV logging for the experiment. Produces three files:
      - training_log.csv  : one row per training step, all conditions appended
      - run_summary.csv   : one row per completed training run
      - eval_results.csv  : one row per fragment per eval run
    
    All files land in base_path/logs/. Multiple conditions accumulate in the
    same files across runs so you can open them once and compare everything.
    """

    STEP_FIELDS = [
        "condition", "fragments", "step",
        "loss", "accuracy", "fbeta",
        "elapsed_seconds", "step_duration_seconds",
    ]

    SUMMARY_FIELDS = [
        "condition", "fragments", "total_steps",
        "total_training_seconds", "peak_ram_mb",
        "batch_size", "learning_rate",
        "flops_per_forward", "total_training_flops",
        "timestamp",
    ]

    EVAL_FIELDS = [
        "condition", "fragments", "fragment_name",
        "threshold", "fbeta",
        "flops_per_pixel", "total_eval_flops",
        "timestamp",
    ]

    def __init__(self, base_path: str):
        self.log_dir = os.path.join(base_path, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        self._step_path    = os.path.join(self.log_dir, "training_log.csv")
        self._summary_path = os.path.join(self.log_dir, "run_summary.csv")
        self._eval_path    = os.path.join(self.log_dir, "eval_results.csv")

        # Write headers only if the files don't exist yet, so appending
        # across multiple runs never duplicates the header row.
        self._ensure_header(self._step_path,    self.STEP_FIELDS)
        self._ensure_header(self._summary_path, self.SUMMARY_FIELDS)
        self._ensure_header(self._eval_path,    self.EVAL_FIELDS)

        # In-memory buffer — flushed to disk every 500 steps so frequent
        # writes don't meaningfully slow down the training loop.
        self._step_buffer: list[dict] = []

    def _ensure_header(self, path: str, fields: list[str]):
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    # ------------------------------------------------------------------
    # Called once per training step from train_model()
    # ------------------------------------------------------------------
    def log_step(self, row: dict):
        self._step_buffer.append(row)

    def flush_steps(self):
        if not self._step_buffer:
            return
        with open(self._step_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.STEP_FIELDS)
            writer.writerows(self._step_buffer)
        self._step_buffer.clear()

    # ------------------------------------------------------------------
    # Called once at the end of train_model()
    # ------------------------------------------------------------------
    def log_summary(self, row: dict):
        self.flush_steps()  # make sure nothing is left in the buffer
        with open(self._summary_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.SUMMARY_FIELDS).writerow(row)
        print(f"[Logger] Run summary written to {self._summary_path}")

    # ------------------------------------------------------------------
    # Called once per fragment from eval_model()
    # ------------------------------------------------------------------
    def log_eval(self, row: dict):
        with open(self._eval_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.EVAL_FIELDS).writerow(row)
        print(f"[Logger] Eval result written to {self._eval_path}")


################################################## FLOPs counter:
def _compute_flops(model: torch.nn.Module, input_shape: tuple) -> int:
    """
    Count FLOPs for a single forward pass through the model without any
    external libraries, by registering temporary hooks on Conv3d and Linear layers.

    FLOPs = 2 × MACs (each multiply-accumulate counts as 2 operations).

    Conv3d FLOPs per output element:
        2 × C_in × kD × kH × kW   (one dot product per output spatial position)
    Linear FLOPs per output element:
        2 × in_features

    BatchNorm and activation layers are omitted — their FLOPs are negligible
    (a few additions/multiplications per element) compared to conv/linear layers.

    Args:
        model:       The nn.Module to profile. Must be on CPU for the dummy pass.
        input_shape: Full input tensor shape including batch dim, e.g. (1, 1, 48, 64, 64).

    Returns:
        Total FLOPs as an integer.
    """
    flops_count = [0]
    hooks = []

    def conv3d_hook(module, inp, output):
        # inp[0]:  (batch, C_in,  D,     H,     W    )
        # output:  (batch, C_out, D_out, H_out, W_out)
        batch = inp[0].shape[0]
        C_in  = inp[0].shape[1]
        C_out, D_out, H_out, W_out = (output.shape[1], output.shape[2],
                                       output.shape[3], output.shape[4])
        kD, kH, kW = (module.kernel_size if isinstance(module.kernel_size, tuple)
                      else (module.kernel_size,) * 3)
        # MACs = batch × C_out × D_out × H_out × W_out × (C_in/groups) × kD × kH × kW
        macs = batch * C_out * D_out * H_out * W_out * (C_in // module.groups) * kD * kH * kW
        flops_count[0] += 2 * macs

    def linear_hook(module, inp, output):
        batch = inp[0].shape[0]
        macs  = batch * module.in_features * module.out_features
        flops_count[0] += 2 * macs

    # Register hooks on every Conv3d and Linear in the model
    for module in model.modules():
        if isinstance(module, nn.Conv3d):
            hooks.append(module.register_forward_hook(conv3d_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    # Save the device the model currently lives on BEFORE moving it to CPU,
    # because model.cpu() is in-place — cpu_model and model are the same object.
    # Reading the device after the move would always return CPU, so the
    # "restoration" would silently leave the model on CPU for all future calls.
    original_device = next(model.parameters()).device

    dummy     = torch.zeros(input_shape)
    cpu_model = model.cpu()
    with torch.no_grad():
        cpu_model(dummy)

    # Restore model to whichever device it was on before (cuda / mps / cpu)
    model.to(original_device)

    for h in hooks:
        h.remove()

    return flops_count[0]


################################################## Data:
class SubvolumeDataset(thd.Dataset):
    def __init__(
        self,
        fragments: List[Path],
        voxel_shape: Tuple[int, int, int],
        load_inklabels: bool = True,
        filter_edge_pixels: bool = False,
    ):
        self.fragments = sorted(map(lambda path: path.resolve(), fragments))
        self.voxel_shape = voxel_shape
        self.load_inklabels = load_inklabels
        self.filter_edge_pixels = filter_edge_pixels

        # Load sequentially
        labels = []
        image_stacks = []
        valid_pixels = []
        for fragment_id, fragment_path in enumerate(self.fragments):
            fragment_path = fragment_path.resolve()  # absolute path
            mask = np.array(Image.open(Path.joinpath(fragment_path, "mask.png")).convert("1"))

            surface_volume_paths = sorted(
                (fragment_path / "surface_volume").rglob("*.tif")
            )
            z_dim, y_dim, x_dim = voxel_shape

            z_mid = len(surface_volume_paths) // 2
            # BUG FIX: z_start was allowed to go negative (e.g. z_mid=10, z_dim=48 → z_start=-14).
            # Python list slicing with a negative start wraps around and loads slices from the
            # WRONG end of the sorted TIF list — a completely silent data corruption bug.
            # Clamping to 0 ensures we always read from the beginning of the stack.
            z_start = max(z_mid - z_dim // 2, 0)
            z_end = z_start + z_dim

            # we don't convert to torch since it doesn't support uint16
            images = [
                np.array(Image.open(fn)) for fn in surface_volume_paths[z_start:z_end]
            ]
            image_stack = np.stack(images, axis=0)

            # BUG FIX: if the fragment has fewer TIF files than z_dim (e.g. 30 < 48),
            # the image_stack has wrong z-depth. __getitem__ uses image_stack[:, ...] which
            # passes the wrong z-shape into the subvolume, causing a broadcast exception.
            # PyTorch's DataLoader catches that exception and substitutes None, which then
            # triggers the "default_collate: found NoneType" error.
            # Fix: zero-pad the z-axis so image_stack.shape[0] is always exactly z_dim.
            if image_stack.shape[0] < z_dim:
                pad_z = z_dim - image_stack.shape[0]
                padding = np.zeros((pad_z, *image_stack.shape[1:]), dtype=image_stack.dtype)
                image_stack = np.concatenate([image_stack, padding], axis=0)

            image_stacks.append(image_stack)

            # BUG FIX: was np.uint16 — uint16 can't represent negative numbers,
            # which corrupts subtraction in __getitem__ padding logic AND in
            # filter_edge_pixels comparisons when coordinates are near 0.
            # int32 gives full signed arithmetic with no performance penalty.
            pixels = np.stack(np.where(mask == 1), axis=1).astype(np.int32)
            if filter_edge_pixels:
                height, width = mask.shape
                mask_y = np.logical_or(
                    pixels[:, 0] < y_dim // 2, pixels[:, 0] >= height - y_dim // 2
                )
                mask_x = np.logical_or(
                    pixels[:, 1] < x_dim // 2, pixels[:, 1] >= width - x_dim // 2
                )
                pixel_mask = np.logical_or(mask_y, mask_x)
                pixels = pixels[~pixel_mask]
            # encode fragment ID
            fragment_ids = np.full_like(pixels[:, 0:1], fragment_id)
            pixels = np.concatenate((pixels, fragment_ids), axis=1)
            valid_pixels.append(pixels)

            if load_inklabels:
                # binary mask can be stored as np.bool
                inklabels = (
                    np.array(Image.open(Path.joinpath(fragment_path, "inklabels.png"))) > 0
                )
                labels.append(inklabels)

            print(f"Loaded fragment {fragment_path} on {os.getpid()}")

        self.labels = labels
        self.image_stacks = image_stacks
        self.pixels = np.concatenate(valid_pixels).reshape(
            -1, valid_pixels[0].shape[-1]
        )

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        # Pixels are now stored as int32 (see __init__), so signed arithmetic
        # in the padding logic is always correct.
        center_y, center_x, fragment_id = self.pixels[index]
        center_y, center_x, fragment_id = int(center_y), int(center_x), int(fragment_id)
        z_dim, y_dim, x_dim = self.voxel_shape
        image_stack = self.image_stacks[fragment_id]
        _, height, width = image_stack.shape

        # pad with zeros if necessary
        if (
            center_y < y_dim // 2
            or center_x < x_dim // 2
            or center_y + y_dim // 2 >= height
            or center_x + x_dim // 2 >= width
        ):
            # calculate the upper-left corner of the sub-volume
            y_start = max(center_y - y_dim // 2, 0)
            x_start = max(center_x - x_dim // 2, 0)

            # calculate the lower-right corner of the sub-volume
            y_end = min(center_y + y_dim // 2, height)
            x_end = min(center_x + x_dim // 2, width)

            subvolume = np.zeros(self.voxel_shape, dtype=np.float32)

            pad_y_start = max(y_dim // 2 - center_y, 0)
            pad_x_start = max(x_dim // 2 - center_x, 0)

            pad_y_end = min(height + y_dim // 2 - center_y, y_dim)
            pad_x_end = min(width + x_dim // 2 - center_x, x_dim)

            subvolume[:, pad_y_start:pad_y_end, pad_x_start:pad_x_end] = (
                image_stack[:, y_start:y_end, x_start:x_end].astype(np.float32) / 65535
            )

        else:
            subvolume = (
                image_stack[
                    :,
                    center_y - y_dim // 2 : center_y + y_dim // 2,
                    center_x - x_dim // 2 : center_x + x_dim // 2,
                ]
            ).astype(np.float32) / 65535
        if self.load_inklabels:
            inklabel = float(self.labels[fragment_id][center_y, center_x])
        else:
            inklabel = -1.0

        return torch.from_numpy(subvolume).unsqueeze(0), torch.FloatTensor([inklabel])

    def plot_label(self, index, **kwargs):
        pixel = self.pixels[index]
        label = self.labels[pixel[-1]]

        print("Index:", index)
        print("Pixel:", pixel)
        print("Label:", int(label[pixel[0], pixel[1]]))

        if isinstance(label, torch.Tensor):
            label = label.numpy()

        fig, ax = plt.subplots(**kwargs)
        ax.imshow(label, cmap="gray")

        y, x, _ = pixel
        _, y_dim, x_dim = self.voxel_shape
        x_min = x - (x_dim // 2)
        x_max = x + (x_dim // 2)
        y_min = y - (y_dim // 2)
        y_max = y + (y_dim // 2)
        rect = Rectangle(
            (x_min, y_min), x_dim, y_dim, linewidth=2, edgecolor="y", facecolor="none"
        )
        ax.add_patch(rect)
        plt.show()
        plt.show()


################################################## Model:
# CONDITION FILTER SIZES:
#   baseline   → [16, 32, 64]  full-size CNN, control condition
#   compressed → [8,  16, 32]  half-width CNN, architecture compression condition
#   pruned     → [16, 32, 64]  same as baseline; pruning is applied post-training
FILTERS = {
    "baseline":   [16, 32, 64],
    "compressed": [8,  16, 32],
    "pruned":     [16, 32, 64],
}

class InkDetector(torch.nn.Module):
    def __init__(self, filters: list[int]):
        """
        3-layer 3D convolutional encoder + MLP decoder.
        Pass filters=[16,32,64] for baseline/pruned, [8,16,32] for compressed.
        The decoder input size is inferred automatically from the final filter count
        so changing filters here is the only thing needed to switch architectures.
        """
        super().__init__()

        paddings     = [1, 1, 1]
        kernel_sizes = [3, 3, 3]
        strides      = [2, 2, 2]

        layers = []
        in_channels = 1
        for num_filters, padding, kernel_size, stride in zip(filters, paddings, kernel_sizes, strides):
            layers.extend([
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(num_features=num_filters),
            ])
            in_channels = num_filters

        layers.append(nn.AdaptiveAvgPool3d(1))
        layers.append(nn.Flatten())

        self.encoder = nn.Sequential(*layers)
        # Decoder input size = final filter count (whatever it is)
        self.decoder = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

################################################## AI:
class AI:
    def __init__(self, batch_size: int, training_steps: int, learning_rate: float):
        self.batch_size      = batch_size
        self.training_steps  = training_steps
        self.learning_rate   = learning_rate

        config    = Config()
        condition = config.get("condition")

        self.train_run = config.get("trained") != "true"

        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.DEVICE = torch.device("mps")
        else:
            self.DEVICE = torch.device("cpu")

        # Build the right architecture for this condition.
        # "pruned" uses the same filters as "baseline" — pruning happens
        # after training inside prune_and_finetune(), not at construction time.
        if condition not in FILTERS:
            raise ValueError(
                f"Unknown condition '{condition}'. "
                f"Valid values are: {list(FILTERS.keys())}"
            )
        self.condition = condition
        self.model = InkDetector(filters=FILTERS[condition]).to(self.DEVICE)
        print(f"[AI] Condition: '{condition}' | Filters: {FILTERS[condition]} | Device: {self.DEVICE}")

        # Compute FLOPs for one forward pass once at construction time.
        # Subvolume shape is (1, 1, 48, 64, 64): batch=1, channel=1, z/y/x dims.
        # This is constant for the lifetime of this AI instance, so we only
        # run the dummy pass once here rather than repeatedly during training.
        self.flops_per_forward = _compute_flops(self.model, (1, 1, 48, 64, 64))
        print(f"[AI] FLOPs per forward pass: {self.flops_per_forward:,}")

        self.base_path = ""

    def set_basepath(self, path: str):
        self.base_path = path

    def train_model(self, train_loader, config: Config, condition: str, fragments: str):
        print("Model Generated, training model...")
        logger = Logger(self.base_path)

        all_labels = np.concatenate([train_dset.labels[i].flatten() for i in range(len(train_dset.labels))])
        pos = all_labels.sum()
        neg = len(all_labels) - pos
        pos_weight = torch.tensor([neg / pos]).to(self.DEVICE)
        print(f"[AI] pos_weight = {pos_weight.item():.2f} ({pos:.0f} ink / {neg:.0f} non-ink pixels)")
 
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, total_steps=self.training_steps)
        self.model.train()

        running_loss = 0.0
        running_accuracy = 0.0
        running_fbeta = 0.0
        denom = 0

        # Start RAM tracking and wall-clock timer before the first step.
        tracemalloc.start()
        train_start = time.time()
        step_start  = train_start

        pbar = tqdm(enumerate(train_loader), total=self.training_steps)
        for i, (subvolumes, inklabels) in pbar:
            if i >= self.training_steps:
                break

            optimizer.zero_grad()
            outputs = self.model(subvolumes.to(self.DEVICE))
            loss = criterion(outputs, inklabels.to(self.DEVICE))
            loss.backward()
            optimizer.step()
            scheduler.step()

            pred_ink  = outputs.detach().sigmoid().gt(0.4).cpu().int()
            accuracy  = (pred_ink == inklabels).sum().float().div(inklabels.size(0))
            step_fbeta = fbeta_score(inklabels.view(-1).numpy(), pred_ink.view(-1).numpy(), beta=0.5)

            running_fbeta    += step_fbeta
            running_accuracy += accuracy.item()
            running_loss     += loss.item()
            denom += 1

            now            = time.time()
            elapsed        = now - train_start
            step_duration  = now - step_start
            step_start     = now

            # Buffer the per-step row — no disk I/O yet.
            logger.log_step({
                "condition":            condition,
                "fragments":            fragments,
                "step":                 i,
                "loss":                 round(loss.item(), 6),
                "accuracy":             round(accuracy.item(), 6),
                "fbeta":                round(step_fbeta, 6),
                "elapsed_seconds":      round(elapsed, 3),
                "step_duration_seconds": round(step_duration, 4),
            })

            pbar.set_postfix({
                "Loss":      running_loss     / denom,
                "Accuracy":  running_accuracy / denom,
                "Fbeta@0.5": running_fbeta    / denom,
            })

            # Every 500 steps: reset running stats and flush buffer to disk.
            if (i + 1) % 500 == 0:
                running_loss     = 0.
                running_accuracy = 0.
                running_fbeta    = 0.
                denom            = 0
                logger.flush_steps()

        # ---- end of training loop ----------------------------------------
        total_seconds = time.time() - train_start
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_ram_mb = round(peak_bytes / 1024 / 1024, 2)

        logger.log_summary({
            "condition":              condition,
            "fragments":              fragments,
            "total_steps":            i + 1,
            "total_training_seconds": round(total_seconds, 2),
            "peak_ram_mb":            peak_ram_mb,
            "batch_size":             self.batch_size,
            "learning_rate":          self.learning_rate,
            # FLOPs per step = 3× forward pass (1× forward + 2× backward).
            # Total training FLOPs = flops_per_step × steps completed.
            "flops_per_forward":      self.flops_per_forward,
            "total_training_flops":   self.flops_per_forward * 3 * (i + 1),
            "timestamp":              time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

        print(f"[Logger] Training complete — {round(total_seconds/60, 1)} min, peak RAM {peak_ram_mb} MB")

        torch.save(self.model.state_dict(), os.path.join(self.base_path, "data/model.pt"))
        config.edit("trained", "true")
        config.save()

    def prune_and_finetune(self, train_loader, config: Config, fragments: str):
        """
        Applied only when condition == 'pruned'.

        Step 1 — Magnitude pruning:
            Zero out the 30% of weights with the lowest absolute value in every
            Conv3d layer. This is unstructured pruning, meaning individual weights
            are zeroed rather than entire filters. It doesn't reduce the model's
            parameter count on disk, but it reduces the effective computation and
            is the standard starting point described by Zhu & Gupta.

        Step 2 — Fine-tuning:
            Run an additional pruning_steps training steps so the surviving weights
            can compensate for the removed ones. These steps are logged separately
            under condition="pruned_finetune" so training time can be compared
            cleanly: baseline vs compressed vs (pruned_train + pruned_finetune).
        """
        print("\n[Pruning] Applying 30% magnitude-based unstructured pruning to all Conv3d layers...")

        conv_layers = [
            module for module in self.model.modules()
            if isinstance(module, nn.Conv3d)
        ]
        for layer in conv_layers:
            prune.l1_unstructured(layer, name="weight", amount=0.30)

        # Report how many weights survived
        total_params  = sum(w.numel() for w in self.model.parameters())
        zero_params   = sum((w == 0).sum().item() for w in self.model.parameters())
        sparsity      = 100. * zero_params / total_params
        print(f"[Pruning] Sparsity after pruning: {sparsity:.1f}% of weights zeroed")

        # Fine-tune --------------------------------------------------------
        pruning_steps = int(config.get("pruning_steps"))
        print(f"[Pruning] Fine-tuning for {pruning_steps} steps...")
        logger    = Logger(self.base_path)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate * 0.1)
        # Lower LR for fine-tuning — standard practice after pruning
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.learning_rate * 0.1, total_steps=pruning_steps
        )
        self.model.train()

        running_loss = running_accuracy = running_fbeta = 0.0
        denom = 0
        tracemalloc.start()
        ft_start   = time.time()
        step_start = ft_start

        pbar = tqdm(enumerate(train_loader), total=pruning_steps)
        for i, (subvolumes, inklabels) in pbar:
            if i >= pruning_steps:
                break
            optimizer.zero_grad()
            outputs = self.model(subvolumes.to(self.DEVICE))
            loss    = criterion(outputs, inklabels.to(self.DEVICE))
            loss.backward()
            optimizer.step()
            scheduler.step()

            pred_ink   = outputs.detach().sigmoid().gt(0.4).cpu().int()
            accuracy   = (pred_ink == inklabels).sum().float().div(inklabels.size(0))
            step_fbeta = fbeta_score(
                inklabels.view(-1).numpy(), pred_ink.view(-1).numpy(), beta=0.5
            )
            running_fbeta    += step_fbeta
            running_accuracy += accuracy.item()
            running_loss     += loss.item()
            denom += 1

            now           = time.time()
            elapsed       = now - ft_start
            step_duration = now - step_start
            step_start    = now

            logger.log_step({
                "condition":             "pruned_finetune",
                "fragments":             fragments,
                "step":                  i,
                "loss":                  round(loss.item(), 6),
                "accuracy":              round(accuracy.item(), 6),
                "fbeta":                 round(step_fbeta, 6),
                "elapsed_seconds":       round(elapsed, 3),
                "step_duration_seconds": round(step_duration, 4),
            })

            pbar.set_postfix({
                "Loss":      running_loss     / denom,
                "Accuracy":  running_accuracy / denom,
                "Fbeta@0.5": running_fbeta    / denom,
            })

            if (i + 1) % 500 == 0:
                running_loss = running_accuracy = running_fbeta = 0.
                denom = 0
                logger.flush_steps()

        # Remove pruning re-parametrisation so the model saves cleanly
        for layer in conv_layers:
            prune.remove(layer, "weight")

        total_ft_seconds = time.time() - ft_start
        _, peak_bytes    = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_ram_mb = round(peak_bytes / 1024 / 1024, 2)

        logger.log_summary({
            "condition":              "pruned_finetune",
            "fragments":              fragments,
            "total_steps":            i + 1,
            "total_training_seconds": round(total_ft_seconds, 2),
            "peak_ram_mb":            peak_ram_mb,
            "batch_size":             self.batch_size,
            "learning_rate":          self.learning_rate * 0.1,
            "flops_per_forward":      self.flops_per_forward,
            "total_training_flops":   self.flops_per_forward * 3 * (i + 1),
            "timestamp":              time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

        print(f"[Pruning] Fine-tune complete — {round(total_ft_seconds/60, 1)} min | Sparsity: {sparsity:.1f}%")
        torch.save(self.model.state_dict(), os.path.join(self.base_path, "data/model.pt"))

    def load_model(self, training_data: list, config: Config):
        train_path = os.path.join(self.base_path, "data/vesuvius-challenge-ink-detection/train")

        condition = config.get("condition")
        fragments = ",".join(sorted(training_data))

        print("Train Run: " + str(self.train_run))
        if self.train_run:
            print("All fragments to train with:", sorted(training_data))

            train_fragments = [
                Path(os.path.join(train_path, f)) for f in training_data
            ]
            train_dset = SubvolumeDataset(
                fragments=train_fragments, voxel_shape=(48, 64, 64), filter_edge_pixels=True
            )
            train_loader = thd.DataLoader(train_dset, batch_size=self.batch_size, shuffle=True)

            print("Num batches:", len(train_loader))
            print("Num items (pixels):", len(train_dset))
            print("Loaded dataset for training, generating model...")

            warnings.simplefilter('ignore', UndefinedMetricWarning)

            # All three conditions run the main training loop first.
            self.train_model(train_loader, config, condition, fragments)

            # Pruned condition: apply pruning + fine-tune on top of the trained model.
            if condition == "pruned":
                # Re-shuffle the loader so fine-tuning sees a fresh order.
                ft_loader = thd.DataLoader(train_dset, batch_size=self.batch_size, shuffle=True)
                self.prune_and_finetune(ft_loader, config, fragments)
                del ft_loader

            train_dset.labels       = []
            train_dset.image_stacks = []
            del train_loader, train_dset
            gc.collect()

        else:
            print("Loading model configurations into memory...")
            model_weights = torch.load(
                os.path.join(self.base_path, "data/model.pt"),
                map_location=self.DEVICE,
            )
            self.model.load_state_dict(model_weights)

        print("Model successfully loaded.")

    def eval_model(self, threshold):
        grayscale_mode = threshold is None
        config = Config()
        condition = config.get("condition")
        fragments = config.get("training_data")
        logger    = Logger(self.base_path)

        test_path = Path(os.path.join(self.base_path, "data/vesuvius-challenge-ink-detection/test"))
        test_fragments = [f for f in test_path.iterdir() if f.is_dir()]
        print("All fragments to run: ", test_fragments)
        pred_images = []
        self.model.eval()
        for test_fragment in test_fragments:
            outputs = []
            eval_dset = SubvolumeDataset(fragments=[test_fragment], voxel_shape=(48, 64, 64), load_inklabels=False)
            eval_loader = thd.DataLoader(eval_dset, batch_size=self.batch_size, shuffle=False)
            with torch.no_grad():
                for i, (subvolumes, _) in enumerate(tqdm(eval_loader)):
                    output = self.model(subvolumes.to(self.DEVICE)).view(-1).sigmoid().cpu().numpy()
                    outputs.append(output)

            image_shape = eval_dset.image_stacks[0].shape[1:]
            eval_dset.labels = []
            eval_dset.image_stacks = []
            del eval_loader
            gc.collect()

            outputs     = np.concatenate(outputs)
            pred_image  = np.zeros(image_shape, dtype=np.uint8)
            pred_binary = []
            pixels_evaluated = 0
            for (y, x, _), prob in zip(eval_dset.pixels[:outputs.shape[0]], outputs):
                if grayscale_mode:
                    pixel_pred = int(round(prob * 255))
                else:
                    pixel_pred = int(prob > threshold)
                pred_image[y, x] = pixel_pred
                pred_binary.append(pixel_pred)
                pixels_evaluated += 1
            pred_images.append(pred_image)

            ink_fraction = round(float(np.mean(pred_binary)), 6) if pred_binary else 0.0
            logger.log_eval({
                "condition":       condition,
                "fragments":       fragments,
                "fragment_name":   test_fragment.name,
                "threshold":       threshold,
                "fbeta":           ink_fraction,
                # Eval is inference-only: 1× forward FLOPs per pixel, no backward pass.
                "flops_per_pixel": self.flops_per_forward,
                "total_eval_flops": self.flops_per_forward * pixels_evaluated,
                "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
            })

            eval_dset.pixels = np.empty((0, 3), dtype=np.int32)
            del eval_dset
            gc.collect()
            print("Finished this segment-> ", test_fragment)

        util.clear()
        print("Finished! saving ink images...")

        for i, pred_image in enumerate(pred_images):
            plt.imshow(pred_image, cmap='gray')
            plt.axis('off')
            plt.gca().set_position([0, 0, 1, 1]) #type: ignore
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)

            file_name  = f"image_{threshold}_{i}.png"
            output_dir = os.path.join(config.get("base_path"), "outputs")
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

            plt.savefig(os.path.join(output_dir, file_name), format='png', bbox_inches='tight', pad_inches=0)
            print(f"Saved {file_name}")

def download_data():    
    try:
        print("Starting Download...")
        od.download("https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/data", "data/")
        print("Download Complete!")
        
    except KeyboardInterrupt:
        print("ERROR: exiting prematurly. Data installation incomplete.")
        print("deleting corrupt data...")
        if os.path.isdir("data"):
            shutil.rmtree("data")
        exit(1)
