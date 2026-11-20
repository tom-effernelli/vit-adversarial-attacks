"""Compare random noise, FGSM, and PGD on a CIFAR-10 Vision Transformer.

Usage: python experiment.py --samples 16 --steps 10
The first run downloads CIFAR-10 and the model weights.
"""

import argparse
import csv
import io
import json
import time
from pathlib import Path
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

from attacks import fgsm, pgd, random_noise


MODEL_ID = "nateraw/vit-base-patch16-224-cifar10"


class QuickCIFAR10(Dataset):
    """A small official CIFAR-10 test sample, served by Hugging Face Datasets."""

    API = "https://datasets-server.huggingface.co/rows?dataset=uoft-cs/cifar10&config=plain_text&split=test"

    def __init__(self, root, transform, count):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.transform = transform
        # Images and labels are cached locally, so later runs need no network.
        self.index_path = self.root / "labels.json"
        cache = json.loads(self.index_path.read_text()) if self.index_path.exists() else {}
        self.classes = cache.get("classes")
        labels = {int(k): v for k, v in cache.get("labels", {}).items()}
        have = [i for i in range(count) if i in labels and self._path(i).exists()]
        if len(have) < count:
            self._download(count, labels)
        self.items = [(i, labels[i]) for i in range(count) if i in labels and self._path(i).exists()]

    def _path(self, row_idx):
        return self.root / f"test_{row_idx:05d}.jpg"

    @staticmethod
    def _fetch(url, attempts=4):
        for attempt in range(attempts):
            try:
                with urlopen(url, timeout=30) as response:
                    return response.read()
            except Exception:  # the public server sometimes answers 5xx; retry
                if attempt == attempts - 1:
                    raise
                time.sleep(2 * (attempt + 1))

    def _download(self, count, labels):
        # The server returns at most 100 rows per request; image URLs are signed and
        # expire, so each image is downloaded right away rather than in __getitem__.
        for offset in range(0, count, 100):
            length = min(100, count - offset)
            payload = json.loads(self._fetch(f"{self.API}&offset={offset}&length={length}"))
            self.classes = payload["features"][1]["type"]["names"]
            for entry in payload["rows"]:
                row_idx = entry["row_idx"]
                if not self._path(row_idx).exists():
                    self._path(row_idx).write_bytes(self._fetch(entry["row"]["img"]["src"]))
                labels[row_idx] = int(entry["row"]["label"])
        self.index_path.write_text(json.dumps({"classes": self.classes, "labels": labels}))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row_idx, label = self.items[index]
        with Image.open(self._path(row_idx)) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, label


def normalization(processor, device):
    mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)
    return lambda pixels: (pixels - mean) / std


@torch.no_grad()
def observe(model, pixels, labels, normalize):
    """Return predictions, per-image loss, and the final block's CLS vector."""
    captured = []

    def capture(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[:, 0, :].detach().clone())

    # Transformers 5 exposes `vit.layers`; Transformers 4 used `vit.encoder.layer`.
    blocks = model.vit.layers if hasattr(model.vit, "layers") else model.vit.encoder.layer
    handle = blocks[-1].register_forward_hook(capture)
    try:
        logits = model(pixel_values=normalize(pixels)).logits
    finally:
        handle.remove()
    losses = F.cross_entropy(logits, labels, reduction="none")
    return logits.argmax(dim=1), losses, captured[0]


def relative_drift(clean_representation, attacked_representation):
    numerator = torch.linalg.vector_norm(attacked_representation - clean_representation, dim=1)
    denominator = torch.linalg.vector_norm(clean_representation, dim=1).clamp_min(1e-12)
    return (numerator / denominator).cpu().tolist()


def rank(values):
    """Ranks 0..n-1 (ties broken by order), used for a simple Spearman correlation."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    return ranks


def drift_vs_error(drifts, losses, fooled):
    """Does a larger internal change go with more errors?

    Pools every attacked image (random, FGSM and PGD together) and reports:
    - the mean CLS drift of images whose prediction changed vs. stayed correct;
    - the Spearman rank correlation between drift and loss.
    Descriptive only: a correlation here does not show that the drift causes the error.
    """
    drift = np.concatenate([drifts[name] for name in drifts])
    loss = np.concatenate([losses[name] for name in drifts])
    wrong = np.concatenate([fooled[name] for name in drifts]).astype(bool)
    spearman = float(np.corrcoef(rank(drift), rank(loss))[0, 1])
    return {
        "n_attacked_images": int(len(drift)),
        "mean_drift_when_fooled": float(drift[wrong].mean()) if wrong.any() else None,
        "mean_drift_when_still_correct": float(drift[~wrong].mean()) if (~wrong).any() else None,
        "spearman_drift_vs_loss": spearman,
    }


def save_drift_plot(path, drifts, losses, fooled):
    """Scatter of internal drift vs. loss, one point per attacked image."""
    markers = {"random": "o", "fgsm": "s", "pgd": "^"}
    figure, ax = plt.subplots(figsize=(7, 5))
    for name in drifts:
        drift = np.asarray(drifts[name])
        loss = np.asarray(losses[name])
        wrong = np.asarray(fooled[name]).astype(bool)
        ax.scatter(drift[~wrong], loss[~wrong], marker=markers[name], facecolors="none",
                   edgecolors="tab:blue", label=f"{name}: still correct")
        ax.scatter(drift[wrong], loss[wrong], marker=markers[name], color="tab:red",
                   label=f"{name}: fooled")
    ax.set_xlabel("Relative drift of the final [CLS] vector")
    ax.set_ylabel("Cross-entropy loss on the true class")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_title("Bigger internal change vs. classification error")
    ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_grid(path, images, labels, predictions, class_names, epsilon, rows=6):
    """One row per image: clean, random, FGSM, PGD, and the PGD perturbation magnified.

    The perturbation is at most ε per pixel, invisible at normal scale; the last column
    maps [-ε, +ε] to [0, 1] (grey = unchanged pixel) so it can be seen.
    """
    count = min(rows, images["clean"].shape[0])
    names = ["clean", "random", "fgsm", "pgd"]
    figure, axes = plt.subplots(count, len(names) + 1, figsize=(12.5, 2.6 * count), squeeze=False)
    for row in range(count):
        truth = int(labels[row])
        for col, name in enumerate(names):
            ax = axes[row, col]
            image = images[name][row].detach().cpu().permute(1, 2, 0).numpy()
            ax.imshow(np.clip(image, 0, 1))
            pred = int(predictions[name][row])
            mark = "✓" if pred == truth else "✗"
            ax.set_title(f"{name}: {class_names[pred]} {mark}", fontsize=10,
                         color="black" if pred == truth else "tab:red")
            ax.axis("off")
        delta = (images["pgd"][row] - images["clean"][row]).detach().cpu().permute(1, 2, 0).numpy()
        axes[row, -1].imshow(np.clip(delta / (2 * epsilon) + 0.5, 0, 1))
        axes[row, -1].set_title("PGD perturbation (magnified)", fontsize=10)
        axes[row, -1].axis("off")
    figure.suptitle(f"Same budget for every column: each pixel changes by at most ε = {epsilon * 255:.0f}/255")
    figure.tight_layout(rect=(0, 0, 1, 0.97))  # leave room so the title does not cover row 1
    figure.savefig(path, dpi=90)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=16, help="Initially correct test images to attack")
    parser.add_argument("--epsilon", type=float, default=8 / 255, help="Maximum per-pixel change in [0,1]")
    parser.add_argument("--step-size", type=float, default=2 / 255, help="PGD change per iteration")
    parser.add_argument("--steps", type=int, default=10, help="PGD iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--source", choices=["quick", "full"], default="quick",
                        help="quick: streamed sample; full: torchvision's complete CIFAR-10 test set")
    args = parser.parse_args()
    if args.samples < 1 or args.steps < 1 or not 0 < args.epsilon <= 1 or args.step_size <= 0:
        parser.error("samples and steps must be positive; epsilon must be in (0,1]; step-size must be positive")

    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} on {device}", flush=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    normalize = normalization(processor, device)

    # Attacks operate on the resized, raw [0,1] pixels. Normalization stays
    # inside the differentiable model path, so gradients refer to pixel values.
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    if args.source == "quick":
        dataset = QuickCIFAR10(args.data_dir, transform, min(100, max(32, args.samples * 2)))
    else:
        dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    chosen_images, chosen_labels, chosen_indices = [], [], []
    offset = 0
    for batch_images, batch_labels in loader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        clean_pred, _, _ = observe(model, batch_images, batch_labels, normalize)
        for index in torch.where(clean_pred == batch_labels)[0].tolist():
            chosen_images.append(batch_images[index:index + 1])
            chosen_labels.append(batch_labels[index:index + 1])
            chosen_indices.append(offset + index)
            if len(chosen_images) == args.samples:
                break
        if len(chosen_images) == args.samples:
            break
        offset += len(batch_images)
    if not chosen_images:
        raise RuntimeError("The model classified no test examples correctly; check model and label mapping.")
    if len(chosen_images) < args.samples:
        print(f"Only {len(chosen_images)} of {args.samples} requested correctly classified images were available.")
    pixels = torch.cat(chosen_images)
    labels = torch.cat(chosen_labels)
    print(f"Attacking {len(pixels)} initially correct test images", flush=True)

    variants = {"clean": pixels}
    variants["random"] = random_noise(pixels, args.epsilon)
    variants["fgsm"] = fgsm(model, pixels, labels, normalize, args.epsilon)
    # Attack one image at a time to keep CPU memory use predictable.
    variants["pgd"] = torch.cat([
        pgd(model, pixels[i:i + 1], labels[i:i + 1], normalize,
            args.epsilon, args.step_size, args.steps)
        for i in range(len(pixels))
    ])

    predictions, losses, representations = {}, {}, {}
    for name, batch in variants.items():
        predictions[name], losses[name], representations[name] = observe(
            model, batch, labels, normalize
        )
    drifts = {
        name: relative_drift(representations["clean"], representations[name])
        for name in ("random", "fgsm", "pgd")
    }
    rows = []
    for i, index in enumerate(chosen_indices):
        row = {"test_index": index, "true_class": dataset.classes[int(labels[i])]}
        for name in variants:
            row[f"{name}_prediction"] = dataset.classes[int(predictions[name][i])]
            row[f"{name}_loss"] = round(float(losses[name][i]), 6)
            if name != "clean":
                row[f"{name}_drift"] = round(float(drifts[name][i]), 6)
                row[f"{name}_linf"] = round(
                    float((variants[name][i] - pixels[i]).abs().max()), 6
                )
        rows.append(row)

    with (args.out_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": MODEL_ID,
        "dataset": f"CIFAR-10 test set ({args.source} source); only initially correct examples retained",
        "pixel_space": "224x224 resized RGB, each channel in [0,1] before normalization",
        "n": len(rows), "epsilon": args.epsilon, "pgd_step_size": args.step_size,
        "pgd_steps": args.steps, "seed": args.seed,
        "results": {
            name: {
                "attack_success_rate": float((predictions[name] != labels).float().mean()),
                "mean_loss": float(losses[name].mean()),
                "mean_relative_cls_drift": float(np.mean(drifts[name])),
            }
            for name in ("random", "fgsm", "pgd")
        },
    }
    attacked = ("random", "fgsm", "pgd")
    attacked_losses = {name: losses[name].cpu().numpy() for name in attacked}
    fooled = {name: (predictions[name] != labels).cpu().numpy() for name in attacked}
    summary["drift_vs_error"] = drift_vs_error(drifts, attacked_losses, fooled)
    save_drift_plot(args.out_dir / "drift_vs_loss.png", drifts, attacked_losses, fooled)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_grid(args.out_dir / "examples.png", variants, labels, predictions,
              dataset.classes, args.epsilon)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved results in {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

