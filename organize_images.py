#!/usr/bin/env python3
"""Organize images into people/scenery/objects buckets based on content."""

from __future__ import annotations

import argparse
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

PEOPLE_KEYWORDS = {
    "person",
    "man",
    "woman",
    "boy",
    "girl",
    "bride",
    "groom",
    "scuba diver",
    "baseball player",
}

SCENERY_KEYWORDS = {
    "mountain",
    "valley",
    "volcano",
    "cliff",
    "seashore",
    "lakeside",
    "sandbar",
    "promontory",
    "geyser",
    "alp",
    "coral reef",
    "beach",
    "forest",
}


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    category: str


class ImageOrganizer:
    def __init__(self, topk: int = 5) -> None:
        try:
            from torchvision import models
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency 'torchvision'. Please run: pip install -r requirements.txt"
            ) from exc

        self.topk = topk
        self.weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=self.weights)
        self.model.eval()
        self.preprocess = self.weights.transforms()
        self.labels = self.weights.meta["categories"]

    def classify(self, image_path: Path) -> ClassificationResult:
        try:
            from PIL import Image
            import torch
        except ImportError as exc:
            raise SystemExit(
                "Missing dependencies. Please run: pip install -r requirements.txt"
            ) from exc

        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)

        conf, idx = torch.topk(probs, self.topk)
        labels = [self.labels[i] for i in idx[0].tolist()]
        confidences = conf[0].tolist()

        top_label = labels[0]
        top_conf = float(confidences[0])
        category = self._map_category(labels)

        return ClassificationResult(label=top_label, confidence=top_conf, category=category)

    def _map_category(self, labels: Iterable[str]) -> str:
        normalized = {label.lower().strip() for label in labels}

        if normalized & PEOPLE_KEYWORDS:
            return "people"
        if normalized & SCENERY_KEYWORDS:
            return "scenery"

        # If neither matches, default to objects.
        return "objects"


def iter_images(source_dir: Path) -> Iterable[Path]:
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def unique_path(dest_path: Path) -> Path:
    if not dest_path.exists():
        return dest_path

    stem, suffix = dest_path.stem, dest_path.suffix
    counter = 1
    while True:
        candidate = dest_path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def organize(
    source_dir: Path,
    output_dir: Path,
    move: bool,
    min_confidence: float,
) -> None:
    organizer = ImageOrganizer()
    action = shutil.move if move else shutil.copy2

    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for image_path in iter_images(source_dir):
        total += 1
        result = organizer.classify(image_path)
        category = result.category if result.confidence >= min_confidence else "unknown"

        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        destination = unique_path(category_dir / image_path.name)
        action(str(image_path), str(destination))

        print(
            f"[{category.upper()}] {image_path} -> {destination} "
            f"(label={result.label}, conf={result.confidence:.2%})"
        )

    print(f"\nDone. Processed {total} image(s).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify and organize images into people/scenery/objects folders."
    )
    parser.add_argument("source", type=Path, nargs="?", help="Source directory containing images")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("organized_images"),
        help="Output directory for organized images (default: organized_images)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.20,
        help="Minimum confidence for category assignment, else goes to unknown (default: 0.20)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch desktop GUI",
    )
    return parser.parse_args()


def launch_gui() -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        raise SystemExit("Tkinter is not available in this Python environment.") from exc

    root = tk.Tk()
    root.title("Image Organizer v2.1 fixed")
    root.geometry("720x300")

    source_var = tk.StringVar()
    output_var = tk.StringVar(value=str(Path("organized_images").resolve()))
    move_var = tk.BooleanVar(value=False)
    confidence_var = tk.StringVar(value="0.20")
    status_var = tk.StringVar(value="Ready")

    def choose_source() -> None:
        selected = filedialog.askdirectory(title="Select source directory")
        if selected:
            source_var.set(selected)

    def choose_output() -> None:
        selected = filedialog.askdirectory(title="Select output directory")
        if selected:
            output_var.set(selected)

    def run_task() -> None:
        source_text = source_var.get().strip()
        output_text = output_var.get().strip()

        if not source_text:
            messagebox.showerror("Input error", "Please choose a source directory.")
            return

        source_path = Path(source_text)
        output_path = Path(output_text or "organized_images")

        if not source_path.exists() or not source_path.is_dir():
            messagebox.showerror("Input error", "Source directory does not exist.")
            return

        try:
            min_confidence = float(confidence_var.get())
        except ValueError:
            messagebox.showerror("Input error", "Min confidence must be a number.")
            return

        if not (0.0 <= min_confidence <= 1.0):
            messagebox.showerror("Input error", "Min confidence must be between 0 and 1.")
            return

        start_btn.configure(state="disabled")
        status_var.set("Processing...")

        def worker() -> None:
            try:
                organize(
                    source_dir=source_path,
                    output_dir=output_path,
                    move=move_var.get(),
                    min_confidence=min_confidence,
                )
            except Exception as exc:  # runtime errors should still be shown in GUI
                root.after(
                    0,
                    lambda: (
                        start_btn.configure(state="normal"),
                        status_var.set("Failed"),
                        messagebox.showerror("Run failed", str(exc)),
                    ),
                )
                return

            root.after(
                0,
                lambda: (
                    start_btn.configure(state="normal"),
                    status_var.set("Done"),
                    messagebox.showinfo("Completed", "Image organization completed."),
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    frm = ttk.Frame(root, padding=16)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="Source").grid(row=0, column=0, sticky="w", pady=6)
    ttk.Entry(frm, textvariable=source_var, width=70).grid(row=0, column=1, padx=8)
    ttk.Button(frm, text="Browse", command=choose_source).grid(row=0, column=2)

    ttk.Label(frm, text="Output").grid(row=1, column=0, sticky="w", pady=6)
    ttk.Entry(frm, textvariable=output_var, width=70).grid(row=1, column=1, padx=8)
    ttk.Button(frm, text="Browse", command=choose_output).grid(row=1, column=2)

    ttk.Label(frm, text="Min confidence (0~1)").grid(row=2, column=0, sticky="w", pady=6)
    ttk.Entry(frm, textvariable=confidence_var, width=20).grid(row=2, column=1, sticky="w", padx=8)

    ttk.Checkbutton(frm, text="Move files (instead of copy)", variable=move_var).grid(
        row=3, column=1, sticky="w", padx=8, pady=6
    )

    start_btn = ttk.Button(frm, text="Start", command=run_task)
    start_btn.grid(row=4, column=1, sticky="w", padx=8, pady=12)
    ttk.Label(frm, textvariable=status_var).grid(row=4, column=1, sticky="e", padx=8)

    frm.columnconfigure(1, weight=1)
    root.mainloop()


def main() -> None:
    args = parse_args()

    if args.gui:
        launch_gui()
        return

    if args.source is None:
        raise SystemExit("Source directory is required when not using --gui")

    if not args.source.exists() or not args.source.is_dir():
        raise SystemExit(f"Source directory does not exist: {args.source}")
    if not (0.0 <= args.min_confidence <= 1.0):
        raise SystemExit("--min-confidence must be between 0 and 1")

    organize(
        source_dir=args.source,
        output_dir=args.output,
        move=args.move,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
