#!/usr/bin/env python3
"""Organize images into richer content buckets based on model predictions."""

from __future__ import annotations

import argparse
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

CATEGORY_KEYWORDS: dict[str, set[str]] = {
    "people": {
        "person",
        "man",
        "woman",
        "boy",
        "girl",
        "bride",
        "groom",
        "scuba diver",
        "baseball player",
        "swimmer",
        "skier",
        "soldier",
        "academic gown",
    },
    "scenery": {
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
        "desert",
        "ocean",
        "rainforest",
        "waterfall",
    },
    "animals": {
        "goldfish",
        "tabby",
        "persian cat",
        "siamese cat",
        "egyptian cat",
        "lion",
        "tiger",
        "cheetah",
        "bear",
        "zebra",
        "hippopotamus",
        "ox",
        "ram",
        "llama",
        "camel",
        "elephant",
        "giant panda",
        "koala",
        "otter",
        "chimpanzee",
        "gorilla",
        "dog",
        "retriever",
        "shepherd",
        "poodle",
        "wolf",
        "fox",
        "hare",
        "rabbit",
        "deer",
        "squirrel",
        "bird",
        "eagle",
        "parrot",
        "owl",
        "duck",
        "penguin",
        "flamingo",
        "shark",
        "ray",
        "snake",
        "lizard",
        "turtle",
        "insect",
        "butterfly",
        "bee",
        "spider",
    },
    "vehicles": {
        "car wheel",
        "sports car",
        "convertible",
        "jeep",
        "limousine",
        "cab",
        "minivan",
        "truck",
        "pickup",
        "trailer truck",
        "fire engine",
        "ambulance",
        "bus",
        "school bus",
        "trolleybus",
        "motor scooter",
        "moped",
        "mountain bike",
        "bicycle",
        "airliner",
        "warplane",
        "airship",
        "space shuttle",
        "bullet train",
        "steam locomotive",
        "submarine",
        "boat",
        "catamaran",
        "sailboat",
        "speedboat",
        "tractor",
        "forklift",
        "locomotive",
    },
    "food": {
        "pizza",
        "cheeseburger",
        "hotdog",
        "french loaf",
        "bagel",
        "pretzel",
        "ice cream",
        "trifle",
        "potpie",
        "carbonara",
        "guacamole",
        "consomme",
        "red wine",
        "espresso",
        "banana",
        "pineapple",
        "orange",
        "lemon",
        "pomegranate",
        "strawberry",
        "fig",
        "granny smith",
        "mushroom",
        "broccoli",
        "cauliflower",
        "artichoke",
        "soup bowl",
        "plate",
        "dining table",
    },
    "buildings": {
        "church",
        "mosque",
        "palace",
        "monastery",
        "dome",
        "library",
        "planetarium",
        "greenhouse",
        "movie theater",
        "restaurant",
        "lighthouse",
        "castle",
        "barn",
        "boathouse",
        "schoolhouse",
        "bell cote",
        "bridge",
        "suspension bridge",
        "viaduct",
        "pier",
        "fountain",
        "gazebo",
        "patio",
        "skyscraper",
        "tower",
        "arch",
        "building",
        "apartment",
        "house",
        "office building",
        "stadium",
        "airport terminal",
    },
    "objects": {
        "laptop",
        "computer",
        "keyboard",
        "mouse",
        "phone",
        "camera",
        "book",
        "chair",
        "table",
        "lamp",
        "backpack",
        "watch",
        "bottle",
        "cup",
        "vase",
        "sofa",
        "television",
        "monitor",
        "toaster",
        "microwave",
    },
}

TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    category: str
    category_score: float


class ImageOrganizer:
    def __init__(self, topk: int = 8) -> None:
        try:
            from torchvision import models
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency 'torchvision'. Please run: pip install -r requirements.txt"
            ) from exc

        self.topk = topk
        self.models: list[tuple[object, object, str, float]] = []

        # 融合轻量模型 + 高精度模型，降低单模型误判（尤其是人像/建筑/物品）
        mobilenet_weights = models.MobileNet_V3_Small_Weights.DEFAULT
        mobilenet_model = models.mobilenet_v3_small(weights=mobilenet_weights)
        mobilenet_model.eval()
        self.models.append((mobilenet_model, mobilenet_weights, "mobilenet_v3_small", 0.45))

        resnet_weights = models.ResNet50_Weights.DEFAULT
        resnet_model = models.resnet50(weights=resnet_weights)
        resnet_model.eval()
        self.models.append((resnet_model, resnet_weights, "resnet50", 0.55))

    def classify(self, image_path: Path) -> ClassificationResult:
        try:
            from PIL import Image
            import torch
        except ImportError as exc:
            raise SystemExit(
                "Missing dependencies. Please run: pip install -r requirements.txt"
            ) from exc

        image = Image.open(image_path).convert("RGB")
        merged_scores: dict[str, float] = {}
        label_confidence: dict[str, float] = {}

        for model, weights, _name, model_weight in self.models:
            preprocess = weights.transforms()
            labels = weights.meta["categories"]
            tensor = preprocess(image).unsqueeze(0)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)

            conf, idx = torch.topk(probs, self.topk)
            for label_idx, confidence in zip(idx[0].tolist(), conf[0].tolist()):
                label = labels[label_idx]
                weighted = float(confidence) * model_weight
                merged_scores[label] = merged_scores.get(label, 0.0) + weighted
                label_confidence[label] = max(label_confidence.get(label, 0.0), float(confidence))

        sorted_labels = sorted(merged_scores.items(), key=lambda item: item[1], reverse=True)
        top_labels = [label for label, _ in sorted_labels[: self.topk]]
        top_confidences = [label_confidence[label] for label in top_labels]

        if not top_labels:
            return ClassificationResult(label="object", confidence=0.0, category="objects", category_score=0.0)

        top_label = top_labels[0]
        top_conf = top_confidences[0]
        category, category_score = self._map_category(top_labels, top_confidences)

        return ClassificationResult(
            label=top_label,
            confidence=top_conf,
            category=category,
            category_score=category_score,
        )

    def _map_category(self, labels: Iterable[str], confidences: Iterable[float]) -> tuple[str, float]:
        scores = {category: 0.0 for category in CATEGORY_KEYWORDS}

        for label, confidence in zip(labels, confidences):
            normalized_label = self._normalize_text(label)
            label_tokens = self._tokens(normalized_label)
            if not normalized_label:
                continue

            for category, keywords in CATEGORY_KEYWORDS.items():
                for keyword in keywords:
                    normalized_keyword = self._normalize_text(keyword)
                    if not normalized_keyword:
                        continue

                    keyword_tokens = self._tokens(normalized_keyword)

                    if normalized_label == normalized_keyword:
                        scores[category] += confidence * 1.3
                        continue

                    if normalized_keyword in normalized_label or normalized_label in normalized_keyword:
                        scores[category] += confidence
                        continue

                    overlap = len(label_tokens & keyword_tokens)
                    if overlap:
                        token_ratio = overlap / max(len(keyword_tokens), len(label_tokens))
                        scores[category] += confidence * token_ratio * 0.75

        best_category = max(scores, key=scores.get, default="objects")
        best_score = scores.get(best_category, 0.0)

        if best_score <= 0:
            return "objects", 0.0

        return best_category, best_score

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(TOKEN_SPLIT_RE.sub(" ", text.lower()).split())

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(text.split())


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
    min_category_score: float,
    allow_unknown: bool,
) -> None:
    organizer = ImageOrganizer()
    action = shutil.move if move else shutil.copy2

    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for image_path in iter_images(source_dir):
        total += 1
        result = organizer.classify(image_path)
        if result.confidence >= min_confidence and result.category_score >= min_category_score:
            category = result.category
        elif allow_unknown:
            category = "unknown"
        else:
            category = result.category or "objects"

        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        destination = unique_path(category_dir / image_path.name)
        action(str(image_path), str(destination))

        print(
            f"[{category.upper()}] {image_path} -> {destination} "
            f"(label={result.label}, conf={result.confidence:.2%}, category_score={result.category_score:.3f})"
        )

    print(f"\nDone. Processed {total} image(s).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify and organize images into people/scenery/animals/"
            "vehicles/food/buildings/objects folders."
        )
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
        default=0.16,
        help="Minimum model confidence for strict unknown routing (default: 0.16)",
    )
    parser.add_argument(
        "--min-category-score",
        type=float,
        default=0.08,
        help="Minimum category match score for strict unknown routing (default: 0.08)",
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help="Allow low-confidence samples to be put into unknown (disabled by default)",
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
    root.title("Image Organizer Pro")
    root.geometry("980x620")
    root.minsize(920, 580)
    root.configure(bg="#0f172a")

    source_var = tk.StringVar()
    output_var = tk.StringVar(value=str(Path("organized_images").resolve()))
    move_var = tk.BooleanVar(value=False)
    confidence_var = tk.StringVar(value="0.16")
    category_score_var = tk.StringVar(value="0.08")
    allow_unknown_var = tk.BooleanVar(value=False)
    status_var = tk.StringVar(value="Ready")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Main.TFrame", background="#0f172a")
    style.configure("Card.TFrame", background="#111827", relief="flat")
    style.configure("Section.TFrame", background="#0b1220")
    style.configure("Main.TLabel", background="#111827", foreground="#e5e7eb", font=("Segoe UI", 11))
    style.configure("Title.TLabel", background="#0f172a", foreground="#f8fafc", font=("Segoe UI", 24, "bold"))
    style.configure("Hint.TLabel", background="#0f172a", foreground="#93c5fd", font=("Segoe UI", 11))
    style.configure("SectionTitle.TLabel", background="#0b1220", foreground="#bfdbfe", font=("Segoe UI", 10, "bold"))
    style.configure("Subtle.TLabel", background="#111827", foreground="#9ca3af", font=("Segoe UI", 10))
    style.configure("Main.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 9), borderwidth=0)
    style.map("Main.TButton", background=[("active", "#2563eb"), ("!disabled", "#1d4ed8")], foreground=[("!disabled", "#ffffff")])
    style.configure("Secondary.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8), borderwidth=0)
    style.map("Secondary.TButton", background=[("active", "#334155"), ("!disabled", "#1e293b")], foreground=[("!disabled", "#e2e8f0")])
    style.configure("Main.TCheckbutton", background="#111827", foreground="#e5e7eb", font=("Segoe UI", 10))
    style.map("Main.TCheckbutton", background=[("active", "#111827")], foreground=[("active", "#f8fafc")])
    style.configure("Main.TEntry", fieldbackground="#1f2937", foreground="#f9fafb", insertcolor="#f9fafb")
    style.configure("Status.TLabel", background="#111827", foreground="#34d399", font=("Segoe UI", 10, "bold"))

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

        try:
            min_category_score = float(category_score_var.get())
        except ValueError:
            messagebox.showerror("Input error", "Category score must be a number.")
            return

        if not (0.0 <= min_category_score <= 1.0):
            messagebox.showerror("Input error", "Category score must be between 0 and 1.")
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
                    min_category_score=min_category_score,
                    allow_unknown=allow_unknown_var.get(),
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

    outer = ttk.Frame(root, style="Main.TFrame", padding=22)
    outer.pack(fill="both", expand=True)

    ttk.Label(outer, text="Image Organizer Pro", style="Title.TLabel").pack(anchor="w")
    ttk.Label(
        outer,
        text="v5.0 双模型融合：强化人物/建筑/物品识别，默认不落 unknown",
        style="Hint.TLabel",
    ).pack(anchor="w", pady=(8, 18))

    top_meta = ttk.Frame(outer, style="Main.TFrame")
    top_meta.pack(fill="x", pady=(0, 14))
    ttk.Label(
        top_meta,
        text="支持格式: JPG · PNG · BMP · WEBP · TIFF",
        style="Hint.TLabel",
    ).pack(side="left")
    ttk.Label(
        top_meta,
        text="建议：先用复制模式试跑，再决定是否移动原图",
        style="Hint.TLabel",
    ).pack(side="right")

    frm = ttk.Frame(outer, style="Card.TFrame", padding=20)
    frm.pack(fill="both", expand=True)

    input_section = ttk.Frame(frm, style="Section.TFrame", padding=14)
    input_section.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 12))
    ttk.Label(input_section, text="路径设置", style="SectionTitle.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

    ttk.Label(input_section, text="Source folder", style="Main.TLabel").grid(row=1, column=0, sticky="w", pady=8)
    ttk.Entry(input_section, textvariable=source_var, width=82, style="Main.TEntry").grid(
        row=1, column=1, padx=10, sticky="ew"
    )
    ttk.Button(input_section, text="Browse", command=choose_source, style="Secondary.TButton").grid(row=1, column=2)

    ttk.Label(input_section, text="Output folder", style="Main.TLabel").grid(row=2, column=0, sticky="w", pady=8)
    ttk.Entry(input_section, textvariable=output_var, width=82, style="Main.TEntry").grid(
        row=2, column=1, padx=10, sticky="ew"
    )
    ttk.Button(input_section, text="Browse", command=choose_output, style="Secondary.TButton").grid(row=2, column=2)

    ttk.Label(input_section, text="不会覆盖原文件，重名将自动追加序号", style="Subtle.TLabel").grid(
        row=3, column=1, sticky="w", padx=10, pady=(2, 0)
    )

    threshold_section = ttk.Frame(frm, style="Section.TFrame", padding=14)
    threshold_section.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 12))
    ttk.Label(threshold_section, text="阈值设置", style="SectionTitle.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

    ttk.Label(threshold_section, text="Min confidence (0~1)", style="Main.TLabel").grid(row=1, column=0, sticky="w", pady=8)
    ttk.Entry(threshold_section, textvariable=confidence_var, width=18, style="Main.TEntry").grid(
        row=1, column=1, sticky="w", padx=10
    )

    ttk.Label(threshold_section, text="Min category score (0~1)", style="Main.TLabel").grid(row=2, column=0, sticky="w", pady=8)
    ttk.Entry(threshold_section, textvariable=category_score_var, width=18, style="Main.TEntry").grid(
        row=2, column=1, sticky="w", padx=10
    )

    ttk.Label(
        threshold_section,
        text="建议默认值：0.16 / 0.08（平衡准确率与召回率）",
        style="Subtle.TLabel",
    ).grid(row=3, column=1, sticky="w", padx=10, pady=(2, 0))

    option_section = ttk.Frame(frm, style="Section.TFrame", padding=14)
    option_section.grid(row=2, column=0, columnspan=3, sticky="ew")
    ttk.Label(option_section, text="运行选项", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))

    ttk.Checkbutton(
        option_section,
        text="Move files (instead of copy)",
        variable=move_var,
        style="Main.TCheckbutton",
    ).grid(row=1, column=0, sticky="w", padx=10, pady=6)

    ttk.Checkbutton(
        option_section,
        text="Allow unknown folder (strict mode)",
        variable=allow_unknown_var,
        style="Main.TCheckbutton",
    ).grid(row=2, column=0, sticky="w", padx=10, pady=2)

    action_row = ttk.Frame(frm, style="Card.TFrame")
    action_row.grid(row=3, column=0, columnspan=3, sticky="ew", padx=10, pady=(18, 0))

    start_btn = ttk.Button(action_row, text="Start organizing", command=run_task, style="Main.TButton")
    start_btn.pack(side="left")
    ttk.Label(action_row, textvariable=status_var, style="Status.TLabel").pack(side="right")

    frm.columnconfigure(0, weight=1)
    input_section.columnconfigure(1, weight=1)
    threshold_section.columnconfigure(1, weight=1)
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
    if not (0.0 <= args.min_category_score <= 1.0):
        raise SystemExit("--min-category-score must be between 0 and 1")

    organize(
        source_dir=args.source,
        output_dir=args.output,
        move=args.move,
        min_confidence=args.min_confidence,
        min_category_score=args.min_category_score,
        allow_unknown=args.allow_unknown,
    )


if __name__ == "__main__":
    main()
