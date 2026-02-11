#!/usr/bin/env python3
"""Organize images into richer content buckets based on model predictions."""

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

ANIMAL_KEYWORDS = {
    "goldfish",
    "tabby",
    "persian cat",
    "siamese cat",
    "egyptian cat",
    "lion",
    "tiger",
    "cheetah",
    "brown bear",
    "american black bear",
    "sloth bear",
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
}

VEHICLE_KEYWORDS = {
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
    "bicycle-built-for-two",
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
}

FOOD_KEYWORDS = {
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
    "cup",
    "plate",
    "dining table",
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
}

BUILDING_KEYWORDS = {
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

        if self._matches(normalized, PEOPLE_KEYWORDS):
            return "people"
        if self._matches(normalized, SCENERY_KEYWORDS):
            return "scenery"
        if self._matches(normalized, ANIMAL_KEYWORDS):
            return "animals"
        if self._matches(normalized, VEHICLE_KEYWORDS):
            return "vehicles"
        if self._matches(normalized, FOOD_KEYWORDS):
            return "food"
        if self._matches(normalized, BUILDING_KEYWORDS):
            return "buildings"

        # If neither matches, default to objects.
        return "objects"

    @staticmethod
    def _matches(labels: Iterable[str], keywords: set[str]) -> bool:
        for label in labels:
            for keyword in keywords:
                if label == keyword or keyword in label or label in keyword:
                    return True
        return False


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
    root.title("Image Organizer Pro")
    root.geometry("920x460")
    root.configure(bg="#0f172a")

    source_var = tk.StringVar()
    output_var = tk.StringVar(value=str(Path("organized_images").resolve()))
    move_var = tk.BooleanVar(value=False)
    confidence_var = tk.StringVar(value="0.20")
    status_var = tk.StringVar(value="Ready")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Main.TFrame", background="#0f172a")
    style.configure("Card.TFrame", background="#111827")
    style.configure("Main.TLabel", background="#111827", foreground="#e5e7eb", font=("Segoe UI", 11))
    style.configure("Title.TLabel", background="#0f172a", foreground="#f8fafc", font=("Segoe UI", 18, "bold"))
    style.configure("Hint.TLabel", background="#0f172a", foreground="#93c5fd", font=("Segoe UI", 10))
    style.configure("Main.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))
    style.configure("Main.TCheckbutton", background="#111827", foreground="#e5e7eb", font=("Segoe UI", 10))
    style.configure("Main.TEntry", fieldbackground="#1f2937", foreground="#f9fafb")

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

    outer = ttk.Frame(root, style="Main.TFrame", padding=18)
    outer.pack(fill="both", expand=True)

    ttk.Label(outer, text="Image Organizer Pro", style="Title.TLabel").pack(anchor="w")
    ttk.Label(
        outer,
        text="更多分类、更高颜值：people / scenery / animals / vehicles / food / buildings / objects",
        style="Hint.TLabel",
    ).pack(anchor="w", pady=(4, 14))

    frm = ttk.Frame(outer, style="Card.TFrame", padding=18)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="Source folder", style="Main.TLabel").grid(row=0, column=0, sticky="w", pady=8)
    ttk.Entry(frm, textvariable=source_var, width=82, style="Main.TEntry").grid(
        row=0, column=1, padx=10, sticky="ew"
    )
    ttk.Button(frm, text="Browse", command=choose_source, style="Main.TButton").grid(row=0, column=2)

    ttk.Label(frm, text="Output folder", style="Main.TLabel").grid(row=1, column=0, sticky="w", pady=8)
    ttk.Entry(frm, textvariable=output_var, width=82, style="Main.TEntry").grid(
        row=1, column=1, padx=10, sticky="ew"
    )
    ttk.Button(frm, text="Browse", command=choose_output, style="Main.TButton").grid(row=1, column=2)

    ttk.Label(frm, text="Min confidence (0~1)", style="Main.TLabel").grid(row=2, column=0, sticky="w", pady=8)
    ttk.Entry(frm, textvariable=confidence_var, width=18, style="Main.TEntry").grid(
        row=2, column=1, sticky="w", padx=10
    )

    ttk.Checkbutton(
        frm,
        text="Move files (instead of copy)",
        variable=move_var,
        style="Main.TCheckbutton",
    ).grid(row=3, column=1, sticky="w", padx=10, pady=8)

    action_row = ttk.Frame(frm, style="Card.TFrame")
    action_row.grid(row=4, column=1, sticky="ew", padx=10, pady=(16, 0))

    start_btn = ttk.Button(action_row, text="Start organizing", command=run_task, style="Main.TButton")
    start_btn.pack(side="left")
    ttk.Label(action_row, textvariable=status_var, style="Main.TLabel").pack(side="right")

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
