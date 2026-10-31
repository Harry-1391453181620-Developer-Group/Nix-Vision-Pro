"""MAOL field-station GUI.

A deliberately quiet, natural-history instrument: deep ink surfaces, parchment
type, a single amber signal colour, and border-only depth.  The probability bars
are the signature readout: each class is a well whose confidence is a measured
specimen tag.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import config
from backends.torch.model import TorchCNN, resolve_checkpoint_runtime_config
from backends.torch.predict_backend import load_maol_model
from data.loaders import load_image
from data.preprocessing import preprocess_image
from utils.safety import install_dataset_write_guard

install_dataset_write_guard()

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False


INK = "#0D1411"
PANEL = "#141E18"
PANEL_RAISED = "#1A2620"
BORDER = "#2B3A31"
PARCHMENT = "#EFE7D6"
MUTED = "#9AA99E"
AMBER = "#D6A24B"
AMBER_HOVER = "#E5B967"
SAGE = "#7C9A84"
RUST = "#C06A4F"


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProbabilityBars(tk.Canvas):
    """Top-k confidence ladder with a quiet measured look."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, bg=PANEL, highlightthickness=0, height=220)
        self._names: list[str] = []
        self._values: list[float] = []
        self.bind("<Configure>", lambda _event: self._draw())

    def set_values(self, names: list[str], values: list[float]) -> None:
        order = np.argsort(values)[::-1][:5]
        self._names = [names[int(index)] for index in order]
        self._values = [float(values[int(index)]) for index in order]
        self._draw()

    def clear(self) -> None:
        self._names = []
        self._values = []
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 320)
        if not self._names:
            self.create_text(
                8,
                16,
                anchor="nw",
                text="No specimen read",
                fill=MUTED,
                font=("Segoe UI", 10),
            )
            return
        label_width = 112
        right_margin = 46
        bar_width = max(40, width - label_width - right_margin)
        y = 10
        for index, (name, value) in enumerate(zip(self._names, self._values)):
            self.create_text(
                4,
                y + 10,
                anchor="w",
                text=name,
                fill=PARCHMENT if index == 0 else MUTED,
                font=("Segoe UI", 9, "bold" if index == 0 else "normal"),
            )
            self.create_rectangle(
                label_width,
                y,
                label_width + bar_width,
                y + 16,
                outline="",
                fill=PANEL_RAISED,
            )
            self.create_rectangle(
                label_width,
                y,
                label_width + max(2, int(bar_width * max(0.0, min(1.0, value)))),
                y + 16,
                outline="",
                fill=AMBER if index == 0 else SAGE,
            )
            self.create_text(
                width - 4,
                y + 10,
                anchor="e",
                text=f"{value * 100:5.1f}%",
                fill=PARCHMENT,
                font=("Cascadia Mono", 9),
            )
            y += 32


class InferenceApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        initial_checkpoint: Path | None = None,
    ) -> None:
        self.root = root
        self.root.title("Nix Vision / MAOL Field Station")
        self.root.geometry("1180x760")
        self.root.minsize(980, 640)
        self.root.configure(bg=INK)

        self.device = _resolve_device()
        self.model: Optional[TorchCNN] = None
        self.weights_path: Optional[Path] = None
        self.class_names: list[str] = []
        self.input_size = config.INPUT_SIZE
        self._current_image: Optional[Image.Image] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._last_frame: Optional[np.ndarray] = None
        self._webcam_running = False
        self._webcam_thread: Optional[threading.Thread] = None

        self._configure_style()
        self._build_ui()
        if initial_checkpoint is not None:
            self.root.after(120, lambda: self._load_checkpoint(initial_checkpoint))

    # ------------------------------------------------------------------
    # Style and layout
    # ------------------------------------------------------------------
    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=(12, 8), borderwidth=0)
        style.configure(
            "Accent.TButton",
            background=AMBER,
            foreground=INK,
            focuscolor=AMBER,
            borderwidth=0,
            padding=(14, 9),
        )
        style.map(
            "Accent.TButton",
            background=[("active", AMBER_HOVER), ("disabled", "#5B4A2D")],
            foreground=[("disabled", "#2A2419")],
        )
        style.configure(
            "Ghost.TButton",
            background=PANEL_RAISED,
            foreground=PARCHMENT,
            focuscolor=BORDER,
            borderwidth=0,
            padding=(12, 9),
        )
        style.map(
            "Ghost.TButton",
            background=[("active", "#26352C"), ("disabled", "#18211C")],
            foreground=[("disabled", "#58665D")],
        )

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg=INK, height=84)
        header.pack(side=tk.TOP, fill=tk.X, padx=28, pady=(20, 0))
        header.pack_propagate(False)

        title_block = tk.Frame(header, bg=INK)
        title_block.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(
            title_block,
            text="NIX VISION",
            bg=INK,
            fg=PARCHMENT,
            font=("Georgia", 24, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_block,
            text="MAOL FIELD STATION  /  MULTI-ATTRACTOR WELLS",
            bg=INK,
            fg=AMBER,
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", pady=(2, 0))

        status_block = tk.Frame(header, bg=INK)
        status_block.pack(side=tk.RIGHT, fill=tk.Y)
        self.header_status = tk.Label(
            status_block,
            text="NO CHECKPOINT",
            bg=INK,
            fg=MUTED,
            font=("Cascadia Mono", 9),
        )
        self.header_status.pack(anchor="e", pady=(8, 0))
        self.device_label = tk.Label(
            status_block,
            text=f"DEVICE  {str(self.device).upper()}",
            bg=INK,
            fg=SAGE,
            font=("Cascadia Mono", 8),
        )
        self.device_label.pack(anchor="e", pady=(2, 0))

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X, padx=28, pady=(12, 18))

        body = tk.Frame(self.root, bg=INK)
        body.pack(fill=tk.BOTH, expand=True, padx=28, pady=(0, 18))

        left = tk.Frame(body, bg=INK)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.image_canvas = tk.Canvas(
            left,
            bg="#0A100D",
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        self.image_canvas.bind("<Configure>", lambda _event: self._render_current_image())

        controls = tk.Frame(left, bg=INK)
        controls.pack(fill=tk.X, pady=(14, 0))
        self.open_button = ttk.Button(
            controls,
            text="OPEN SPECIMEN",
            style="Accent.TButton",
            command=self.on_choose_image,
            state=tk.DISABLED,
        )
        self.open_button.pack(side=tk.LEFT)
        self.camera_button = ttk.Button(
            controls,
            text="CAMERA",
            style="Ghost.TButton",
            command=self.on_toggle_webcam,
            state=tk.DISABLED,
        )
        self.camera_button.pack(side=tk.LEFT, padx=(10, 0))
        self.checkpoint_button = ttk.Button(
            controls,
            text="LOAD CHECKPOINT",
            style="Ghost.TButton",
            command=self.on_choose_weights,
        )
        self.checkpoint_button.pack(side=tk.RIGHT)

        right = tk.Frame(body, bg=PANEL, width=392)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(22, 0))
        right.pack_propagate(False)

        tk.Label(
            right,
            text="SPECIMEN READOUT",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", padx=22, pady=(20, 0))

        self.prediction_label = tk.Label(
            right,
            text="Awaiting\nspecimen",
            bg=PANEL,
            fg=PARCHMENT,
            font=("Georgia", 30, "bold"),
            justify=tk.LEFT,
            anchor="w",
        )
        self.prediction_label.pack(fill=tk.X, padx=22, pady=(2, 0))

        self.confidence_label = tk.Label(
            right,
            text="—",
            bg=PANEL,
            fg=AMBER,
            font=("Cascadia Mono", 22, "bold"),
        )
        self.confidence_label.pack(anchor="w", padx=22, pady=(4, 0))

        self.bars = ProbabilityBars(right)
        self.bars.pack(fill=tk.X, padx=16, pady=(18, 6))

        tk.Frame(right, bg=BORDER, height=1).pack(fill=tk.X, padx=22, pady=(8, 14))

        self.meta_frame = tk.Frame(right, bg=PANEL)
        self.meta_frame.pack(fill=tk.X, padx=22)
        self.meta_labels: dict[str, tk.Label] = {}
        for row, (key, label) in enumerate(
            (
                ("classes", "CLASSES"),
                ("tree", "TREE DEPTH"),
                ("tau", "TAU"),
                ("checkpoint", "CHECKPOINT"),
            )
        ):
            tk.Label(
                self.meta_frame,
                text=label,
                bg=PANEL,
                fg=MUTED,
                font=("Segoe UI", 8, "bold"),
            ).grid(row=row * 2, column=0, sticky="w", pady=(0, 2))
            value = tk.Label(
                self.meta_frame,
                text="—",
                bg=PANEL,
                fg=PARCHMENT,
                font=("Cascadia Mono", 9),
                anchor="w",
                justify=tk.LEFT,
                wraplength=330,
            )
            value.grid(row=row * 2 + 1, column=0, sticky="w", pady=(0, 10))
            self.meta_labels[key] = value

        footer = tk.Frame(self.root, bg=INK)
        footer.pack(side=tk.BOTTOM, fill=tk.X, padx=28, pady=(0, 14))
        self.status_label = tk.Label(
            footer,
            text="Load a MAOL checkpoint to begin.",
            bg=INK,
            fg=MUTED,
            font=("Segoe UI", 9),
            anchor="w",
        )
        self.status_label.pack(side=tk.LEFT)
        tk.Label(
            footer,
            text="CNN  /  PROTOTYPE HEAD",
            bg=INK,
            fg=SAGE,
            font=("Cascadia Mono", 8),
        ).pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _set_status(self, text: str, *, error: bool = False) -> None:
        self.status_label.config(text=text, fg=RUST if error else MUTED)

    def _set_header_status(self, text: str, *, ready: bool = False) -> None:
        self.header_status.config(text=text, fg=SAGE if ready else MUTED)

    def _set_buttons_ready(self) -> None:
        self.open_button.config(state=tk.NORMAL if self.model is not None else tk.DISABLED)
        camera_state = tk.NORMAL if _HAS_CV2 and self.model is not None else tk.DISABLED
        self.camera_button.config(state=camera_state)

    def _update_metadata(self) -> None:
        if self.model is None:
            for label in self.meta_labels.values():
                label.config(text="—")
            return
        tree = self.model.tree
        self.meta_labels["classes"].config(text=str(self.model.num_classes))
        self.meta_labels["tree"].config(text=f"{max(tree.depth) + 1} levels / {tree.num_nodes} nodes")
        self.meta_labels["tau"].config(text=f"{float(self.model.head.tau):.4f}")
        self.meta_labels["checkpoint"].config(
            text=self.weights_path.name if self.weights_path is not None else "—"
        )

    # ------------------------------------------------------------------
    # Checkpoint and prediction
    # ------------------------------------------------------------------
    def on_choose_weights(self) -> None:
        path_str = filedialog.askopenfilename(
            title="Choose a MAOL checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")],
        )
        if path_str:
            self._load_checkpoint(Path(path_str))

    def _load_checkpoint(self, checkpoint: Path) -> None:
        try:
            config_info = resolve_checkpoint_runtime_config(checkpoint, map_location=self.device)
            if config_info.head_type != "prototype" or config_info.tree is None:
                raise ValueError(
                    "This build supports MAOL prototype checkpoints only; "
                    "the old MAOIDL architecture is not loaded."
                )
            if config_info.embedding_norm != "l2":
                raise ValueError(
                    "This checkpoint predates the L2-normalized MAOL head; retrain it."
                )
            model = load_maol_model(checkpoint, self.device)
            self.model = model
            self.weights_path = checkpoint
            self.class_names = list(model.class_names)
            self.input_size = model.input_size
            self._set_buttons_ready()
            self._update_metadata()
            self._set_header_status("CHECKPOINT READY", ready=True)
            self._set_status(f"Loaded {checkpoint.name}  /  {model.num_classes} classes")
        except Exception as exc:
            self._set_header_status("NO CHECKPOINT")
            self._set_status(f"Checkpoint rejected: {exc}", error=True)
            messagebox.showerror("Cannot load checkpoint", str(exc))

    def on_choose_image(self) -> None:
        path_str = filedialog.askopenfilename(
            title="Open a specimen image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if not path_str:
            return
        try:
            image = Image.fromarray(np.clip(load_image(path_str), 0, 255).astype(np.uint8))
            self._current_image = image
            self._render_current_image()
            self._predict_image(image)
        except Exception as exc:
            self._set_status(f"Image failed: {exc}", error=True)
            messagebox.showerror("Cannot open image", str(exc))

    def _render_current_image(self) -> None:
        self.image_canvas.delete("all")
        width = max(self.image_canvas.winfo_width(), 320)
        height = max(self.image_canvas.winfo_height(), 240)
        if self._current_image is None:
            self.image_canvas.create_text(
                width / 2,
                height / 2 - 12,
                text="NO SPECIMEN",
                fill=MUTED,
                font=("Georgia", 16, "bold"),
            )
            self.image_canvas.create_text(
                width / 2,
                height / 2 + 14,
                text="load a checkpoint, then open an image or camera",
                fill="#5D6C62",
                font=("Segoe UI", 9),
            )
            return
        image = self._current_image.copy()
        image.thumbnail((max(1, width - 24), max(1, height - 24)), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image)
        self.image_canvas.create_image(width / 2, height / 2, image=self._photo)

    def _predict_image(self, image: Image.Image) -> None:
        if self.model is None:
            self._set_status("Load a checkpoint first.", error=True)
            return
        x = np.asarray(image, dtype=np.float32)
        x = preprocess_image(
            x,
            target_size=self.input_size,
            normalize_to=config.NORMALIZE_TO,
            input_value_range=config.INPUT_VALUE_RANGE,
        )
        tensor = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0).to(
            self.device, dtype=torch.float32
        )
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        predicted = int(np.argmax(probs))
        self.prediction_label.config(text=self.class_names[predicted])
        self.confidence_label.config(text=f"{probs[predicted] * 100:.1f}%")
        self.bars.set_values(self.class_names, probs.tolist())
        self._set_status(f"Nearest leaf well: {self.class_names[predicted]}")

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def on_toggle_webcam(self) -> None:
        if not _HAS_CV2:
            messagebox.showwarning("Camera unavailable", "Install opencv-python to use the camera.")
            return
        if self._webcam_running:
            self._webcam_running = False
            self.camera_button.config(text="CAMERA")
            return
        self._webcam_running = True
        self.camera_button.config(text="STOP CAMERA")
        self._start_webcam_thread()

    def _start_webcam_thread(self) -> None:
        if self._webcam_thread is not None and self._webcam_thread.is_alive():
            return
        worker = threading.Thread(target=self._webcam_loop, daemon=True)
        self._webcam_thread = worker
        worker.start()
        self.root.after(80, self._update_preview)

    def _webcam_loop(self) -> None:
        assert _HAS_CV2 and cv2 is not None
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            self._webcam_running = False
            self._set_status("Camera 0 could not be opened.", error=True)
            return
        try:
            while self._webcam_running:
                ok, frame = capture.read()
                if not ok:
                    continue
                self._last_frame = frame[:, :, ::-1]
                time.sleep(0.03)
        finally:
            capture.release()

    def _update_preview(self) -> None:
        if not self._webcam_running:
            return
        if self._last_frame is not None:
            image = Image.fromarray(self._last_frame)
            self._current_image = image
            self._render_current_image()
            self._predict_image(image)
        self.root.after(80, self._update_preview)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MAOL inference GUI")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional MAOL checkpoint to load on start")
    args = parser.parse_args()

    root = tk.Tk()
    InferenceApp(
        root,
        initial_checkpoint=Path(args.checkpoint) if args.checkpoint else None,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
