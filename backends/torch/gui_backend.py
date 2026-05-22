"""
GUI for inference using the PyTorch backend.

The UI intentionally mirrors the legacy NumPy GUI so backend switching does not
change the user workflow. Runtime class labels are resolved from the dataset so
this GUI stays aligned with the active project layout.
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
from tkinter import filedialog, messagebox

from utils.safety import install_dataset_write_guard

install_dataset_write_guard()

import config
from backends.torch.model import DEFAULT_OMEGA_FEATURE_DIM, TorchCNN, resolve_checkpoint_runtime_config
from data.loaders import load_image
from data.preprocessing import preprocess_image

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _validate_checkpoint_overrides(
    *,
    class_count_override: int | None,
    width_scale_override: float | None,
    checkpoint_num_classes: int,
    checkpoint_width_scale: float,
) -> None:
    if class_count_override is not None and int(class_count_override) != int(checkpoint_num_classes):
        raise ValueError(
            f"--class-count={class_count_override} conflicts with checkpoint num_classes={checkpoint_num_classes}"
        )
    if width_scale_override is not None and abs(float(width_scale_override) - float(checkpoint_width_scale)) > 1e-9:
        raise ValueError(
            f"--model-width-scale={width_scale_override} conflicts with checkpoint width_scale={checkpoint_width_scale:.6f}"
        )


class InferenceApp:
    def __init__(
        self,
        root: tk.Tk,
        class_names: list[str],
        width_scale: float | None,
        *,
        class_source_dir: Path | None,
        class_count_override: int | None,
    ) -> None:
        self.root = root
        self.root.title("Nix Vision Pro - PyTorch Inference GUI")
        self.root.geometry("960x640")

        self.input_size = config.INPUT_SIZE
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.device = _resolve_device()
        self.width_scale_override = width_scale
        self.width_scale = 0.75 if width_scale is None else float(width_scale)
        self.class_source_dir = class_source_dir
        self.class_count_override = class_count_override

        self.model: Optional[TorchCNN] = None
        self.weights_path: Optional[Path] = None
        self._webcam_running = False
        self._webcam_thread: Optional[threading.Thread] = None
        self._last_frame: Optional[np.ndarray] = None

        self._build_ui()
        self._init_model()

    def _build_ui(self) -> None:
        bar = tk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        self.weights_label = tk.Label(bar, text="未加载权重 (.pt/.pth)")
        self.weights_label.pack(side=tk.LEFT)

        tk.Button(bar, text="选择权重(.pt/.pth)", command=self.on_choose_weights).pack(side=tk.LEFT, padx=6)
        tk.Button(bar, text="上传图片识别", command=self.on_choose_image).pack(side=tk.LEFT, padx=6)

        self.webcam_btn = tk.Button(bar, text="打开摄像头识别", command=self.on_toggle_webcam)
        self.webcam_btn.pack(side=tk.LEFT, padx=6)
        if not _HAS_CV2:
            self.webcam_btn.config(state=tk.DISABLED)
            tk.Label(bar, text="未检测到OpenCV，无法使用摄像头。", fg="#a00").pack(side=tk.LEFT, padx=6)

        content = tk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Label(content, bg="#222")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        side = tk.Frame(content)
        side.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(side, text="预测结果", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, padx=8, pady=(8, 4))
        self.pred_text = tk.Text(side, width=36, height=24)
        self.pred_text.pack(padx=8, pady=4)
        self.pred_text.insert(tk.END, "加载权重后，上传图片或打开摄像头进行识别。\n")
        self.pred_text.config(state=tk.DISABLED)

    def _init_model(self) -> None:
        self.model = TorchCNN(input_size=self.input_size, num_classes=self.num_classes, seed=42, width_scale=self.width_scale)
        self.model.to(self.device)
        self.model.eval()

    def on_choose_weights(self) -> None:
        path_str = filedialog.askopenfilename(
            title="选择 .pt/.pth 权重文件",
            filetypes=[("PyTorch Weights", ".pt .pth"), ("所有文件", "*.*")],
        )
        if not path_str:
            return
        checkpoint = Path(path_str)
        try:
            checkpoint_config = resolve_checkpoint_runtime_config(
                checkpoint,
                map_location=self.device,
                default_input_size=self.input_size,
            )
            _validate_checkpoint_overrides(
                class_count_override=self.class_count_override,
                width_scale_override=self.width_scale_override,
                checkpoint_num_classes=checkpoint_config.num_classes,
                checkpoint_width_scale=checkpoint_config.width_scale,
            )
            self.input_size = checkpoint_config.input_size
            self.num_classes = checkpoint_config.num_classes
            self.width_scale = checkpoint_config.width_scale
            self.class_names = config.resolve_runtime_class_names(
                self.class_source_dir,
                num_classes=self.num_classes,
                checkpoint_class_names=checkpoint_config.class_names,
                require_images=False,
            )
            self.model = TorchCNN(
                input_size=self.input_size,
                num_classes=self.num_classes,
                seed=42,
                width_scale=self.width_scale,
                omega_enabled=checkpoint_config.omega_enabled,
                omega_projector_depth=checkpoint_config.omega_projector_depth or 1,
                omega_hidden_dim=checkpoint_config.omega_hidden_dim or DEFAULT_OMEGA_FEATURE_DIM,
            )
            self.model.load_weights(checkpoint, map_location=self.device)
            self.model.to(self.device)
            self.model.eval()
            self.weights_path = checkpoint
            self._set_status(f"已加载权重: {checkpoint}")
        except Exception as exc:
            messagebox.showerror("加载失败", f"无法加载权重: {exc}")

    def on_choose_image(self) -> None:
        path_str = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图像文件", ".jpg .jpeg .png .bmp"), ("所有文件", "*.*")],
        )
        if not path_str:
            return
        try:
            self._predict_image(Path(path_str))
        except Exception as exc:
            messagebox.showerror("识别失败", str(exc))

    def on_toggle_webcam(self) -> None:
        if not _HAS_CV2:
            messagebox.showwarning("不可用", "未安装OpenCV，无法使用摄像头。请先安装 opencv-python。")
            return
        if self._webcam_running:
            self._webcam_running = False
            self.webcam_btn.config(text="打开摄像头识别")
        else:
            self._webcam_running = True
            self.webcam_btn.config(text="关闭摄像头识别")
            self._start_webcam_thread()

    def _start_webcam_thread(self) -> None:
        if self._webcam_thread and self._webcam_thread.is_alive():
            return
        worker = threading.Thread(target=self._webcam_loop, daemon=True)
        self._webcam_thread = worker
        worker.start()
        self._schedule_preview_update()

    def _webcam_loop(self) -> None:
        assert _HAS_CV2 and cv2 is not None
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            self._set_status("无法打开摄像头 0")
            self._webcam_running = False
            return
        try:
            while self._webcam_running:
                ok, frame = capture.read()
                if not ok:
                    continue
                self._last_frame = frame[:, :, ::-1]
                time.sleep(0.02)
        finally:
            capture.release()

    def _schedule_preview_update(self) -> None:
        if not self._webcam_running:
            return
        if self._last_frame is not None:
            image = Image.fromarray(self._last_frame)
            self._show_on_canvas(image)
            self._predict_pil(image)
        self.root.after(50, self._schedule_preview_update)

    def _predict_image(self, path: Path) -> None:
        image_arr = load_image(path)
        image = Image.fromarray(np.clip(image_arr, 0, 255).astype(np.uint8))
        self._show_on_canvas(image)
        self._predict_pil(image)

    def _predict_pil(self, pil_image: Image.Image) -> None:
        if self.model is None:
            messagebox.showwarning("未就绪", "模型尚未初始化。")
            return
        x = np.asarray(pil_image, dtype=np.float32)
        x = preprocess_image(
            x,
            target_size=self.input_size,
            normalize_to=config.NORMALIZE_TO,
            input_value_range=config.INPUT_VALUE_RANGE,
        )
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        self._write_pred(pred_idx, probs)

    def _show_on_canvas(self, pil_image: Image.Image) -> None:
        width = self.canvas.winfo_width() or 640
        height = self.canvas.winfo_height() or 480
        image = pil_image.copy()
        image.thumbnail((width, height))
        tk_image = ImageTk.PhotoImage(image)
        self.canvas.configure(image=tk_image)
        self.canvas.image = tk_image

    def _write_pred(self, pred_idx: int, probs: np.ndarray) -> None:
        lines = [f"预测: {self.class_names[pred_idx]}  (#{pred_idx})  置信度: {probs[pred_idx]:.4f}", "", "Top-5:"]
        top5 = np.argsort(probs)[::-1][:5]
        for index in top5:
            lines.append(f"  {self.class_names[index]}: {probs[index]:.4f}")
        self.pred_text.config(state=tk.NORMAL)
        self.pred_text.delete("1.0", tk.END)
        self.pred_text.insert(tk.END, "\n".join(lines))
        self.pred_text.config(state=tk.DISABLED)

    def _set_status(self, text: str) -> None:
        self.weights_label.config(text=text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PyTorch inference GUI")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root used to resolve class labels")
    parser.add_argument("--class-count", type=int, default=None, help="Optional class-count override for model output size")
    parser.add_argument("--model-width-scale", type=float, default=None, help="Optional width multiplier override when no checkpoint is loaded")
    args = parser.parse_args()

    class_source_dir = Path(args.data_dir) if args.data_dir else None
    try:
        class_names = config.get_class_names(class_source_dir, class_count=args.class_count, require_images=False)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    root = tk.Tk()
    InferenceApp(
        root,
        class_names=class_names,
        width_scale=args.model_width_scale,
        class_source_dir=class_source_dir,
        class_count_override=args.class_count,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
