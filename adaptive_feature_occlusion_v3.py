from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn.functional as F
from torchvision import models

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ============================================================
# Data structure
# ============================================================

@dataclass
class RegionResult:
    x0: int
    y0: int
    x1: int
    y1: int
    depth: int
    area: int

    original_score: float
    occluded_score: float
    importance: float

    mean_luminance: float
    mean_hue: float
    mean_saturation: float
    edge_density: float
    texture_std: float
    high_frequency_energy: float


# ============================================================
# Generic AI adapter
# ============================================================

class AIAdapter:
    """任意のAIを共通インターフェースへ変換する基底クラス."""

    model_type = "base"

    def predict(self, images: List[Image.Image]) -> List[Any]:
        raise NotImplementedError

    def score(self, prediction: Any, target: Any) -> float:
        raise NotImplementedError

    def target_name(self, target: Any) -> str:
        return str(target)


# ============================================================
# Torchvision classification adapter
# ============================================================

class TorchvisionClassificationAdapter(AIAdapter):
    model_type = "torchvision_classification"

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        if not hasattr(models, model_name):
            raise ValueError(f"Unknown torchvision model: {model_name}")

        constructor = getattr(models, model_name)

        known_weights = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
            "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
            "vit_b_16": models.ViT_B_16_Weights.DEFAULT,
        }

        weights = known_weights.get(model_name)

        if weights is None:
            try:
                model = constructor(weights="DEFAULT")
                from torchvision import transforms

                preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
                categories = None
            except Exception as e:
                raise RuntimeError(
                    f"Could not load default weights for {model_name}."
                ) from e
        else:
            model = constructor(weights=weights)
            preprocess = weights.transforms()
            categories = weights.meta["categories"]

        model.eval().to(device)

        self.model = model
        self.preprocess = preprocess
        self.categories = categories

    @torch.inference_mode()
    def predict(self, images: List[Image.Image]) -> List[np.ndarray]:
        outputs = []

        for start in range(0, len(images), self.batch_size):
            batch_images = images[start:start + self.batch_size]

            batch = torch.stack([
                self.preprocess(img)
                for img in batch_images
            ]).to(self.device)

            logits = self.model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()

            outputs.extend(list(probs))

        return outputs

    def score(self, prediction: np.ndarray, target: int) -> float:
        return float(prediction[target])

    def target_name(self, target: int) -> str:
        if self.categories is not None:
            return self.categories[target]
        return str(target)


# ============================================================
# YOLO adapter
# ============================================================

class YOLOAdapter(AIAdapter):
    """
    Ultralytics YOLO detection adapter.

    score:
        target class の検出confidenceの最大値を返す.

    例:
        元画像     = 0.93
        遮蔽画像   = 0.41
        importance = 0.52
    """

    model_type = "ultralytics_yolo_detection"

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        target_class: int | None = None,
        conf: float = 0.001,
    ):
        if YOLO is None:
            raise ImportError(
                "ultralytics が必要です。"
                "pip install ultralytics"
            )

        self.model_path = model_path
        self.device = device
        self.default_target_class = target_class
        self.conf = conf

        self.model = YOLO(model_path)

    def predict(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        predictions = []

        for image in images:
            # Ultralytics側でPIL画像を直接処理
            result = self.model.predict(
                source=image,
                conf=self.conf,
                device=self.device,
                verbose=False,
            )[0]

            boxes = []

            if result.boxes is not None:
                xyxy = result.boxes.xyxy.detach().cpu().numpy()
                confs = result.boxes.conf.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy().astype(int)

                for box, confidence, cls in zip(
                    xyxy, confs, classes
                ):
                    boxes.append({
                        "class_id": int(cls),
                        "confidence": float(confidence),
                        "box": box.tolist(),
                    })

            predictions.append({
                "boxes": boxes
            })

        return predictions

    def available_classes(self, prediction: Dict[str, Any]) -> List[int]:
        return sorted({
            box["class_id"]
            for box in prediction["boxes"]
        })

    def score(
        self,
        prediction: Dict[str, Any],
        target: int,
    ) -> float:
        scores = [
            box["confidence"]
            for box in prediction["boxes"]
            if box["class_id"] == target
        ]

        if not scores:
            return 0.0

        return float(max(scores))

    def target_name(self, target: int) -> str:
        names = self.model.names

        if isinstance(names, dict):
            return str(names.get(target, target))

        if 0 <= target < len(names):
            return str(names[target])

        return str(target)


# ============================================================
# Image features
# ============================================================

def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32) / 255.0

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    d = mx - mn

    h = np.zeros_like(mx)

    mask = d != 0

    rmask = mask & (mx == r)
    gmask = mask & (mx == g)
    bmask = mask & (mx == b)

    h[rmask] = ((g[rmask] - b[rmask]) / d[rmask]) % 6
    h[gmask] = ((b[mask & (mx == g)] - r[mask & (mx == g)])
                / d[mask & (mx == g)]) + 2
    h[bmask] = ((r[mask & (mx == b)] - g[mask & (mx == b)])
                / d[mask & (mx == b)]) + 4

    h /= 6.0

    s = np.zeros_like(mx)
    nz = mx != 0
    s[nz] = d[nz] / mx[nz]

    v = mx

    return np.stack([h, s, v], axis=-1)


def compute_features(image: Image.Image) -> Dict[str, float]:
    rgb = np.asarray(image.convert("RGB"))

    if rgb.size == 0:
        return {
            "mean_luminance": 0.0,
            "mean_hue": 0.0,
            "mean_saturation": 0.0,
            "edge_density": 0.0,
            "texture_std": 0.0,
            "high_frequency_energy": 0.0,
        }

    gray = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    ) / 255.0

    hsv = rgb_to_hsv_np(rgb)

    mean_luminance = float(gray.mean())

    h = hsv[..., 0] * 2.0 * np.pi
    hue_x = np.cos(h).mean()
    hue_y = np.sin(h).mean()

    mean_hue = float(
        (np.arctan2(hue_y, hue_x) % (2 * np.pi))
        / (2 * np.pi)
    )

    mean_saturation = float(hsv[..., 1].mean())

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)

    if gray.shape[1] >= 3:
        gx[:, 1:-1] = (
            gray[:, 2:] - gray[:, :-2]
        ) * 0.5

    if gray.shape[0] >= 3:
        gy[1:-1, :] = (
            gray[2:, :] - gray[:-2, :]
        ) * 0.5

    magnitude = np.sqrt(gx * gx + gy * gy)

    edge_density = float(
        (magnitude > 0.10).mean()
    )

    pil_gray = Image.fromarray(
        np.uint8(gray * 255)
    )

    blurred = np.asarray(
        pil_gray.filter(
            ImageFilter.GaussianBlur(radius=1.5)
        ),
        dtype=np.float32,
    ) / 255.0

    residual = gray - blurred

    texture_std = float(residual.std())

    fft = np.fft.fftshift(
        np.fft.fft2(
            gray - gray.mean()
        )
    )

    power = np.abs(fft) ** 2

    H, W = gray.shape

    yy, xx = np.ogrid[:H, :W]

    cy = H / 2.0
    cx = W / 2.0

    radius = np.sqrt(
        (yy - cy) ** 2
        + (xx - cx) ** 2
    )

    cutoff = 0.25 * min(H, W)

    high = power[radius > cutoff]

    high_frequency_energy = float(
        high.mean()
        / (power.mean() + 1e-12)
    )

    return {
        "mean_luminance": mean_luminance,
        "mean_hue": mean_hue,
        "mean_saturation": mean_saturation,
        "edge_density": edge_density,
        "texture_std": texture_std,
        "high_frequency_energy": high_frequency_energy,
    }


# ============================================================
# Occlusion
# ============================================================

def make_occluded(
    image: Image.Image,
    box: Tuple[int, int, int, int],
    mode: str = "mean",
):
    img = image.convert("RGB")
    arr = np.asarray(img).copy()

    x0, y0, x1, y1 = box

    if mode == "mean":
        fill = arr.reshape(-1, 3).mean(axis=0)
        arr[y0:y1, x0:x1] = np.uint8(
            np.clip(fill, 0, 255)
        )

    elif mode == "gray":
        arr[y0:y1, x0:x1] = 127

    elif mode == "blur":
        blurred = np.asarray(
            img.filter(
                ImageFilter.GaussianBlur(radius=15)
            )
        )
        arr[y0:y1, x0:x1] = blurred[y0:y1, x0:x1]

    else:
        raise ValueError(
            f"Unknown occlusion mode: {mode}"
        )

    return Image.fromarray(arr)


# ============================================================
# Quadtree
# ============================================================

def split_region(
    box: Tuple[int, int, int, int],
    min_size: int,
):
    x0, y0, x1, y1 = box

    w = x1 - x0
    h = y1 - y0

    if w <= min_size or h <= min_size:
        return []

    mx = x0 + w // 2
    my = y0 + h // 2

    children = [
        (x0, y0, mx, my),
        (mx, y0, x1, my),
        (x0, my, mx, y1),
        (mx, my, x1, y1),
    ]

    return [
        b for b in children
        if b[2] > b[0] and b[3] > b[1]
    ]


# ============================================================
# Analyzer
# ============================================================

class AdaptiveOcclusionAnalyzer:

    def __init__(
        self,
        adapter: AIAdapter,
        threshold=0.10,
        max_depth=4,
        min_size=32,
        occlusion_mode="mean",
    ):
        self.adapter = adapter
        self.threshold = threshold
        self.max_depth = max_depth
        self.min_size = min_size
        self.occlusion_mode = occlusion_mode

        self.results: List[RegionResult] = []

        self.images_inferred = 0

    def evaluate_regions(
        self,
        image: Image.Image,
        regions,
        original_score,
    ):
        occluded_images = [
            make_occluded(
                image,
                box,
                mode=self.occlusion_mode,
            )
            for box in regions
        ]

        predictions = self.adapter.predict(
            occluded_images
        )

        self.images_inferred += len(
            occluded_images
        )

        scores = [
            self.adapter.score(
                prediction,
                self.target_class,
            )
            for prediction in predictions
        ]

        return scores

    def analyze(
        self,
        image: Image.Image,
        target_class=None,
    ):
        start_time = time.perf_counter()

        original_prediction = self.adapter.predict(
            [image]
        )[0]

        self.images_inferred += 1

        if target_class is None:
            raise ValueError(
                "--target-class is required for YOLO "
                "and generic detection models."
            )

        self.target_class = target_class

        original_score = self.adapter.score(
            original_prediction,
            target_class,
        )

        root = (
            0,
            0,
            image.width,
            image.height,
        )

        current_regions = [root]

        for depth in range(
            self.max_depth + 1
        ):
            if not current_regions:
                break

            print(
                f"Depth {depth}: "
                f"{len(current_regions)} regions"
            )

            scores = self.evaluate_regions(
                image,
                current_regions,
                original_score,
            )

            next_regions = []

            for box, occluded_score in zip(
                current_regions,
                scores,
            ):
                importance = (
                    original_score
                    - occluded_score
                )

                features = compute_features(
                    image.crop(box)
                )

                x0, y0, x1, y1 = box

                self.results.append(
                    RegionResult(
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        depth=depth,
                        area=(
                            (x1 - x0)
                            * (y1 - y0)
                        ),
                        original_score=original_score,
                        occluded_score=occluded_score,
                        importance=importance,
                        **features,
                    )
                )

                if (
                    depth < self.max_depth
                    and importance >= self.threshold
                ):
                    children = split_region(
                        box,
                        self.min_size,
                    )

                    next_regions.extend(
                        children
                    )

            current_regions = next_regions

        elapsed = (
            time.perf_counter()
            - start_time
        )

        self.elapsed_seconds = elapsed
        self.original_prediction = original_prediction
        self.original_score = original_score

        return original_prediction


# ============================================================
# Heatmaps
# ============================================================

def build_heatmap(
    results: List[RegionResult],
    width: int,
    height: int,
    attribute: str,
):
    heat = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    depth_map = np.full(
        (height, width),
        -1,
        dtype=np.int16,
    )

    for r in results:
        x0, y0, x1, y1 = (
            r.x0,
            r.y0,
            r.x1,
            r.y1,
        )

        value = float(
            getattr(r, attribute)
        )

        current_depth = depth_map[
            y0:y1,
            x0:x1
        ]

        current_heat = heat[
            y0:y1,
            x0:x1
        ]

        mask = r.depth >= current_depth

        current_heat[mask] = value
        current_depth[mask] = r.depth

    return heat


def save_heatmap_png(
    heat,
    path,
    title,
    cmap="viridis",
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.imshow(heat, cmap=cmap)
    plt.colorbar(label=title)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def save_overlay_png(
    image,
    heat,
    path,
    title,
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.imshow(
        heat,
        cmap="magma",
        alpha=0.55,
    )
    plt.colorbar(label=title)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# Numeric outputs
# ============================================================

def save_heatmap_matrix(
    heat,
    path,
):
    np.savetxt(
        path,
        heat,
        delimiter=",",
        fmt="%.8g",
    )


def save_heatmap_long(
    heat,
    path,
):
    H, W = heat.shape

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "x",
            "y",
            "value",
        ])

        for y in range(H):
            for x in range(W):
                writer.writerow([
                    x,
                    y,
                    float(heat[y, x]),
                ])


def save_regions_csv(
    results,
    path,
):
    if not results:
        return

    fields = list(
        asdict(results[0]).keys()
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                asdict(result)
            )


def save_regions_json(
    results,
    path,
):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            [
                asdict(result)
                for result in results
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Model creation
# ============================================================

def create_adapter(
    model_type: str,
    model_name: str,
    device: torch.device,
    batch_size: int,
    target_class: int | None,
):
    if model_type == "torchvision":
        return TorchvisionClassificationAdapter(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )

    if model_type == "yolo":
        return YOLOAdapter(
            model_path=model_name,
            device=device,
            target_class=target_class,
        )

    raise ValueError(
        f"Unknown model type: {model_type}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Adaptive quadtree occlusion analysis "
            "for classification and YOLO models."
        )
    )

    parser.add_argument(
        "image",
        type=str,
        help="入力画像",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="feature_analysis",
        help="出力フォルダ",
    )

    parser.add_argument(
        "--model-type",
        choices=[
            "torchvision",
            "yolo",
        ],
        default="torchvision",
        help="使用するAIの種類",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet18",
        help=(
            "torchvisionモデル名、またはYOLOモデルファイル"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help=(
            "この値以上のimportanceを持つ"
            "領域を再分割する。"
        ),
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="四分木の最大深度",
    )

    parser.add_argument(
        "--min-size",
        type=int,
        default=32,
        help="最小領域サイズ[pixel]",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="torchvisionモデルのbatch size",
    )

    parser.add_argument(
        "--occlusion",
        choices=[
            "mean",
            "gray",
            "blur",
        ],
        default="mean",
        help="遮蔽方法",
    )

    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help=(
            "解析対象クラスID。"
            "分類モデルでは省略すると元画像の予測クラス。"
            "YOLOでは必須。"
        ),
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    out = Path(args.out)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    image = Image.open(
        args.image
    ).convert("RGB")

    print(
        f"Image size: "
        f"{image.width} x {image.height}"
    )

    # target_classがない場合、分類モデルでは
    # 元画像の予測クラスを後から決定する必要がある。
    adapter = create_adapter(
        model_type=args.model_type,
        model_name=args.model_name,
        device=device,
        batch_size=args.batch_size,
        target_class=args.target_class,
    )

    # --------------------------------------------------------
    # Classification model: target class auto-selection
    # --------------------------------------------------------

    if (
        args.model_type == "torchvision"
        and args.target_class is None
    ):
        initial_prediction = adapter.predict(
            [image]
        )[0]

        target_class = int(
            np.argmax(initial_prediction)
        )

        # 最初の推論結果を捨てることを避けるため、
        # analyzer側でも再推論する。
        # 互換性を優先した実装。
    else:
        target_class = args.target_class

    if target_class is None:
        raise ValueError(
            "--target-class is required for this model."
        )

    analyzer = AdaptiveOcclusionAnalyzer(
        adapter=adapter,
        threshold=args.threshold,
        max_depth=args.max_depth,
        min_size=args.min_size,
        occlusion_mode=args.occlusion,
    )

    print()
    print("Starting analysis...")
    print()

    original_prediction = analyzer.analyze(
        image,
        target_class=target_class,
    )

    target = analyzer.target_class
    target_name = adapter.target_name(target)

    print()
    print(
        f"Model type: {adapter.model_type}"
    )
    print(
        f"Model: {args.model_name}"
    )
    print(
        f"Target class: "
        f"{target} ({target_name})"
    )
    print(
        f"Original score: "
        f"{analyzer.original_score:.6f}"
    )
    print(
        f"Regions analyzed: "
        f"{len(analyzer.results)}"
    )
    print(
        f"Images inferred: "
        f"{analyzer.images_inferred}"
    )
    print(
        f"Elapsed time: "
        f"{analyzer.elapsed_seconds:.2f} s"
    )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    image.save(
        out / "original.png"
    )

    # --------------------------------------------------------
    # Heatmaps
    # --------------------------------------------------------

    heatmap_attributes = [
        "importance",
        "mean_luminance",
        "mean_hue",
        "mean_saturation",
        "edge_density",
        "texture_std",
        "high_frequency_energy",
    ]

    for attribute in heatmap_attributes:

        heat = build_heatmap(
            analyzer.results,
            image.width,
            image.height,
            attribute,
        )

        save_heatmap_png(
            heat,
            out / f"heatmap_{attribute}.png",
            attribute,
        )

        save_overlay_png(
            image,
            heat,
            out / (
                f"heatmap_{attribute}_overlay.png"
            ),
            attribute,
        )

        save_heatmap_matrix(
            heat,
            out / (
                f"heatmap_{attribute}.csv"
            ),
        )

        save_heatmap_long(
            heat,
            out / (
                f"heatmap_{attribute}_long.csv"
            ),
        )

    # --------------------------------------------------------
    # Region data
    # --------------------------------------------------------

    save_regions_csv(
        analyzer.results,
        out / "regions.csv",
    )

    save_regions_json(
        analyzer.results,
        out / "regions.json",
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "image": str(
            Path(args.image).resolve()
        ),
        "image_width": image.width,
        "image_height": image.height,
        "model_type": adapter.model_type,
        "model": args.model_name,
        "device": str(device),
        "target_class": target,
        "target_name": target_name,
        "original_score": float(
            analyzer.original_score
        ),
        "threshold": args.threshold,
        "max_depth": args.max_depth,
        "min_size": args.min_size,
        "batch_size": args.batch_size,
        "occlusion_mode": args.occlusion,
        "regions_analyzed": len(
            analyzer.results
        ),
        "images_inferred": (
            analyzer.images_inferred
        ),
        "elapsed_seconds": (
            analyzer.elapsed_seconds
        ),
        "feature_names": [
            "mean_luminance",
            "mean_hue",
            "mean_saturation",
            "edge_density",
            "texture_std",
            "high_frequency_energy",
        ],
        "score_definition": (
            "classification: target class probability; "
            "YOLO: maximum confidence of target class"
        ),
        "note": (
            "Image-derived feature maps are "
            "candidate visual features. They "
            "are not direct neural-network "
            "internal feature representations."
        ),
    }

    with open(
        out / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("Results saved to:")
    print(out.resolve())


if __name__ == "__main__":
    main()
