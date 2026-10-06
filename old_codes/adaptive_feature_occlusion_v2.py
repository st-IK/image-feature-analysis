from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn.functional as F
from torchvision import models


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

    original_probability: float
    occluded_probability: float
    importance: float

    mean_luminance: float
    mean_hue: float
    mean_saturation: float
    edge_density: float
    texture_std: float
    high_frequency_energy: float


# ============================================================
# Model
# ============================================================

def load_model(model_name: str, device: torch.device):
    """
    torchvisionの学習済み画像分類モデルをロード.
    """
    if not hasattr(models, model_name):
        raise ValueError(
            f"Unknown torchvision model: {model_name}"
        )

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
        except Exception as e:
            raise RuntimeError(
                f"Could not load default weights for {model_name}. "
                f"Please add the model explicitly to known_weights."
            ) from e

        preprocess = models.get_model_weights(
            model_name.upper()
        ).transforms() if False else None

        # Fallback preprocessing.
        # torchvisionの標準ImageNet前処理。
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

    else:
        model = constructor(weights=weights)
        preprocess = weights.transforms()
        categories = weights.meta["categories"]

    model.eval().to(device)

    return model, preprocess, categories


# ============================================================
# Batched prediction
# ============================================================

@torch.inference_mode()
def batch_predict(
    model,
    preprocess,
    images: List[Image.Image],
    device: torch.device,
    batch_size: int,
):
    """
    複数画像をbatch_sizeごとにまとめて推論.

    return:
        shape = (N, number_of_classes)
    """
    all_probs = []

    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size]

        batch = torch.stack([
            preprocess(img)
            for img in batch_images
        ]).to(device)

        logits = model(batch)
        probs = F.softmax(logits, dim=1)

        all_probs.append(
            probs.detach().cpu()
        )

    return torch.cat(all_probs, dim=0).numpy()


# ============================================================
# Image features
# ============================================================

def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    """
    RGB uint8 -> HSV float32

    H: [0, 1)
    S: [0, 1]
    V: [0, 1]
    """
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
    h[gmask] = ((b[gmask] - r[gmask]) / d[gmask]) + 2
    h[bmask] = ((r[bmask] - g[bmask]) / d[bmask]) + 4

    h /= 6.0

    s = np.zeros_like(mx)
    nz = mx != 0
    s[nz] = d[nz] / mx[nz]

    v = mx

    return np.stack([h, s, v], axis=-1)


def compute_features(image: Image.Image) -> Dict[str, float]:
    """
    指定画像領域の候補特徴量を計算.
    """
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

    # Hueは円環量なのでcircular mean
    h = hsv[..., 0] * 2.0 * np.pi
    hue_x = np.cos(h).mean()
    hue_y = np.sin(h).mean()

    mean_hue = float(
        (np.arctan2(hue_y, hue_x) % (2 * np.pi))
        / (2 * np.pi)
    )

    mean_saturation = float(
        hsv[..., 1].mean()
    )

    # Sobelに近い単純な中央差分
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

    magnitude = np.sqrt(
        gx * gx + gy * gy
    )

    edge_density = float(
        (magnitude > 0.10).mean()
    )

    # Local texture proxy
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

    texture_std = float(
        residual.std()
    )

    # High-frequency energy
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


def compute_region_features(
    image: Image.Image,
    box: Tuple[int, int, int, int],
):
    return compute_features(
        image.crop(box)
    )


# ============================================================
# Occlusion
# ============================================================

def make_occluded(
    image: Image.Image,
    box: Tuple[int, int, int, int],
    mode: str = "mean",
):
    """
    遮蔽画像を生成。

    mean:
        画像全体の平均RGBで塗る。

    gray:
        RGB=127で塗る。

    blur:
        Gaussian blurした画像で置換。
    """
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
                ImageFilter.GaussianBlur(
                    radius=15
                )
            )
        )

        arr[y0:y1, x0:x1] = (
            blurred[y0:y1, x0:x1]
        )

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
        b
        for b in children
        if b[2] > b[0]
        and b[3] > b[1]
    ]


# ============================================================
# Analyzer
# ============================================================

class AdaptiveOcclusionAnalyzer:

    def __init__(
        self,
        model,
        preprocess,
        device,
        threshold=0.10,
        max_depth=4,
        min_size=32,
        occlusion_mode="mean",
        batch_size=8,
    ):
        self.model = model
        self.preprocess = preprocess
        self.device = device

        self.threshold = threshold
        self.max_depth = max_depth
        self.min_size = min_size
        self.occlusion_mode = occlusion_mode
        self.batch_size = batch_size

        self.results: List[RegionResult] = []

        self.inference_calls = 0
        self.images_inferred = 0

    def evaluate_regions(
        self,
        image: Image.Image,
        regions,
        original_probability,
    ):
        """
        複数領域をまとめて遮蔽し、batch inferenceする。
        """
        occluded_images = [
            make_occluded(
                image,
                box,
                mode=self.occlusion_mode,
            )
            for box in regions
        ]

        probs = batch_predict(
            self.model,
            self.preprocess,
            occluded_images,
            self.device,
            self.batch_size,
        )

        self.inference_calls += int(
            np.ceil(
                len(occluded_images)
                / self.batch_size
            )
        )

        self.images_inferred += len(
            occluded_images
        )

        return probs

    def analyze(
        self,
        image: Image.Image,
        target_class=None,
    ):
        start_time = time.perf_counter()

        # 元画像は1回だけ推論
        original_probs = batch_predict(
            self.model,
            self.preprocess,
            [image],
            self.device,
            self.batch_size,
        )[0]

        self.inference_calls += 1
        self.images_inferred += 1

        if target_class is None:
            target_class = int(
                np.argmax(original_probs)
            )

        self.target_class = target_class
        original_probability = float(
            original_probs[target_class]
        )

        # depth 0
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

            probs = self.evaluate_regions(
                image,
                current_regions,
                original_probability,
            )

            next_regions = []

            for box, prob_vector in zip(
                current_regions,
                probs,
            ):
                target_probability = float(
                    prob_vector[target_class]
                )

                importance = (
                    original_probability
                    - target_probability
                )

                features = (
                    compute_region_features(
                        image,
                        box,
                    )
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
                        original_probability=(
                            original_probability
                        ),
                        occluded_probability=(
                            target_probability
                        ),
                        importance=importance,
                        **features,
                    )
                )

                # 次階層へ
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
        self.original_probs = original_probs

        return original_probs


# ============================================================
# Heatmaps
# ============================================================

def build_heatmap(
    results: List[RegionResult],
    width: int,
    height: int,
    attribute: str,
):
    """
    四分木領域から元画像サイズのヒートマップを生成。

    同じ画素に複数階層が存在する場合は、
    より深い（細かい）領域を優先する。
    """
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

        current_depth = (
            depth_map[
                y0:y1,
                x0:x1
            ]
        )

        current_heat = (
            heat[
                y0:y1,
                x0:x1
            ]
        )

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

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        heat,
        cmap=cmap,
    )

    plt.colorbar(
        label=title
    )

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

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(image)

    plt.imshow(
        heat,
        cmap="magma",
        alpha=0.55,
    )

    plt.colorbar(
        label=title
    )

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
# Numeric heatmap output
# ============================================================

def save_heatmap_matrix(
    heat,
    path,
):
    """
    H x W の2次元CSV。
    """
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
    """
    x,y,value形式のCSV。
    """
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


# ============================================================
# Region output
# ============================================================

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
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Adaptive quadtree occlusion "
            "analysis for image classifiers."
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
        "--model-name",
        type=str,
        default="resnet18",
        help="torchvisionモデル名",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help=(
            "この値以上のimportanceを持つ "
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
        help="1回のモデル推論に入れる画像枚数",
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
            "解析対象クラス。"
            "省略すると元画像の予測クラス。"
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

    print(
        f"Device: {device}"
    )

    image = Image.open(
        args.image
    ).convert("RGB")

    print(
        f"Image size: "
        f"{image.width} x {image.height}"
    )

    model, preprocess, categories = (
        load_model(
            args.model_name,
            device,
        )
    )

    analyzer = (
        AdaptiveOcclusionAnalyzer(
            model=model,
            preprocess=preprocess,
            device=device,
            threshold=args.threshold,
            max_depth=args.max_depth,
            min_size=args.min_size,
            occlusion_mode=args.occlusion,
            batch_size=args.batch_size,
        )
    )

    print()
    print("Starting analysis...")
    print()

    probs = analyzer.analyze(
        image,
        target_class=args.target_class,
    )

    target = analyzer.target_class

    if categories is not None:
        target_name = categories[target]
    else:
        target_name = str(target)

    print()
    print(
        f"Target class: "
        f"{target} ({target_name})"
    )

    print(
        f"Original probability: "
        f"{probs[target]:.6f}"
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

        # PNG
        save_heatmap_png(
            heat,
            out / f"heatmap_{attribute}.png",
            attribute,
        )

        # Overlay
        save_overlay_png(
            image,
            heat,
            out / (
                f"heatmap_{attribute}_overlay.png"
            ),
            attribute,
        )

        # 2D CSV
        save_heatmap_matrix(
            heat,
            out / (
                f"heatmap_{attribute}.csv"
            ),
        )

        # Long CSV
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
        "model": args.model_name,
        "device": str(device),
        "target_class": target,
        "target_name": target_name,
        "original_probability": float(
            probs[target]
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
        "inference_batches": (
            analyzer.inference_calls
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
        "output_heatmap_format": {
            "matrix_csv": (
                "H x W numeric matrix"
            ),
            "long_csv": (
                "x,y,value"
            ),
        },
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
    print(
        "Results saved to:"
    )
    print(
        out.resolve()
    )


if __name__ == "__main__":
    main()
