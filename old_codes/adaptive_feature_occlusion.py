"""
Adaptive Feature Occlusion Analyzer
-----------------------------------

目的
    学習済み画像分類モデルに対し、四分木（quadtree）で画像領域を
    再帰的に遮蔽し、モデル出力の変化から「どの空間領域が重要か」を
    推定する基盤コード。

    同時に、各領域について候補視覚特徴量
        - mean_luminance
        - mean_hue
        - mean_saturation
        - edge_density
        - texture_std
        - high_frequency_energy
    を計算し、各特徴量の空間ヒートマップを出力する。

重要
    この初版は「特徴量そのものがモデル内部に存在する」と仮定しない。
    画像から測定可能な特徴量と、モデル出力に対する遮蔽感度を
    対応付けるための解析基盤である。

対応モデル
    torchvision の ImageNet 系分類モデルを想定。
    --model-name を変更すれば torchvision.models の他の分類モデルも利用可能。

例
    python adaptive_feature_occlusion.py image.jpg --out result
    python adaptive_feature_occlusion.py image.jpg --model-name resnet18 --out result

出力
    result/
        original.png
        importance_heatmap.png
        feature_mean_luminance.png
        feature_mean_hue.png
        feature_mean_saturation.png
        feature_edge_density.png
        feature_texture_std.png
        feature_high_frequency_energy.png
        regions.csv
        regions.json
        metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn.functional as F
from torchvision import models, transforms


# ============================================================
# Data structures
# ============================================================

@dataclass
class RegionResult:
    x0: int
    y0: int
    x1: int
    y1: int
    depth: int
    area: int

    # Model sensitivity
    original_probability: float
    occluded_probability: float
    importance: float

    # Candidate image features
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
    torchvision classification model.
    ImageNet weights are used by default.
    """
    if not hasattr(models, model_name):
        raise ValueError(
            f"Unknown torchvision model: {model_name}"
        )

    constructor = getattr(models, model_name)

    # New torchvision API
    weights = None
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
    elif model_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT

    if weights is not None:
        model = constructor(weights=weights)
        preprocess = weights.transforms()
        categories = weights.meta["categories"]
    else:
        # Fallback for models without an explicit mapping above.
        model = constructor(weights="DEFAULT")
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

    model.eval().to(device)
    return model, preprocess, categories


@torch.inference_mode()
def predict(model, preprocess, image: Image.Image, device):
    x = preprocess(image).unsqueeze(0).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    return probs[0].detach().cpu().numpy()


# ============================================================
# Image features
# ============================================================

def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    """
    RGB uint8 -> HSV float32.
    H: [0, 1), S: [0, 1], V: [0, 1]
    """
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

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
    画像全体の候補特徴量を計算。
    """
    rgb = np.asarray(image.convert("RGB"))
    gray = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    ) / 255.0

    hsv = rgb_to_hsv_np(rgb)

    # Luminance
    mean_luminance = float(gray.mean())

    # Hue:
    # 円環量なので、単純平均ではなく circular mean を使用。
    h = hsv[..., 0] * 2.0 * np.pi
    hue_x = np.cos(h).mean()
    hue_y = np.sin(h).mean()
    mean_hue = float((np.arctan2(hue_y, hue_x) % (2*np.pi)) / (2*np.pi))

    mean_saturation = float(hsv[..., 1].mean())

    # Sobel-like gradient without scipy
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
    magnitude = np.sqrt(gx * gx + gy * gy)

    edge_density = float((magnitude > 0.10).mean())

    # Local texture proxy:
    # 周辺平均との差の標準偏差
    pil_gray = Image.fromarray(np.uint8(gray * 255))
    blurred = np.asarray(
        pil_gray.filter(ImageFilter.GaussianBlur(radius=1.5)),
        dtype=np.float32,
    ) / 255.0
    texture_residual = gray - blurred
    texture_std = float(texture_residual.std())

    # High-frequency energy:
    # FFT の高周波領域の平均エネルギー
    fft = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    power = np.abs(fft) ** 2

    H, W = gray.shape
    yy, xx = np.ogrid[:H, :W]
    cy, cx = H / 2.0, W / 2.0
    radius = np.sqrt((yy-cy)**2 + (xx-cx)**2)

    cutoff = 0.25 * min(H, W)
    high = power[radius > cutoff]

    high_frequency_energy = float(
        high.mean() / (power.mean() + 1e-12)
    )

    return {
        "mean_luminance": mean_luminance,
        "mean_hue": mean_hue,
        "mean_saturation": mean_saturation,
        "edge_density": edge_density,
        "texture_std": texture_std,
        "high_frequency_energy": high_frequency_energy,
    }


def compute_region_features(image: Image.Image, box):
    crop = image.crop(box)
    return compute_features(crop)


# ============================================================
# Occlusion
# ============================================================

def make_occluded(
    image: Image.Image,
    box: Tuple[int, int, int, int],
    mode: str = "mean",
):
    """
    遮蔽方法。

    mean:
        遮蔽領域の周辺ではなく、画像全体の平均RGBで埋める。
        初期実装として安定。

    gray:
        127で埋める。

    blur:
        元画像をGaussian blurして該当領域に入れる。
    """
    img = image.convert("RGB")
    arr = np.asarray(img).copy()

    x0, y0, x1, y1 = box

    if mode == "mean":
        fill = arr.reshape(-1, 3).mean(axis=0)
        arr[y0:y1, x0:x1] = np.uint8(np.clip(fill, 0, 255))

    elif mode == "gray":
        arr[y0:y1, x0:x1] = 127

    elif mode == "blur":
        blurred = np.asarray(
            img.filter(ImageFilter.GaussianBlur(radius=15))
        )
        arr[y0:y1, x0:x1] = blurred[y0:y1, x0:x1]

    else:
        raise ValueError(f"Unknown occlusion mode: {mode}")

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
# Adaptive analysis
# ============================================================

class AdaptiveOcclusionAnalyzer:

    def __init__(
        self,
        model,
        preprocess,
        device,
        threshold=0.10,
        max_depth=4,
        min_size=16,
        occlusion_mode="mean",
    ):
        self.model = model
        self.preprocess = preprocess
        self.device = device

        self.threshold = threshold
        self.max_depth = max_depth
        self.min_size = min_size
        self.occlusion_mode = occlusion_mode

        self.results: List[RegionResult] = []

    def analyze_region(
        self,
        image: Image.Image,
        box,
        original_probability,
        depth,
    ):
        occluded = make_occluded(
            image,
            box,
            mode=self.occlusion_mode,
        )

        probs = predict(
            self.model,
            self.preprocess,
            occluded,
            self.device,
        )

        # 元画像で選択された target class の確率
        target_probability = probs[self.target_class]

        importance = (
            original_probability - target_probability
        )

        features = compute_region_features(image, box)

        x0, y0, x1, y1 = box

        result = RegionResult(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            depth=depth,
            area=(x1-x0)*(y1-y0),
            original_probability=float(original_probability),
            occluded_probability=float(target_probability),
            importance=float(importance),
            **features,
        )

        self.results.append(result)

        # 重要ならさらに分割
        children = split_region(
            box,
            self.min_size,
        )

        if (
            depth < self.max_depth
            and children
            and importance >= self.threshold
        ):
            for child in children:
                self.analyze_region(
                    image,
                    child,
                    original_probability,
                    depth + 1,
                )

    def analyze(
        self,
        image: Image.Image,
        target_class=None,
    ):
        probs = predict(
            self.model,
            self.preprocess,
            image,
            self.device,
        )

        if target_class is None:
            target_class = int(np.argmax(probs))

        self.target_class = target_class

        H = image.height
        W = image.width

        self.analyze_region(
            image,
            (0, 0, W, H),
            float(probs[target_class]),
            0,
        )

        return probs


# ============================================================
# Heatmap generation
# ============================================================

def build_heatmap(
    results: List[RegionResult],
    width: int,
    height: int,
    attribute: str,
    weighted=True,
):
    """
    領域結果を元画像サイズのヒートマップへ戻す。

    weighted=True:
        quadtree の各セルの値を面積で均等に配置。
        同じ画像位置に複数階層がある場合は、
        最も細かい領域を優先する。
    """
    heat = np.full(
        (height, width),
        np.nan,
        dtype=np.float32,
    )

    depth_map = np.full(
        (height, width),
        -1,
        dtype=np.int16,
    )

    for r in results:
        x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1

        value = float(getattr(r, attribute))

        current_depth = depth_map[y0:y1, x0:x1]

        mask = r.depth >= current_depth

        region_heat = heat[y0:y1, x0:x1]
        region_heat[mask] = value

        current_depth[mask] = r.depth

    # 未探索部分は0
    heat = np.nan_to_num(heat, nan=0.0)

    return heat


def save_heatmap(
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
    plt.savefig(path, dpi=200)
    plt.close()


def save_overlay(
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
    plt.savefig(path, dpi=200)
    plt.close()


# ============================================================
# Output
# ============================================================

def save_csv(results, path):
    fields = list(asdict(results[0]).keys())

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

        for r in results:
            writer.writerow(asdict(r))


def save_json(results, path):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            [asdict(r) for r in results],
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        type=str,
    )

    parser.add_argument(
        "--out",
        type=str,
        default="feature_analysis",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet18",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="この値以上のimportanceを持つ領域を再分割する。",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--min-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--occlusion",
        choices=["mean", "gray", "blur"],
        default="mean",
    )

    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    image = Image.open(args.image).convert("RGB")

    model, preprocess, categories = load_model(
        args.model_name,
        device,
    )

    analyzer = AdaptiveOcclusionAnalyzer(
        model=model,
        preprocess=preprocess,
        device=device,
        threshold=args.threshold,
        max_depth=args.max_depth,
        min_size=args.min_size,
        occlusion_mode=args.occlusion,
    )

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
    print("Target class:")
    print(f"  {target}: {target_name}")
    print(f"  probability: {probs[target]:.6f}")
    print()
    print(f"Regions analyzed: {len(analyzer.results)}")

    image.save(out / "original.png")

    # Importance map
    importance = build_heatmap(
        analyzer.results,
        image.width,
        image.height,
        "importance",
    )

    save_heatmap(
        importance,
        out / "importance_heatmap.png",
        "Occlusion importance",
        cmap="magma",
    )

    save_overlay(
        image,
        importance,
        out / "importance_overlay.png",
        "Occlusion importance",
    )

    feature_names = [
        "mean_luminance",
        "mean_hue",
        "mean_saturation",
        "edge_density",
        "texture_std",
        "high_frequency_energy",
    ]

    for name in feature_names:

        heat = build_heatmap(
            analyzer.results,
            image.width,
            image.height,
            name,
        )

        save_heatmap(
            heat,
            out / f"feature_{name}.png",
            name,
        )

        save_overlay(
            image,
            heat,
            out / f"feature_{name}_overlay.png",
            name,
        )

    save_csv(
        analyzer.results,
        out / "regions.csv",
    )

    save_json(
        analyzer.results,
        out / "regions.json",
    )

    metadata = {
        "image": str(args.image),
        "model": args.model_name,
        "device": str(device),
        "target_class": target,
        "target_name": target_name,
        "original_probability": float(probs[target]),
        "threshold": args.threshold,
        "max_depth": args.max_depth,
        "min_size": args.min_size,
        "occlusion_mode": args.occlusion,
        "num_regions": len(analyzer.results),
        "feature_names": feature_names,
        "note": (
            "Feature maps are image-derived candidate feature "
            "maps, not direct neural-network internal features."
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
    print(f"Results saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
