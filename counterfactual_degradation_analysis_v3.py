from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import torch
import torch.nn.functional as F
from torchvision import models

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ============================================================
# Model adapters
# ============================================================

class ModelAdapter:
    model_type = "base"

    def predict(self, images, batch_size):
        raise NotImplementedError

    def score(self, prediction, target_class):
        raise NotImplementedError

    def predicted_class(self, prediction):
        return None

    def class_name(self, class_id):
        return str(class_id)

    def extra_fields(self, prediction, target_class):
        return {}


class TorchvisionAdapter(ModelAdapter):
    model_type = "torchvision_classification"

    def __init__(self, model_name, device):
        weights_map = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
            "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
            "vit_b_16": models.ViT_B_16_Weights.DEFAULT,
        }

        if model_name not in weights_map:
            raise ValueError(
                "Supported torchvision models: "
                + ", ".join(weights_map)
            )

        weights = weights_map[model_name]
        self.model = getattr(models, model_name)(
            weights=weights
        )
        self.model.eval().to(device)
        self.preprocess = weights.transforms()
        self.categories = weights.meta["categories"]
        self.device = device

    @torch.inference_mode()
    def predict(self, images, batch_size):
        outputs = []

        for start in range(0, len(images), batch_size):
            batch_images = images[start:start + batch_size]
            batch = torch.stack([
                self.preprocess(img)
                for img in batch_images
            ]).to(self.device)

            probs = F.softmax(
                self.model(batch), dim=1
            ).cpu().numpy()

            outputs.extend(list(probs))

        return outputs

    def score(self, prediction, target_class):
        return float(prediction[target_class])

    def predicted_class(self, prediction):
        return int(np.argmax(prediction))

    def class_name(self, class_id):
        return self.categories[class_id]


class YOLOAdapter(ModelAdapter):
    model_type = "ultralytics_yolo_detection"

    def __init__(self, model_path, device):
        if YOLO is None:
            raise ImportError(
                "YOLOにはultralyticsが必要です."
                " pip install ultralytics"
            )

        self.model = YOLO(model_path)
        self.device = device

        names = self.model.names
        self.names = (
            names
            if isinstance(names, dict)
            else {i: n for i, n in enumerate(names)}
        )

    def predict(self, images, batch_size):
        predictions = []

        for start in range(0, len(images), batch_size):
            batch_images = images[start:start + batch_size]

            results = self.model.predict(
                source=batch_images,
                device=self.device,
                verbose=False,
            )

            for result in results:
                boxes = []

                if result.boxes is not None:
                    xyxy = result.boxes.xyxy.cpu().numpy()
                    conf = result.boxes.conf.cpu().numpy()
                    cls = result.boxes.cls.cpu().numpy().astype(int)

                    for box, c, k in zip(xyxy, conf, cls):
                        boxes.append({
                            "class_id": int(k),
                            "confidence": float(c),
                            "box": box.tolist(),
                        })

                predictions.append({"boxes": boxes})

        return predictions

    def score(self, prediction, target_class):
        scores = [
            b["confidence"]
            for b in prediction["boxes"]
            if b["class_id"] == target_class
        ]
        return float(max(scores)) if scores else 0.0

    def predicted_class(self, prediction):
        if not prediction["boxes"]:
            return None
        return max(
            prediction["boxes"],
            key=lambda b: b["confidence"]
        )["class_id"]

    def class_name(self, class_id):
        return str(self.names.get(class_id, class_id))

    def extra_fields(self, prediction, target_class):
        scores = [
            b["confidence"]
            for b in prediction["boxes"]
            if b["class_id"] == target_class
        ]
        return {
            "target_detection_count": len(scores),
            "target_max_confidence": max(scores) if scores else 0.0,
            "target_mean_confidence": (
                float(np.mean(scores)) if scores else 0.0
            ),
            "total_detection_count": len(prediction["boxes"]),
        }


def create_adapter(model_type, model_name, device):
    if model_type == "torchvision":
        return TorchvisionAdapter(model_name, device)
    if model_type == "yolo":
        return YOLOAdapter(model_name, device)
    raise ValueError(f"Unknown model type: {model_type}")


# ============================================================
# Contribution mask
# ============================================================

def make_mask_from_regions(
    regions,
    width,
    height,
    contribution_column,
    threshold,
):
    mask = np.zeros((height, width), dtype=bool)
    depth_map = np.full((height, width), -1, dtype=np.int16)

    for _, row in regions.iterrows():
        value = row[contribution_column]

        if not np.isfinite(value):
            continue

        x0 = max(0, min(width, int(row["x0"])))
        x1 = max(0, min(width, int(row["x1"])))
        y0 = max(0, min(height, int(row["y0"])))
        y1 = max(0, min(height, int(row["y1"])))
        depth = int(row["depth"])

        if x1 <= x0 or y1 <= y0:
            continue

        d = depth_map[y0:y1, x0:x1]
        m = depth >= d

        view = mask[y0:y1, x0:x1]
        view[m] = value < threshold
        d[m] = depth

    return mask


# ============================================================
# Degradation
# ============================================================

def blur_region(arr, mask, radius):
    blurred = np.asarray(
        Image.fromarray(arr).filter(
            ImageFilter.GaussianBlur(radius=radius)
        )
    )
    out = arr.copy()
    out[mask] = blurred[mask]
    return out


def noise_region(arr, mask, sigma, rng):
    noise = rng.normal(0, sigma, arr.shape)
    noisy = np.clip(
        arr.astype(np.float32) + noise,
        0, 255
    ).astype(np.uint8)

    out = arr.copy()
    out[mask] = noisy[mask]
    return out


def desaturate_region(arr, mask, factor):
    rgb = arr.astype(np.float32) / 255.0
    gray = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )[..., None]

    converted = np.clip(
        (gray + (rgb - gray) * factor) * 255,
        0, 255
    ).astype(np.uint8)

    out = arr.copy()
    out[mask] = converted[mask]
    return out


def apply_degradation(
    image,
    mask,
    mode,
    blur_radius,
    noise_sigma,
    saturation_factor,
    seed,
):
    arr = np.asarray(image.convert("RGB")).copy()
    rng = np.random.default_rng(seed)

    if mode in ("blur", "blur_noise", "blur_desaturate", "all"):
        arr = blur_region(arr, mask, blur_radius)

    if mode in ("noise", "blur_noise", "noise_desaturate", "all"):
        arr = noise_region(arr, mask, noise_sigma, rng)

    if mode in (
        "desaturate",
        "blur_desaturate",
        "noise_desaturate",
        "all",
    ):
        arr = desaturate_region(
            arr, mask, saturation_factor
        )

    return Image.fromarray(arr)


# ============================================================
# Main g
# ============================================================

def main():
    # ==================
    # | code runner v3 |
    # | WRITE TAG 01   |
    # | WRITE TAG main |
    # ==================
    parser = argparse.ArgumentParser(
        description=(
            "Counterfactual degradation analysis using "
            "low-contribution image regions."
        )
    )

    parser.add_argument("image", help="元画像")
    parser.add_argument(
        "regions",
        help="regions.csv / region_contributions.csv",
    )

    parser.add_argument(
        "--out",
        default="counterfactual_analysis",
    )

    parser.add_argument(
        "--model-type",
        choices=["torchvision", "yolo"],
        default="torchvision",
        help="使用モデルの種類",
    )

    parser.add_argument(
        "--model-name",
        default="resnet18",
        help="torchvisionモデル名またはYOLO .pt",
    )

    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help=(
            "対象クラスID.分類では省略すると元画像のTop-1."
            "YOLOでは必須."
        ),
    )

    parser.add_argument(
        "--contribution-column",
        default="importance",
        help="低寄与判定に使うregions.csvの列",
    )

    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.01, 0.03, 0.05, 0.10],
        help="低寄与領域の閾値を複数指定",
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "blur",
            "noise",
            "desaturate",
            "blur_noise",
            "blur_desaturate",
            "noise_desaturate",
            "all",
        ],
        choices=[
            "blur",
            "noise",
            "desaturate",
            "blur_noise",
            "blur_desaturate",
            "noise_desaturate",
            "all",
        ],
    )

    parser.add_argument(
        "--blur-radius",
        type=float,
        default=7.0,
    )

    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=25.0,
    )

    parser.add_argument(
        "--saturation-factor",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )

    args = parser.parse_args()

    # ============================================================

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    if not 0 <= args.saturation_factor <= 1:
        raise ValueError(
            "--saturation-factor must be between 0 and 1"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    image = Image.open(args.image).convert("RGB")
    regions = pd.read_csv(args.regions)

    required = {
        "x0", "x1", "y0", "y1",
        "depth", args.contribution_column,
    }

    missing = required - set(regions.columns)
    if missing:
        raise ValueError(
            "Required columns not found: "
            + ", ".join(sorted(missing))
        )

    adapter = create_adapter(
        args.model_type,
        args.model_name,
        device,
    )

    # 対象クラスを決定
    # ============================================================
    if args.model_type == "yolo":
        if args.target_class is None:
            raise ValueError(
                "e-gev Title01 YOLOでは --target-class が必要です."
            )
        target_class = args.target_class

    else:
        initial = adapter.predict(
            [image], args.batch_size
        )[0]

        target_class = (
            args.target_class
            if args.target_class is not None
            else adapter.predicted_class(initial)
        )

    # 元画像
    original_prediction = adapter.predict(
        [image], args.batch_size
    )[0]

    original_score = adapter.score(
        original_prediction,
        target_class,
    )

    original_predicted = adapter.predicted_class(
        original_prediction
    )

    target_name = adapter.class_name(target_class)

    print(f"Model type: {adapter.model_type}")
    print(f"Model: {args.model_name}")
    print(
        f"Target: {target_class} "
        f"({target_name})"
    )
    print(
        f"Original target score: "
        f"{original_score:.6f}"
    )

    # 全条件の画像を生成
    images = []
    metadata = []

    for threshold in args.thresholds:
        mask = make_mask_from_regions(
            regions,
            image.width,
            image.height,
            args.contribution_column,
            threshold,
        )

        mask_name = (
            f"mask_{args.contribution_column}_"
            f"{threshold:.4g}.png"
        )

        Image.fromarray(
            np.uint8(mask) * 255
        ).save(out / mask_name)

        for mode in args.modes:
            degraded = apply_degradation(
                image,
                mask,
                mode,
                args.blur_radius,
                args.noise_sigma,
                args.saturation_factor,
                args.seed,
            )

            filename = (
                f"{args.contribution_column}"
                f"_threshold_{threshold:.4g}"
                f"_{mode}.png"
            )

            degraded.save(out / filename)
            images.append(degraded)

            metadata.append({
                "threshold": threshold,
                "mode": mode,
                "filename": filename,
                "mask_filename": mask_name,
                "masked_fraction": float(mask.mean()),
            })

    # 一括推論
    predictions = adapter.predict(
        images,
        args.batch_size,
    )

    rows = []

    for info, prediction in zip(
        metadata, predictions
    ):
        score = adapter.score(
            prediction,
            target_class,
        )

        predicted = adapter.predicted_class(
            prediction
        )

        row = {
            **info,
            "model_type": adapter.model_type,
            "model_name": args.model_name,
            "target_class": target_class,
            "target_class_name": target_name,
            "original_target_score": original_score,
            "degraded_target_score": score,
            "score_change": score - original_score,
            "absolute_score_change": abs(
                score - original_score
            ),
            "original_predicted_class": (
                original_predicted
            ),
            "original_predicted_class_name": (
                adapter.class_name(original_predicted)
                if original_predicted is not None
                else None
            ),
            "predicted_class": predicted,
            "predicted_class_name": (
                adapter.class_name(predicted)
                if predicted is not None
                else None
            ),
        }

        if args.model_type == "torchvision":
            row["top1_changed"] = (
                predicted != original_predicted
            )

        row.update(
            adapter.extra_fields(
                prediction,
                target_class,
            )
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    result.to_csv(
        out / "counterfactual_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        out / "counterfactual_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result.to_dict(orient="records"),
            f,
            ensure_ascii=False,
            indent=2,
        )

    summary = {
        "image": str(Path(args.image).resolve()),
        "regions": str(Path(args.regions).resolve()),
        "model_type": adapter.model_type,
        "model": args.model_name,
        "device": str(device),
        "target_class": target_class,
        "target_class_name": target_name,
        "contribution_column": args.contribution_column,
        "thresholds": args.thresholds,
        "modes": args.modes,
        "blur_radius": args.blur_radius,
        "noise_sigma": args.noise_sigma,
        "saturation_factor": args.saturation_factor,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "original_target_score": original_score,
        "original_predicted_class": original_predicted,
        "original_predicted_class_name": (
            adapter.class_name(original_predicted)
            if original_predicted is not None
            else None
        ),
        "interpretation": (
            "Low-contribution regions are progressively "
            "degraded and the target model score is "
            "compared with the original. This is a "
            "counterfactual robustness/necessity test; "
            "it does not by itself prove causal attribution."
        ),
    }

    with open(
        out / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(result.to_string(index=False))
    print()
    print(
        f"Results saved to: {out.resolve()}"
    )


if __name__ == "__main__":
    main()
