from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance

import torch
import torch.nn.functional as F
from torchvision import models


FEATURES = [
    "mean_luminance",
    "mean_hue",
    "mean_saturation",
    "edge_density",
    "texture_std",
    "high_frequency_energy",
]


def load_model(model_name, device):
    known_weights = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
        "vit_b_16": models.ViT_B_16_Weights.DEFAULT,
    }

    if model_name not in known_weights:
        raise ValueError(
            "Supported models: "
            + ", ".join(known_weights)
        )

    weights = known_weights[model_name]

    model = getattr(
        models,
        model_name,
    )(weights=weights)

    model.eval().to(device)

    return (
        model,
        weights.transforms(),
        weights.meta["categories"],
    )


@torch.inference_mode()
def predict(
    model,
    preprocess,
    images,
    device,
    batch_size,
):
    outputs = []

    for start in range(
        0,
        len(images),
        batch_size,
    ):
        batch_images = images[
            start:start + batch_size
        ]

        batch = torch.stack([
            preprocess(img)
            for img in batch_images
        ]).to(device)

        logits = model(batch)

        outputs.append(
            F.softmax(
                logits,
                dim=1,
            ).cpu().numpy()
        )

    return np.concatenate(
        outputs,
        axis=0,
    )


def make_mask_from_regions(
    regions,
    width,
    height,
    contribution_column,
    threshold,
):
    """
    contribution_column の値が threshold 未満の
    四分木領域をマスクにする。

    より深い領域を優先する。
    """
    mask = np.zeros(
        (height, width),
        dtype=bool,
    )

    depth_map = np.full(
        (height, width),
        -1,
        dtype=np.int16,
    )

    for _, row in regions.iterrows():

        value = row[
            contribution_column
        ]

        if not np.isfinite(value):
            continue

        x0 = int(row["x0"])
        x1 = int(row["x1"])
        y0 = int(row["y0"])
        y1 = int(row["y1"])
        depth = int(row["depth"])

        region_mask = (
            depth >= depth_map[
                y0:y1,
                x0:x1
            ]
        )

        if value < threshold:
            mask[
                y0:y1,
                x0:x1
            ][region_mask] = True

        depth_map[
            y0:y1,
            x0:x1
        ][region_mask] = depth

    return mask


def blur_region(
    arr,
    mask,
    radius,
):
    img = Image.fromarray(arr)
    blurred = img.filter(
        ImageFilter.GaussianBlur(
            radius=radius
        )
    )

    b = np.asarray(
        blurred
    )

    out = arr.copy()
    out[mask] = b[mask]

    return out


def noise_region(
    arr,
    mask,
    sigma,
    rng,
):
    noise = rng.normal(
        0,
        sigma,
        arr.shape,
    )

    out = arr.astype(
        np.float32
    ) + noise

    out = np.clip(
        out,
        0,
        255,
    ).astype(np.uint8)

    result = arr.copy()
    result[mask] = out[mask]

    return result


def desaturate_region(
    arr,
    mask,
    factor,
):
    """
    factor=0:
        完全脱色

    factor=1:
        元の彩度
    """
    rgb = (
        arr.astype(np.float32)
        / 255.0
    )

    gray = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )[..., None]

    out = (
        gray
        + (rgb - gray) * factor
    )

    out = np.clip(
        out * 255,
        0,
        255,
    ).astype(np.uint8)

    result = arr.copy()
    result[mask] = out[mask]

    return result


def apply_degradation(
    image,
    mask,
    mode,
    blur_radius,
    noise_sigma,
    saturation_factor,
    seed,
):
    arr = np.asarray(
        image.convert("RGB")
    ).copy()

    rng = np.random.default_rng(
        seed
    )

    if mode == "blur":
        arr = blur_region(
            arr,
            mask,
            blur_radius,
        )

    elif mode == "noise":
        arr = noise_region(
            arr,
            mask,
            noise_sigma,
            rng,
        )

    elif mode == "desaturate":
        arr = desaturate_region(
            arr,
            mask,
            saturation_factor,
        )

    elif mode == "blur_noise":
        arr = blur_region(
            arr,
            mask,
            blur_radius,
        )

        arr = noise_region(
            arr,
            mask,
            noise_sigma,
            rng,
        )

    elif mode == "blur_desaturate":
        arr = blur_region(
            arr,
            mask,
            blur_radius,
        )

        arr = desaturate_region(
            arr,
            mask,
            saturation_factor,
        )

    elif mode == "noise_desaturate":
        arr = noise_region(
            arr,
            mask,
            noise_sigma,
            rng,
        )

        arr = desaturate_region(
            arr,
            mask,
            saturation_factor,
        )

    elif mode == "all":
        arr = blur_region(
            arr,
            mask,
            blur_radius,
        )

        arr = noise_region(
            arr,
            mask,
            noise_sigma,
            rng,
        )

        arr = desaturate_region(
            arr,
            mask,
            saturation_factor,
        )

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    return Image.fromarray(arr)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        help="元画像",
    )

    parser.add_argument(
        "regions",
        help="region_contributions.csv",
    )

    parser.add_argument(
        "--out",
        default="counterfactual_analysis",
    )

    parser.add_argument(
        "--model-name",
        default="resnet18",
    )

    parser.add_argument(
        "--contribution-column",
        default="importance",
        help=(
            "低寄与判定に使う列。"
            "例: importance または "
            "mean_luminance_relative"
        ),
    )

    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[
            0.01,
            0.03,
            0.05,
            0.10,
        ],
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

    image = Image.open(
        args.image
    ).convert("RGB")

    regions = pd.read_csv(
        args.regions
    )

    if (
        args.contribution_column
        not in regions.columns
    ):
        raise ValueError(
            f"Column not found: "
            f"{args.contribution_column}"
        )

    model, preprocess, categories = (
        load_model(
            args.model_name,
            device,
        )
    )

    # 元画像
    original_probs = predict(
        model,
        preprocess,
        [image],
        device,
        args.batch_size,
    )[0]

    original_class = int(
        np.argmax(original_probs)
    )

    original_probability = float(
        original_probs[
            original_class
        ]
    )

    images = []
    metadata = []

    # 全条件の画像を生成してからbatch inference
    for threshold in args.thresholds:

        mask = make_mask_from_regions(
            regions,
            image.width,
            image.height,
            args.contribution_column,
            threshold,
        )

        mask_image = Image.fromarray(
            np.uint8(mask) * 255
        )

        mask_image.save(
            out / (
                f"mask_{args.contribution_column}"
                f"_{threshold:.4g}.png"
            )
        )

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

            degraded.save(
                out / filename
            )

            images.append(degraded)

            metadata.append({
                "threshold": threshold,
                "mode": mode,
                "filename": filename,
                "masked_fraction": float(
                    mask.mean()
                ),
            })

    probs = predict(
        model,
        preprocess,
        images,
        device,
        args.batch_size,
    )

    rows = []

    for info, probability in zip(
        metadata,
        probs,
    ):
        predicted_class = int(
            np.argmax(probability)
        )

        target_probability = float(
            probability[
                original_class
            ]
        )

        rows.append({
            **info,
            "original_class": original_class,
            "original_class_name": categories[
                original_class
            ],
            "original_probability": (
                original_probability
            ),
            "degraded_probability": (
                target_probability
            ),
            "probability_change": (
                target_probability
                - original_probability
            ),
            "absolute_probability_change": (
                abs(
                    target_probability
                    - original_probability
                )
            ),
            "predicted_class": predicted_class,
            "predicted_class_name": categories[
                predicted_class
            ],
            "top1_changed": (
                predicted_class
                != original_class
            ),
        })

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
            result.to_dict(
                orient="records"
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Summary
    summary = {
        "image": str(
            Path(args.image).resolve()
        ),
        "model": args.model_name,
        "device": str(device),
        "contribution_column": (
            args.contribution_column
        ),
        "thresholds": args.thresholds,
        "modes": args.modes,
        "original_class": original_class,
        "original_class_name": categories[
            original_class
        ],
        "original_probability": (
            original_probability
        ),
        "interpretation": (
            "This is a counterfactual "
            "robustness/necessity test. "
            "It does not by itself prove "
            "causal attribution."
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

    print(
        f"Original: "
        f"{categories[original_class]} "
        f"({original_probability:.6f})"
    )

    print()
    print(result.to_string(
        index=False
    ))

    print()
    print(
        f"Results saved to: "
        f"{out.resolve()}"
    )


if __name__ == "__main__":
    main()
