from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import spearmanr


FEATURES = [
    "mean_luminance",
    "mean_hue",
    "mean_saturation",
    "edge_density",
    "texture_std",
    "high_frequency_energy",
]


def standardize(df, columns):
    out = df.copy()

    for c in columns:
        mean = out[c].mean()
        std = out[c].std()

        if std == 0 or not np.isfinite(std):
            out[c] = 0.0
        else:
            out[c] = (out[c] - mean) / std

    return out


def pairwise_correlations(df):
    rows = []

    for feature in FEATURES:
        x = df[feature]
        y = df["importance"]

        valid = x.notna() & y.notna()

        if valid.sum() < 3:
            continue

        rho, p = spearmanr(
            x[valid],
            y[valid],
        )

        rows.append({
            "feature": feature,
            "spearman_rho": float(rho),
            "p_value": float(p),
            "n": int(valid.sum()),
        })

    return pd.DataFrame(rows)


def feature_feature_correlations(df):
    rows = []

    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1:]:
            valid = (
                df[a].notna()
                & df[b].notna()
            )

            if valid.sum() < 3:
                continue

            rho, p = spearmanr(
                df.loc[valid, a],
                df.loc[valid, b],
            )

            rows.append({
                "feature_a": a,
                "feature_b": b,
                "spearman_rho": float(rho),
                "p_value": float(p),
                "n": int(valid.sum()),
            })

    return pd.DataFrame(rows)


def fit_main_effect_model(df):
    """
    主効果モデル。

    importance ~ features + depth + log(area)
    """
    work = df.copy()

    work["log_area"] = np.log(
        work["area"].clip(lower=1)
    )

    formula = (
        "importance ~ "
        + " + ".join(FEATURES)
        + " + C(depth)"
        + " + log_area"
    )

    model = smf.ols(
        formula,
        data=work,
    ).fit()

    return model


def fit_interaction_models(df):
    """
    全特徴量ペアについて、

        importance ~ A + B + A:B + depth + log(area)

    を個別に推定する。

    全ペアを一つの巨大なモデルに入れないことで、
    多重共線性による不安定化をある程度避ける。
    """
    work = df.copy()

    work["log_area"] = np.log(
        work["area"].clip(lower=1)
    )

    rows = []

    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1:]:

            formula = (
                f"importance ~ "
                f"{a} + {b} + "
                f"{a}:{b} + "
                f"C(depth) + log_area"
            )

            try:
                model = smf.ols(
                    formula,
                    data=work,
                ).fit()

                term = f"{a}:{b}"

                rows.append({
                    "feature_a": a,
                    "feature_b": b,
                    "interaction": term,
                    "coefficient": float(
                        model.params.get(
                            term,
                            np.nan,
                        )
                    ),
                    "p_value": float(
                        model.pvalues.get(
                            term,
                            np.nan,
                        )
                    ),
                    "ci_low": float(
                        model.conf_int()
                        .loc[term, 0]
                    ),
                    "ci_high": float(
                        model.conf_int()
                        .loc[term, 1]
                    ),
                    "r_squared": float(
                        model.rsquared
                    ),
                    "adjusted_r_squared": float(
                        model.rsquared_adj
                    ),
                    "n": int(
                        model.nobs
                    ),
                })

            except Exception as e:
                rows.append({
                    "feature_a": a,
                    "feature_b": b,
                    "interaction": f"{a}:{b}",
                    "coefficient": np.nan,
                    "p_value": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "r_squared": np.nan,
                    "adjusted_r_squared": np.nan,
                    "n": 0,
                    "error": str(e),
                })

    return pd.DataFrame(rows)


def calculate_relative_contributions(df):
    """
    標準化された特徴量について、主効果モデルの
    |beta * z(feature)| を相対寄与量として計算する。

    これは因果的な「寄与率」ではなく、
    この線形近似モデルにおける局所的な寄与指標。
    """
    work = standardize(
        df,
        FEATURES,
    )

    model = fit_main_effect_model(
        standardize(
            df,
            FEATURES,
        )
    )

    contributions = {}

    for feature in FEATURES:
        beta = model.params.get(
            feature,
            0.0,
        )

        contributions[feature] = (
            np.abs(
                beta * work[feature]
            )
        )

    contrib = pd.DataFrame(
        contributions,
        index=df.index,
    )

    total = contrib.sum(
        axis=1
    )

    for feature in FEATURES:
        contrib[
            f"{feature}_relative"
        ] = np.where(
            total > 0,
            contrib[feature] / total,
            0,
        )

    return model, contrib


def save_heatmaps(df, out):
    """
    四分木領域の中心点へ統計量を配置する簡易ヒートマップ。
    """
    if df.empty:
        return

    x = (
        (df["x0"] + df["x1"]) / 2
    )
    y = (
        (df["y0"] + df["y1"]) / 2
    )

    width = int(
        max(
            df["x1"].max(),
            1,
        )
    )
    height = int(
        max(
            df["y1"].max(),
            1,
        )
    )

    for feature in FEATURES:
        column = f"{feature}_relative"

        if column not in df:
            continue

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

        for idx, row in df.iterrows():

            x0 = int(row["x0"])
            x1 = int(row["x1"])
            y0 = int(row["y0"])
            y1 = int(row["y1"])
            depth = int(row["depth"])

            value = float(
                row[column]
            )

            region_depth = depth_map[
                y0:y1,
                x0:x1
            ]

            region_heat = heat[
                y0:y1,
                x0:x1
            ]

            mask = depth >= region_depth

            region_heat[mask] = value
            region_depth[mask] = depth

        plt.figure(
            figsize=(8, 6)
        )

        plt.imshow(
            heat,
            cmap="viridis",
            vmin=0,
            vmax=np.nanmax(heat)
            if np.isfinite(heat).any()
            else 1,
        )

        plt.colorbar(
            label="relative contribution"
        )

        plt.title(
            f"Relative contribution: {feature}"
        )

        plt.axis("off")
        plt.tight_layout()

        plt.savefig(
            out / (
                f"contribution_{feature}.png"
            ),
            dpi=200,
        )

        plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "regions",
        help="regions.csv",
    )

    parser.add_argument(
        "--out",
        default="interaction_analysis",
    )

    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        args.regions
    )

    required = (
        FEATURES
        + [
            "importance",
            "depth",
            "area",
            "x0",
            "x1",
            "y0",
            "y1",
        ]
    )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    # 主効果
    main_corr = pairwise_correlations(df)
    main_corr.to_csv(
        out / "feature_importance_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 特徴量同士の相関
    feature_corr = feature_feature_correlations(df)
    feature_corr.to_csv(
        out / "feature_feature_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 主効果モデル
    model, contributions = (
        calculate_relative_contributions(df)
    )

    with open(
        out / "main_effect_model.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            model.summary().as_text()
        )

    # interaction
    interaction = fit_interaction_models(
        df
    )

    interaction.to_csv(
        out / "feature_interactions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # contribution table
    contribution_df = df.copy()

    for feature in FEATURES:
        contribution_df[
            f"{feature}_contribution"
        ] = contributions[feature]

        contribution_df[
            f"{feature}_relative"
        ] = contributions[
            f"{feature}_relative"
        ]

    contribution_df.to_csv(
        out / "region_contributions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # interaction heatmap
    matrix = pd.DataFrame(
        np.nan,
        index=FEATURES,
        columns=FEATURES,
    )

    for _, row in interaction.iterrows():
        a = row["feature_a"]
        b = row["feature_b"]
        value = row["coefficient"]

        matrix.loc[a, b] = value
        matrix.loc[b, a] = value

    np.fill_diagonal(
        matrix.values,
        0.0,
    )

    matrix.to_csv(
        out / "interaction_matrix.csv",
        encoding="utf-8-sig",
    )

    # plots
    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        main_corr["feature"],
        main_corr["spearman_rho"],
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.ylabel(
        "Spearman correlation with importance"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        out / "feature_importance_correlation.png",
        dpi=200,
    )

    plt.close()

    plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(
        matrix.values,
        cmap="coolwarm",
        aspect="auto",
    )

    plt.xticks(
        range(len(FEATURES)),
        FEATURES,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(FEATURES)),
        FEATURES,
    )

    plt.colorbar(
        label="interaction coefficient"
    )

    plt.title(
        "Feature interaction coefficients"
    )

    plt.tight_layout()

    plt.savefig(
        out / "feature_interaction_matrix.png",
        dpi=200,
    )

    plt.close()

    # JSON summary
    summary = {
        "features": FEATURES,
        "n_regions": int(len(df)),
        "main_effects": (
            main_corr.to_dict(
                orient="records"
            )
        ),
        "interactions": (
            interaction.to_dict(
                orient="records"
            )
        ),
        "interpretation_note": (
            "Coefficients quantify association in "
            "the specified statistical model. "
            "They should not be interpreted as "
            "causal feature attribution without "
            "controlled intervention experiments."
        ),
    }

    with open(
        out / "analysis_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    save_heatmaps(
        contribution_df,
        out,
    )

    print(
        f"Analysis complete: {out.resolve()}"
    )


if __name__ == "__main__":
    main()
