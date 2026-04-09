from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ML_MODELS_DIR = REPO_ROOT / "outputs" / "ml_models"
IMAGES_DIR = REPO_ROOT / "dissertation_latex" / "images"
OUTPUT_DIR = ML_MODELS_DIR / "analysis"

RUN_PATTERN = re.compile(r"^(?P<key>.+)_walk_forward_(?P<stamp>\d{8}T\d{6}Z)$")
LABEL_COL = "Label_PositivePnL"
RANK_COL = "rank_pct"
SELECTED_COL = "selected"
REPRESENTATIVE_KEY = "GBPUSD_M15"


@dataclass(slots=True)
class SplitMetrics:
    roc_auc: float | None
    average_precision: float | None
    precision: float
    recall: float
    f1: float
    accuracy: float
    tn: int
    fp: int
    fn: int
    tp: int
    rows: int
    selected_rate: float
    positive_rate: float


def _roc_curve_points(y_true: pd.Series, y_score: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    true = y_true.to_numpy(dtype=int)
    score = y_score.to_numpy(dtype=float)
    order = np.argsort(-score, kind="mergesort")
    true = true[order]
    score = score[order]

    distinct = np.where(np.diff(score))[0]
    threshold_idxs = np.r_[distinct, true.size - 1]

    tps = np.cumsum(true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    total_pos = true.sum()
    total_neg = true.size - total_pos
    if total_pos == 0 or total_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    tpr = np.r_[0.0, tps / total_pos, 1.0]
    fpr = np.r_[0.0, fps / total_neg, 1.0]
    return fpr, tpr


def _roc_auc(y_true: pd.Series, y_score: pd.Series) -> float | None:
    true = y_true.to_numpy(dtype=int)
    score = y_score.to_numpy(dtype=float)
    pos = score[true == 1]
    neg = score[true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    combined = np.concatenate([pos, neg])
    ranks = pd.Series(combined).rank(method="average").to_numpy()
    pos_ranks = ranks[: len(pos)]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def _average_precision(y_true: pd.Series, y_score: pd.Series) -> float | None:
    true = y_true.to_numpy(dtype=int)
    score = y_score.to_numpy(dtype=float)
    total_pos = true.sum()
    if total_pos == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    true = true[order]
    tp = np.cumsum(true)
    fp = np.cumsum(1 - true)
    precision = tp / (tp + fp)
    recall = tp / total_pos
    recall_prev = np.r_[0.0, recall[:-1]]
    ap = np.sum((recall - recall_prev) * precision)
    return float(ap)


def _latest_run_dirs() -> dict[str, Path]:
    latest: dict[str, tuple[str, Path]] = {}
    for path in ML_MODELS_DIR.iterdir():
        if not path.is_dir():
            continue
        match = RUN_PATTERN.match(path.name)
        if not match:
            continue
        key = match.group("key")
        stamp = match.group("stamp")
        current = latest.get(key)
        if current is None or stamp > current[0]:
            latest[key] = (stamp, path)
    return {key: item[1] for key, item in sorted(latest.items())}


def _load_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(run_dir / f"{split}_predictions.csv")
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int)
    df[SELECTED_COL] = pd.to_numeric(df[SELECTED_COL], errors="coerce").fillna(0).astype(int)
    # rank_pct is monotonic with the original XGBoost score, so 1-rank_pct is a valid
    # continuous ranking score for AUC-style analysis.
    df["ranking_score"] = 1.0 - pd.to_numeric(df[RANK_COL], errors="coerce").fillna(0.0)
    return df


def _compute_split_metrics(df: pd.DataFrame) -> SplitMetrics:
    y_true = df[LABEL_COL].astype(int)
    y_pred = df[SELECTED_COL].astype(int)
    y_score = df["ranking_score"].astype(float)

    unique_classes = y_true.nunique(dropna=True)
    roc_auc = _roc_auc(y_true, y_score) if unique_classes > 1 else None
    average_precision = _average_precision(y_true, y_score) if unique_classes > 1 else None

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return SplitMetrics(
        roc_auc=None if roc_auc is None else float(roc_auc),
        average_precision=None if average_precision is None else float(average_precision),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        accuracy=float((y_true == y_pred).mean()),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        rows=int(len(df)),
        selected_rate=float(y_pred.mean()),
        positive_rate=float(y_true.mean()),
    )


def _metrics_row(run_key: str, split: str, metrics: SplitMetrics) -> dict[str, object]:
    return {
        "run": run_key,
        "split": split,
        **asdict(metrics),
    }


def _aggregate_metrics(run_dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    per_run_rows: list[dict[str, object]] = []
    split_frames: dict[str, list[pd.DataFrame]] = {"validation": [], "test": []}

    for run_key, run_dir in run_dirs.items():
        for split in ("validation", "test"):
            df = _load_predictions(run_dir, split)
            split_frames[split].append(df)
            per_run_rows.append(_metrics_row(run_key, split, _compute_split_metrics(df)))

    per_run_df = pd.DataFrame(per_run_rows).sort_values(["run", "split"]).reset_index(drop=True)

    aggregate: dict[str, dict[str, object]] = {}
    aggregate_rows: list[dict[str, object]] = []
    for split, frames in split_frames.items():
        combined = pd.concat(frames, ignore_index=True)
        metrics = _compute_split_metrics(combined)
        summary = {
            "micro": asdict(metrics),
            "macro_mean": (
                per_run_df[per_run_df["split"] == split][
                    [
                        "roc_auc",
                        "average_precision",
                        "precision",
                        "recall",
                        "f1",
                        "accuracy",
                        "selected_rate",
                        "positive_rate",
                    ]
                ]
                .mean(numeric_only=True)
                .to_dict()
            ),
        }
        aggregate[split] = summary
        aggregate_rows.append({"run": "AGGREGATE", "split": split, **summary["micro"]})

    aggregate_df = pd.DataFrame(aggregate_rows)
    return per_run_df, aggregate_df, aggregate


def _plot_representative_validation(run_dir: Path, run_key: str) -> Path:
    validation_df = _load_predictions(run_dir, "validation")
    test_df = _load_predictions(run_dir, "test")
    importance_df = pd.read_csv(run_dir / "feature_importance.csv").head(10).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for ax, df, label in (
        (axes[0], validation_df, "Validation"),
        (axes[0], test_df, "Test"),
    ):
        fpr, tpr = _roc_curve_points(df[LABEL_COL], df["ranking_score"])
        auc = _roc_auc(df[LABEL_COL], df["ranking_score"])
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} AUC = {auc:.3f}")

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curves: {run_key.replace('_', ' ')}")
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].barh(importance_df["feature"], importance_df["importance"], color="#4C78A8")
    axes[1].set_xlabel("Importance")
    axes[1].set_title("Top Ten Features")

    fig.tight_layout()
    output_path = IMAGES_DIR / f"ml_model_validation_{run_key.lower()}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    run_dirs = _latest_run_dirs()
    per_run_df, aggregate_df, aggregate = _aggregate_metrics(run_dirs)

    per_run_path = OUTPUT_DIR / "classification_metrics_per_run.csv"
    aggregate_path = OUTPUT_DIR / "classification_metrics_aggregate.csv"
    aggregate_json_path = OUTPUT_DIR / "classification_metrics_aggregate.json"

    per_run_df.to_csv(per_run_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)
    with aggregate_json_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    representative_dir = run_dirs[REPRESENTATIVE_KEY]
    figure_path = _plot_representative_validation(representative_dir, REPRESENTATIVE_KEY)

    print(f"Saved per-run metrics to: {per_run_path}")
    print(f"Saved aggregate metrics to: {aggregate_path}")
    print(f"Saved aggregate JSON to: {aggregate_json_path}")
    print(f"Saved representative figure to: {figure_path}")
    print()
    print(aggregate_df.to_string(index=False))


if __name__ == "__main__":
    main()
