from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.data import (
    build_ml_dataset_labels,
    build_ml_run_labels,
    delete_dataset_file,
    flatten_metrics,
    list_csv_files,
    list_run_directories,
    load_ml_bundle,
    load_ml_dataset_summary,
)
from dashboard.ui import (
    dashboard_panel,
    render_key_value_table,
    render_metric_grid,
    render_metric_row,
    render_page_header,
)
from src.ml.ranking import trading_metrics
from src.mt5.config import ML_DATASET_OUTPUT_DIR, OUTPUTS_DIR


@st.cache_data(show_spinner=False)
def load_ml_bundle_cached(run_dir_str: str) -> dict:
    return load_ml_bundle(Path(run_dir_str))


def _build_ml_output_dir(dataset_path: Path) -> Path:
    # Creating a unique output folder for each dashboard-launched ML run
    dataset_label = build_ml_dataset_labels((str(dataset_path),)).get(str(dataset_path), dataset_path.stem.removesuffix("_ml_dataset"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return OUTPUTS_DIR / "ml_models" / f"{dataset_label}_walk_forward_{stamp}"


def _run_ml_training(project_root: Path, dataset_path: Path) -> dict[str, object]:
    # Launching the existing ML training entrypoint against one selected dataset
    output_dir = _build_ml_output_dir(dataset_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_ml",
            "--data-dir",
            str(ML_DATASET_OUTPUT_DIR),
            "--pattern",
            dataset_path.name,
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(project_root),
        capture_output=True,
        text=True,
    )

    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_dir": str(output_dir),
        "dataset_name": dataset_path.name,
    }


def _render_ml_training_block(project_root: Path) -> None:
    # Letting the dashboard launch a new ML run from one saved trade dataset
    dataset_files = list_csv_files(ML_DATASET_OUTPUT_DIR)

    with dashboard_panel(
        "Train On Trade Dataset",
        subtitle="Choose one saved trade dataset export and launch a new XGBoost training run for it.",
        eyebrow="Training",
    ):
        if not dataset_files:
            st.info("No saved trade datasets are available yet. Run a backtest first to export one.")
            return

        dataset_labels = build_ml_dataset_labels(tuple(str(path) for path in dataset_files))
        selector_left, selector_right = st.columns((12, 1))
        with selector_left:
            selected_dataset = st.selectbox(
                "Trade Dataset",
                options=dataset_files,
                format_func=lambda path: dataset_labels.get(str(path), Path(path).name),
                key="dashboard_ml_dataset_path",
            )
        with selector_right:
            st.markdown("<div class='dashboard-delete-spacer'></div>", unsafe_allow_html=True)
            if st.button(
                "x",
                key="dashboard_ml_delete_dataset",
                width="stretch",
                help=f"Delete {dataset_labels.get(str(selected_dataset), Path(selected_dataset).name)}",
            ):
                try:
                    delete_dataset_file(Path(selected_dataset))
                    list_csv_files.clear()
                    load_ml_dataset_summary.clear()
                    st.session_state.pop("dashboard_ml_dataset_path", None)
                    st.session_state["dashboard_ml_notice"] = (
                        f"Deleted dataset {dataset_labels.get(str(selected_dataset), Path(selected_dataset).name)}."
                    )
                    st.rerun()
                except PermissionError:
                    st.error("That dataset is currently in use by Windows or OneDrive. Close anything using it and try again.")
                except OSError as exc:
                    st.error(f"Unable to delete the selected dataset: {exc}")

        summary = load_ml_dataset_summary(Path(selected_dataset))
        render_metric_row(
            [
                ("Rows", summary.get("rows")),
                ("Symbol", summary.get("symbol")),
                ("Timeframe", summary.get("timeframe")),
                ("Date Range", summary.get("date_range")),
            ]
        )

        button_left, button_right = st.columns((4, 1))
        with button_right:
            if st.button("Train Model", key="dashboard_ml_train_button", width="stretch"):
                with st.spinner("Training XGBoost model on the selected trade dataset..."):
                    result = _run_ml_training(project_root, Path(selected_dataset))

                st.session_state["dashboard_ml_train_result"] = result
                if int(result["returncode"]) == 0:
                    list_run_directories.clear()
                    load_ml_bundle_cached.clear()
                    st.session_state["dashboard_selected_ml_run"] = str(result["output_dir"])
                    st.session_state["dashboard_ml_notice"] = (
                        f"ML training completed for {dataset_labels.get(str(selected_dataset), Path(selected_dataset).name)}."
                    )
                    st.rerun()

        train_result = st.session_state.get("dashboard_ml_train_result")
        if isinstance(train_result, dict) and int(train_result.get("returncode", 0)) != 0:
            st.error("ML training failed. Review the captured output below.")
            if train_result.get("stdout"):
                st.code(str(train_result["stdout"]), language="text")
            if train_result.get("stderr"):
                st.code(str(train_result["stderr"]), language="text")


def _selected_predictions(df: pd.DataFrame) -> pd.DataFrame:
    # Keeping only the rows accepted by the saved ML selection
    if df.empty or "selected" not in df.columns:
        return df.iloc[0:0].copy()

    selected = pd.to_numeric(df["selected"], errors="coerce").fillna(0.0)
    return df.loc[selected > 0].copy()


def _selection_method_label(value: object) -> object:
    # Converting saved selection-method ids into a cleaner dashboard label
    text = str(value or "").strip()
    if not text:
        return value
    if text == "walk_forward_top_x_percent":
        return "Top X Percent Within Each Walk-Forward Fold"
    return text.replace("_", " ").title()


def render_ml_page(selected_ml_run: Path | None, project_root: Path) -> None:
    #Showing ML filtering performance and saved run artifacts
    render_page_header(
        current_page="ML Analysis",
        title="ML Analysis",
        subtitle="Evaluates an XGBoost model trained on trade-level data. Displays walk-forward validation and test trade metrics, plus a detailed training summary and dataset-driven training controls.",
    )

    notice = st.session_state.pop("dashboard_ml_notice", None)
    if notice:
        st.success(str(notice))

    _render_ml_training_block(project_root)

    if selected_ml_run is None:
        st.info("No saved ML training run is available yet.")
        return

    bundle = load_ml_bundle_cached(str(selected_ml_run))
    ml_runs = list_run_directories(OUTPUTS_DIR / "ml_models")
    ml_run_labels = build_ml_run_labels(tuple(str(path) for path in ml_runs))
    selected_ml_run_label = ml_run_labels.get(str(selected_ml_run), selected_ml_run.name)
    summary = bundle["summary"]
    flat_summary = flatten_metrics(summary)
    validation_predictions = bundle["validation_predictions"]
    test_predictions = bundle["test_predictions"]
    bundle_metadata = bundle["bundle_metadata"]
    dataset_overview = summary.get("dataset_overview", {}) if isinstance(summary, dict) else {}
    fold_summaries = summary.get("folds", []) if isinstance(summary, dict) else []

    validation_kept = int(pd.to_numeric(validation_predictions.get("selected", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not validation_predictions.empty else 0
    test_kept = int(pd.to_numeric(test_predictions.get("selected", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not test_predictions.empty else 0
    validation_baseline_metrics = trading_metrics(validation_predictions) if not validation_predictions.empty else {}
    validation_filtered_metrics = trading_metrics(_selected_predictions(validation_predictions)) if not validation_predictions.empty else {}
    test_baseline_metrics = trading_metrics(test_predictions) if not test_predictions.empty else {}
    test_filtered_metrics = trading_metrics(_selected_predictions(test_predictions)) if not test_predictions.empty else {}

    left, right = st.columns(2)
    with left:
        with dashboard_panel(
            "Validation Metrics",
            subtitle="Comparison between baseline and filtered trades on the validation slice.",
            eyebrow="Validation",
        ):
            render_metric_grid(
                [
                    ("Baseline PnL", validation_baseline_metrics.get("pnl")),
                    ("Filtered PnL", validation_filtered_metrics.get("pnl")),
                    ("Baseline Profit Factor", validation_baseline_metrics.get("profit_factor")),
                    ("Filtered Profit Factor", validation_filtered_metrics.get("profit_factor")),
                    ("Baseline Win Rate", validation_baseline_metrics.get("win_rate")),
                    ("Filtered Win Rate", validation_filtered_metrics.get("win_rate")),
                    ("Trades Kept", f"{validation_kept} / {len(validation_predictions)}"),
                    ("Walk-Forward Folds", flat_summary.get("walk_forward_folds", bundle_metadata.get("walk_forward_folds"))),
                ],
                columns=2,
            )

    with right:
        with dashboard_panel(
            "Test Metrics",
            subtitle="Out-of-sample performance of the filtered trade subset against the full baseline.",
            eyebrow="Test",
        ):
            render_metric_grid(
                [
                    ("Baseline PnL", test_baseline_metrics.get("pnl")),
                    ("Filtered PnL", test_filtered_metrics.get("pnl")),
                    ("Baseline Profit Factor", test_baseline_metrics.get("profit_factor")),
                    ("Filtered Profit Factor", test_filtered_metrics.get("profit_factor")),
                    ("Baseline Win Rate", test_baseline_metrics.get("win_rate")),
                    ("Filtered Win Rate", test_filtered_metrics.get("win_rate")),
                    ("Trades Kept", f"{test_kept} / {len(test_predictions)}"),
                    ("Selection Method", _selection_method_label(bundle_metadata.get("selection_method"))),
                ],
                columns=2,
            )

    with dashboard_panel(
        "Training Summary",
        subtitle="High-level run metadata and saved summary values for the selected ML training run.",
        eyebrow="Summary",
        action=selected_ml_run_label,
    ):
        render_metric_row(
            [
                ("Model Run", selected_ml_run_label),
                ("Target Label", bundle_metadata.get("target")),
                ("Keep Ratio", flat_summary.get("selected_keep_ratio", bundle_metadata.get("keep_ratio"))),
                ("Feature Count", flat_summary.get("feature_count")),
            ]
        )
        left, right = st.columns(2)
        with left:
            render_key_value_table(
                [
                    ("Evaluation Mode", flat_summary.get("evaluation_mode")),
                    ("Walk-Forward Folds", flat_summary.get("walk_forward_folds", bundle_metadata.get("walk_forward_folds"))),
                    ("Initial Train Ratio", flat_summary.get("initial_train_ratio")),
                    ("Validation Trades Kept", f"{validation_kept} / {len(validation_predictions)}"),
                    ("Test Trades Kept", f"{test_kept} / {len(test_predictions)}"),
                    ("Selected Output Folder", str(selected_ml_run)),
                ]
            )

        with right:
            target_distribution = dataset_overview.get("target_distribution", {})
            if isinstance(target_distribution, dict):
                positive_count = target_distribution.get("1", target_distribution.get(1, 0))
                negative_count = target_distribution.get("0", target_distribution.get(0, 0))
            else:
                positive_count = 0
                negative_count = 0

            render_key_value_table(
                [
                    ("Dataset Rows", dataset_overview.get("rows")),
                    ("Dataset Start", dataset_overview.get("date_start")),
                    ("Dataset End", dataset_overview.get("date_end")),
                    ("Positive Labels", positive_count),
                    ("Negative Labels", negative_count),
                    ("Symbol / Timeframe Mix", len(dataset_overview.get("by_symbol_timeframe", []))),
                ]
            )

        if fold_summaries:
            st.markdown("##### Fold Coverage")
            fold_rows = []
            for fold in fold_summaries:
                fold_rows.append(
                    {
                        "fold": fold.get("fold"),
                        "train_rows": fold.get("train_rows"),
                        "valid_rows": fold.get("valid_rows"),
                        "test_rows": fold.get("test_rows"),
                        "train_window": f"{fold.get('train_start')} to {fold.get('train_end')}",
                        "valid_window": f"{fold.get('valid_start')} to {fold.get('valid_end')}",
                        "test_window": f"{fold.get('test_start')} to {fold.get('test_end')}",
                    }
                )
            st.dataframe(pd.DataFrame(fold_rows), width="stretch", hide_index=True)
