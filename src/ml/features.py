from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .dataset import LABEL_COLUMNS, OUTCOME_COLUMNS


NON_FEATURE_COLUMNS = {
    "Trade_Date",
    "Entry_Bar",
    "Exit_Time",
    "Signal_Time",
    "Signal_Bar",
    "ML_Schema_Version",
    "Dataset_Source",
    "Backtest_Trade_Source",
}

LEAKAGE_COLUMNS = {
    "Exit_Adaptive_Spread_Pips",
    "Exit_Adaptive_Slippage_Pips",
    "Exit_Spread_Session_Mult",
    "Exit_Spread_Volatility_Mult",
    "Exit_Price",
    "Exit_Bar",
    "Final_Stop",
    "Final_Exit_Type",
    "Runner_Active",
    "Runner_Stop_Price",
    "Runner_Exit_Reason",
    "Partial_At_2R",
    "Partial_Price",
    "Balance_After",
    "Last_Partial_Reason",
    "Last_Partial_Time",
}

RAW_PRICE_LEVEL_COLUMNS = {
    "Entry_Price",
    "Initial_Stop",
    "Take_Profit",
    "Stop_Loss",
    "demand_zone_entry",
    "demand_zone_stop",
    "supply_zone_entry",
    "supply_zone_stop",
    "sr_nearest_resistance",
    "sr_nearest_support",
    "sr_long_reference_level",
    "sr_short_reference_level",
    "sr_long_target",
    "sr_short_target",
    "vp_reference_poc",
    "liquidity_pool_high",
    "liquidity_pool_low",
}


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def add_trade_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    atr = pd.to_numeric(out["atr_feature"], errors="coerce") if "atr_feature" in out.columns else None
    risk_pips = pd.to_numeric(out["Risk_Pips"], errors="coerce") if "Risk_Pips" in out.columns else None

    #Building compact continuous features from trade quality context
    if "Confluence_Count" in out.columns:
        out["feat_confluence_ratio"] = pd.to_numeric(out["Confluence_Count"], errors="coerce") / 9.0

    #Normalizing entry costs relative to risk size
    if risk_pips is not None:
        entry_spread = pd.to_numeric(out.get("Entry_Adaptive_Spread_Pips"), errors="coerce")
        entry_slippage = pd.to_numeric(out.get("Entry_Adaptive_Slippage_Pips"), errors="coerce")
        entry_cost = entry_spread.fillna(0.0) + entry_slippage.fillna(0.0)
        out["feat_entry_cost_pips"] = entry_cost
        out["feat_entry_cost_to_risk"] = _safe_divide(entry_cost, risk_pips)

    if "SR_Runner_Target_RR" in out.columns:
        out["feat_runner_rr_buffer"] = pd.to_numeric(out["SR_Runner_Target_RR"], errors="coerce") - 2.0

    #Adding long-side quality features from zone, SR, POC, and liquidity context
    if {"demand_zone_entry", "demand_zone_stop"}.issubset(out.columns):
        long_risk_price = pd.to_numeric(out["demand_zone_entry"], errors="coerce") - pd.to_numeric(
            out["demand_zone_stop"], errors="coerce"
        )
        out["feat_long_risk_price"] = long_risk_price
        if atr is not None:
            out["feat_long_risk_atr"] = _safe_divide(long_risk_price, atr)
        if {"sr_nearest_resistance", "demand_zone_entry"}.issubset(out.columns):
            res_dist = pd.to_numeric(out["sr_nearest_resistance"], errors="coerce") - pd.to_numeric(
                out["demand_zone_entry"], errors="coerce"
            )
            out["feat_long_rr_to_sr"] = _safe_divide(res_dist, long_risk_price)
        if atr is not None and {"vp_reference_poc", "demand_zone_entry"}.issubset(out.columns):
            poc_dist = (
                pd.to_numeric(out["vp_reference_poc"], errors="coerce")
                - pd.to_numeric(out["demand_zone_entry"], errors="coerce")
            ).abs()
            out["feat_long_poc_distance_atr"] = _safe_divide(poc_dist, atr)
        if atr is not None and {"liquidity_pool_high", "demand_zone_entry"}.issubset(out.columns):
            liq_dist = pd.to_numeric(out["liquidity_pool_high"], errors="coerce") - pd.to_numeric(
                out["demand_zone_entry"], errors="coerce"
            )
            out["feat_long_liquidity_distance_atr"] = _safe_divide(liq_dist, atr)

    #Adding short-side quality features from zone, SR, POC, and liquidity context
    if {"supply_zone_entry", "supply_zone_stop"}.issubset(out.columns):
        short_risk_price = pd.to_numeric(out["supply_zone_stop"], errors="coerce") - pd.to_numeric(
            out["supply_zone_entry"], errors="coerce"
        )
        out["feat_short_risk_price"] = short_risk_price
        if atr is not None:
            out["feat_short_risk_atr"] = _safe_divide(short_risk_price, atr)
        if {"sr_nearest_support", "supply_zone_entry"}.issubset(out.columns):
            sup_dist = pd.to_numeric(out["supply_zone_entry"], errors="coerce") - pd.to_numeric(
                out["sr_nearest_support"], errors="coerce"
            )
            out["feat_short_rr_to_sr"] = _safe_divide(sup_dist, short_risk_price)
        if atr is not None and {"vp_reference_poc", "supply_zone_entry"}.issubset(out.columns):
            poc_dist = (
                pd.to_numeric(out["vp_reference_poc"], errors="coerce")
                - pd.to_numeric(out["supply_zone_entry"], errors="coerce")
            ).abs()
            out["feat_short_poc_distance_atr"] = _safe_divide(poc_dist, atr)
        if atr is not None and {"liquidity_pool_low", "supply_zone_entry"}.issubset(out.columns):
            liq_dist = pd.to_numeric(out["supply_zone_entry"], errors="coerce") - pd.to_numeric(
                out["liquidity_pool_low"], errors="coerce"
            )
            out["feat_short_liquidity_distance_atr"] = _safe_divide(liq_dist, atr)

    return out.replace([np.inf, -np.inf], np.nan)


def build_sample_weights(df: pd.DataFrame) -> np.ndarray:
    pnl_r = pd.to_numeric(df["PnL_R"], errors="coerce").fillna(0.0).abs().clip(lower=0.0, upper=5.0)
    return np.asarray(1.0 + pnl_r, dtype=float)


def _base_feature_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    #Dropping labels, outcomes, metadata, and leakage-prone raw price levels
    drop_cols = set(LABEL_COLUMNS)
    drop_cols.update(OUTCOME_COLUMNS)
    drop_cols.update(NON_FEATURE_COLUMNS)
    drop_cols.update(LEAKAGE_COLUMNS)
    drop_cols.update(RAW_PRICE_LEVEL_COLUMNS)
    drop_cols.add(target)

    X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore").copy()
    for col in X.columns:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype("Int64")
    return X.replace([np.inf, -np.inf], np.nan)


def vectorize_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #Building aligned one-hot encoded feature matrices for each split
    train_features = _base_feature_frame(train_df, target)
    valid_features = _base_feature_frame(valid_df, target)
    test_features = _base_feature_frame(test_df, target)

    X_train = pd.get_dummies(train_features, dummy_na=True)
    X_valid = pd.get_dummies(valid_features, dummy_na=True).reindex(columns=X_train.columns, fill_value=0)
    X_test = pd.get_dummies(test_features, dummy_na=True).reindex(columns=X_train.columns, fill_value=0)

    #Dropping columns that are entirely missing in the training split
    all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
    if all_nan_cols:
        print(f"[ML] Dropping all-NaN train feature columns: {all_nan_cols}")
        X_train = X_train.drop(columns=all_nan_cols)
        X_valid = X_valid.drop(columns=all_nan_cols)
        X_test = X_test.drop(columns=all_nan_cols)

    return X_train, X_valid, X_test


def fit_imputer(train_frame: pd.DataFrame) -> SimpleImputer:
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    imputer.fit(train_frame)
    return imputer


def apply_imputer(imputer: SimpleImputer, frame: pd.DataFrame) -> pd.DataFrame:
    values = imputer.transform(frame)
    return pd.DataFrame(values, columns=frame.columns, index=frame.index)
