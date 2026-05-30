"""Leakage-safe feature engineering for IEEE-CIS fraud models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype


@dataclass
class FeaturePipeline:
    """Fit-on-train preprocessing shared by from-scratch models."""

    categorical_columns: list[str] = field(default_factory=list)
    frequency_maps: dict[str, dict[object, float]] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    uid_amount_means: dict[str, float] = field(default_factory=dict)
    uid_amount_stds: dict[str, float] = field(default_factory=dict)
    lr_medians: pd.Series | None = None
    lr_means: pd.Series | None = None
    lr_stds: pd.Series | None = None
    global_amount_mean: float = 0.0
    global_amount_std: float = 1.0
    fitted: bool = False
    feature_profile: str = "baseline"
    uid_aggregate_maps: dict[str, dict[str, float]] = field(default_factory=dict)
    uid_aggregate_defaults: dict[str, float] = field(default_factory=dict)

    FEATURE_PROFILES: ClassVar[set[str]] = {"baseline", "uid_d", "uid_agg"}
    D_NORMALIZED_COLUMNS: ClassVar[tuple[str, ...]] = ("D1", "D2", "D3", "D4", "D10", "D15")
    UID_AGGREGATE_COLUMNS: ClassVar[tuple[str, ...]] = (
        "TransactionAmt",
        "C1",
        "C11",
        "C13",
        "D2",
        "D15",
        "M5",
        "M9",
    )
    CATEGORICAL_IDENTIFIERS: ClassVar[tuple[str, ...]] = (
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "ProductCD",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
        *(f"M{idx}" for idx in range(1, 10)),
        *(f"id_{idx:02d}" for idx in range(12, 39)),
    )
    DROP_COLUMNS: ClassVar[set[str]] = {
        "TransactionID",
        "TransactionDT",
        "Day",
        "UID",
        "Client_Start",
        "isFraud",
    }
    MISSING_TOKEN: ClassVar[str] = "__MISSING__"

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        self._validate_feature_profile()
        base = self._add_base_derived_columns(df)
        self._fit_uid_amount_stats(base)
        if self.feature_profile == "uid_agg":
            self._fit_uid_aggregate_stats(base)
        transformed = self._add_amount_features(base)

        self.categorical_columns = self._detect_categorical_columns(transformed)
        self.frequency_maps = self._fit_frequency_maps(transformed)
        encoded = self._encode_categorical(transformed)

        self.feature_columns = [col for col in encoded.columns if col not in self.DROP_COLUMNS]
        features = self._numeric_feature_frame(encoded)
        self.lr_medians = features.median(axis=0, skipna=True).fillna(0.0)
        filled = features.fillna(self.lr_medians)
        self.lr_means = filled.mean(axis=0).fillna(0.0)
        self.lr_stds = filled.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        self.fitted = True
        return self

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        self._validate_feature_profile()
        base = self._add_base_derived_columns(df)
        transformed = self._add_amount_features(base)
        return self._encode_categorical(transformed)

    def fit_transform(self, df: pd.DataFrame, model: str) -> np.ndarray:
        self._validate_feature_profile()
        self._validate_model(model)
        self.fit(df)
        return self._matrix_from_frame(self.transform_frame(df), model)

    def transform(self, df: pd.DataFrame, model: str) -> np.ndarray:
        self._validate_feature_profile()
        self._validate_model(model)
        return self._matrix_from_frame(self.transform_frame(df), model)

    def transform_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        self._validate_feature_profile()
        base = self._add_base_derived_columns(df, include_profile_keys=True)
        result = pd.DataFrame(index=df.index)
        if "TransactionID" in base.columns:
            result["TransactionID"] = base["TransactionID"].to_numpy()
        else:
            result["TransactionID"] = df.index.to_numpy()
        for column in ["UID_D1", "UID_Email", "UID"]:
            if column in base.columns:
                result[column] = base[column].astype("string").to_numpy()
        return result

    def _add_base_derived_columns(self, df: pd.DataFrame, include_profile_keys: bool = False) -> pd.DataFrame:
        frame = df.copy()
        amount = self._numeric_column(frame, "TransactionAmt")
        transaction_dt = self._numeric_column(frame, "TransactionDT")
        day = np.floor(transaction_dt / 86400.0)

        frame["Amt_Cents"] = np.round((amount - np.floor(amount)) * 100.0)
        frame["Hour"] = np.floor(transaction_dt / 3600.0) % 24.0
        frame["Day"] = day
        frame["DayOfWeek"] = day % 7.0
        frame["Client_Start"] = frame["Day"] - self._numeric_column(frame, "D1")
        frame["card1_addr1"] = self._combine_keys(frame, ["card1", "addr1"])
        frame["UID"] = self._combine_keys(frame, ["card1_addr1", "Client_Start"])
        if self.feature_profile in {"uid_d", "uid_agg"} or include_profile_keys:
            self._add_uid_profile_columns(frame)
        return frame

    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        amount = self._numeric_column(frame, "TransactionAmt")
        uid_keys = self._string_key_series(frame["UID"])
        means = uid_keys.map(self.uid_amount_means).astype("float64").fillna(self.global_amount_mean)
        stds = uid_keys.map(self.uid_amount_stds).astype("float64").fillna(self.global_amount_std)
        stds = stds.replace(0.0, self.global_amount_std)
        stds = stds.replace(0.0, 1.0)

        frame["Amt_Diff"] = amount - means
        frame["Amt_ZScore"] = frame["Amt_Diff"] / stds
        if self.feature_profile == "uid_agg":
            uid_column = "UID_D1" if "UID_D1" in frame.columns else "UID"
            uid_keys = self._string_key_series(frame[uid_column])
            for feature_name, mapping in self.uid_aggregate_maps.items():
                default = self.uid_aggregate_defaults.get(feature_name, 0.0)
                frame[feature_name] = uid_keys.map(mapping).astype("float64").fillna(default)
        return frame

    def _add_uid_profile_columns(self, frame: pd.DataFrame) -> None:
        for d_col in self.D_NORMALIZED_COLUMNS:
            if d_col in frame.columns:
                frame[f"{d_col}n"] = frame["Day"] - self._numeric_column(frame, d_col)

        d1n = frame["D1n"] if "D1n" in frame.columns else frame["Client_Start"]
        frame["UID_D1"] = self._combine_key_series(
            [
                self._string_key_series(self._frame_column_or_missing(frame, "card1")),
                self._string_key_series(self._frame_column_or_missing(frame, "addr1")),
                self._string_key_series(d1n),
            ],
            index=frame.index,
        )
        frame["UID_Email"] = self._combine_key_series(
            [
                self._string_key_series(self._frame_column_or_missing(frame, "card1")),
                self._string_key_series(self._frame_column_or_missing(frame, "card2")),
                self._string_key_series(self._frame_column_or_missing(frame, "addr1")),
                self._string_key_series(self._frame_column_or_missing(frame, "P_emaildomain")),
            ],
            index=frame.index,
        )

    def _fit_uid_amount_stats(self, df: pd.DataFrame) -> None:
        amount = self._numeric_column(df, "TransactionAmt")
        finite_amount = amount[np.isfinite(amount)]
        if finite_amount.empty:
            self.global_amount_mean = 0.0
            self.global_amount_std = 1.0
        else:
            self.global_amount_mean = float(finite_amount.mean())
            std = float(finite_amount.std(ddof=0))
            self.global_amount_std = std if np.isfinite(std) and std > 0.0 else 1.0

        stats_frame = pd.DataFrame({"UID": self._string_key_series(df["UID"]), "TransactionAmt": amount})
        grouped = stats_frame.groupby("UID", dropna=False)["TransactionAmt"]
        means = grouped.mean()
        stds = grouped.std(ddof=0).replace(0.0, self.global_amount_std).fillna(self.global_amount_std)
        self.uid_amount_means = {str(key): float(value) for key, value in means.items() if np.isfinite(value)}
        self.uid_amount_stds = {
            str(key): float(value) if np.isfinite(value) and float(value) > 0.0 else self.global_amount_std
            for key, value in stds.items()
        }

    def _fit_uid_aggregate_stats(self, df: pd.DataFrame) -> None:
        uid_column = "UID_D1" if "UID_D1" in df.columns else "UID"
        uid_keys = self._string_key_series(df[uid_column])
        self.uid_aggregate_maps = {}
        self.uid_aggregate_defaults = {}

        count_series = uid_keys.value_counts(dropna=False).astype("float64")
        self.uid_aggregate_maps["UID_Count"] = {str(key): float(value) for key, value in count_series.items()}
        self.uid_aggregate_defaults["UID_Count"] = 0.0

        for col in self.UID_AGGREGATE_COLUMNS:
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce").astype("float64")
            finite_values = values[np.isfinite(values)]
            default = float(finite_values.mean()) if not finite_values.empty else 0.0
            grouped = pd.DataFrame({"uid": uid_keys, "value": values}).groupby("uid", dropna=False)["value"]
            mean_name = f"UID_{col}_Mean"
            means = grouped.mean()
            self.uid_aggregate_maps[mean_name] = {
                str(key): float(value) for key, value in means.items() if np.isfinite(value)
            }
            self.uid_aggregate_defaults[mean_name] = default
            if col == "TransactionAmt":
                std_name = "UID_TransactionAmt_Std"
                stds = grouped.std(ddof=0).fillna(0.0)
                self.uid_aggregate_maps[std_name] = {
                    str(key): float(value) for key, value in stds.items() if np.isfinite(value)
                }
                self.uid_aggregate_defaults[std_name] = 0.0

    def _detect_categorical_columns(self, df: pd.DataFrame) -> list[str]:
        identifiers = set(self.CATEGORICAL_IDENTIFIERS)
        columns: list[str] = []
        for col in df.columns:
            dtype = df[col].dtype
            if col in identifiers or is_object_dtype(dtype) or is_string_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
                columns.append(col)
        return columns

    def _fit_frequency_maps(self, df: pd.DataFrame) -> dict[str, dict[object, float]]:
        maps: dict[str, dict[object, float]] = {}
        for col in self.categorical_columns:
            counts = self._category_key_series(df[col]).value_counts(dropna=False)
            maps[col] = {key: float(value) for key, value in counts.items()}
        return maps

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        for col in self.categorical_columns:
            if col not in frame.columns:
                frame[col] = np.nan
            values = self._category_key_series(frame[col])
            frame[col] = values.map(self.frequency_maps[col]).fillna(1.0).astype("float64")
        return frame

    def _numeric_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        for col in self.feature_columns:
            if col in df.columns:
                features[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            else:
                features[col] = np.nan
        return features

    def _matrix_from_frame(self, df: pd.DataFrame, model: str) -> np.ndarray:
        features = self._numeric_feature_frame(df)
        if model == "boosting":
            return features.to_numpy(dtype=np.float32, copy=True)

        if model == "lr":
            assert self.lr_medians is not None
            assert self.lr_means is not None
            assert self.lr_stds is not None
            filled = features.fillna(self.lr_medians)
            standardized = (filled - self.lr_means) / self.lr_stds
            values = standardized.to_numpy(dtype=np.float32, copy=True)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            bias = np.ones((values.shape[0], 1), dtype=np.float32)
            return np.concatenate([bias, values], axis=1)

        raise ValueError("model must be 'lr' or 'boosting'")

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise ValueError("FeaturePipeline must be fit before transform")

    def _validate_model(self, model: str) -> None:
        if model not in {"lr", "boosting"}:
            raise ValueError("model must be 'lr' or 'boosting'")

    def _validate_feature_profile(self) -> None:
        if self.feature_profile not in self.FEATURE_PROFILES:
            raise ValueError(f"feature_profile must be one of {sorted(self.FEATURE_PROFILES)}")

    def _numeric_column(self, df: pd.DataFrame, col: str) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").astype("float64")
        return pd.Series(np.nan, index=df.index, dtype="float64")

    def _frame_column_or_missing(self, df: pd.DataFrame, col: str) -> pd.Series:
        if col in df.columns:
            return df[col]
        return pd.Series(self.MISSING_TOKEN, index=df.index, dtype="string")

    def _combine_keys(self, df: pd.DataFrame, cols: list[str]) -> pd.Series:
        parts = []
        for col in cols:
            if col in df.columns:
                parts.append(self._string_key_series(df[col]))
            else:
                parts.append(pd.Series(self.MISSING_TOKEN, index=df.index, dtype="string"))

        combined = parts[0].copy()
        for part in parts[1:]:
            combined = combined + "_" + part
        return combined

    def _combine_key_series(self, parts: list[pd.Series], index: pd.Index) -> pd.Series:
        if not parts:
            return pd.Series(self.MISSING_TOKEN, index=index, dtype="string")
        combined = parts[0].astype("string").fillna(self.MISSING_TOKEN)
        for part in parts[1:]:
            combined = combined + "_" + part.astype("string").fillna(self.MISSING_TOKEN)
        return combined

    def _string_key_series(self, series: pd.Series) -> pd.Series:
        return series.astype("string").fillna(self.MISSING_TOKEN)

    def _category_key_series(self, series: pd.Series) -> pd.Series:
        values = series.astype("object")
        return values.where(pd.notna(values), self.MISSING_TOKEN)
