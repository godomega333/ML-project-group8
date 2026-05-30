"""Canonical model configuration registry for final experiment closure."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


class ImmutableParams(dict[str, Any]):
    """Dict-compatible immutable params for JSON/copy-friendly configs."""

    def __init__(
        self,
        source: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        data = dict(source or {})
        data.update(kwargs)
        super().__init__(copy.deepcopy(data))

    def __copy__(self) -> "ImmutableParams":
        return ImmutableParams(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> "ImmutableParams":
        return ImmutableParams(copy.deepcopy(dict(self), memo))

    def copy(self) -> dict[str, Any]:  # type: ignore[override]
        return dict(self)

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("ImmutableParams cannot be mutated")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


@dataclass(frozen=True)
class ModelConfig:
    config_id: str
    model_family: str
    feature_profile: str
    params: Mapping[str, Any]
    class_weight_alpha: float | None = None
    positive_weight: float | str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", ImmutableParams(self.params))

    def __deepcopy__(self, memo: dict[int, Any]) -> "ModelConfig":
        del memo
        return ModelConfig(
            config_id=self.config_id,
            model_family=self.model_family,
            feature_profile=self.feature_profile,
            params=dict(self.params),
            class_weight_alpha=self.class_weight_alpha,
            positive_weight=self.positive_weight,
            notes=self.notes,
        )

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "config_id": self.config_id,
            "model_family": self.model_family,
            "feature_profile": self.feature_profile,
            "params": dict(self.params),
            "class_weight_alpha": self.class_weight_alpha,
            "positive_weight": self.positive_weight,
            "notes": self.notes,
        }
        return json.loads(json.dumps(payload, sort_keys=True))

    def identity_json_dict(self) -> dict[str, Any]:
        payload = self.to_json_dict()
        payload.pop("notes", None)
        return payload


LR_BASE: ImmutableParams = ImmutableParams(
    {
        "learning_rate": 0.05,
        "l2": 0.05,
        "epochs": 12,
        "batch_size": 2048,
        "tolerance": 0.0,
    }
)

BOOSTING_BASELINE: ImmutableParams = ImmutableParams(
    {
        "n_estimators": 50,
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_bins": 64,
        "l2": 1.0,
        "gamma": 0.0,
        "min_child_weight": 1.0,
        "subsample": 1.0,
        "colsample": 1.0,
    }
)

BOOSTING_CONFIG_D: ImmutableParams = ImmutableParams(
    {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.08,
        "n_bins": 96,
        "l2": 1.0,
        "gamma": 0.0,
        "min_child_weight": 1.0,
        "subsample": 0.8,
        "colsample": 0.75,
        "early_stopping_rounds": 25,
        "min_delta": 0.0001,
    }
)

_LR_BASE: dict[str, Any] = {
    "learning_rate": 0.05,
    "l2": 0.05,
    "epochs": 12,
    "batch_size": 2048,
    "tolerance": 0.0,
}

_BOOSTING_BASELINE: dict[str, Any] = {
    "n_estimators": 50,
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_bins": 64,
    "l2": 1.0,
    "gamma": 0.0,
    "min_child_weight": 1.0,
    "subsample": 1.0,
    "colsample": 1.0,
}

_BOOSTING_CONFIG_D: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.08,
    "n_bins": 96,
    "l2": 1.0,
    "gamma": 0.0,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample": 0.75,
    "early_stopping_rounds": 25,
    "min_delta": 0.0001,
}

_ROUND_LR_PAIRS: tuple[tuple[int, float], ...] = (
    (100, 0.12),
    (150, 0.10),
    (200, 0.08),
    (300, 0.05),
    (400, 0.03),
)

lr_baseline = ModelConfig(
    config_id="lr_baseline",
    model_family="lr",
    feature_profile="baseline",
    params=_LR_BASE,
    class_weight_alpha=1.0,
    notes="S2 baseline logistic regression with balanced class weights.",
)

boosting_baseline = ModelConfig(
    config_id="boosting_baseline",
    model_family="boosting",
    feature_profile="baseline",
    params=_BOOSTING_BASELINE,
    positive_weight="balanced",
    notes="S1 baseline histogram boosting configuration.",
)

boosting_config_d = ModelConfig(
    config_id="boosting_config_d",
    model_family="boosting",
    feature_profile="baseline",
    params=_BOOSTING_CONFIG_D,
    positive_weight="balanced",
    notes="Retuned Config D local OOT anchor.",
)


def registry() -> dict[str, ModelConfig]:
    configs: dict[str, ModelConfig] = {}
    for config in (lr_baseline, boosting_baseline, boosting_config_d):
        configs[config.config_id] = _copy_config(config)

    for config in _lr_audit_configs():
        configs[config.config_id] = config
    for config in boosting_search_configs():
        configs[config.config_id] = config
    return configs


def get_model_config(config_id: str) -> ModelConfig:
    configs = registry()
    if config_id not in configs:
        raise KeyError(f"Unknown model config: {config_id}")
    return _copy_config(configs[config_id])


def config_hash(config: ModelConfig) -> str:
    payload = json.dumps(config.identity_json_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def class_weight_from_alpha(positive_rate: float, alpha: float | None) -> dict[int, float] | None:
    if alpha is None:
        return None
    p = float(positive_rate)
    a = float(alpha)
    if not math.isfinite(a) or a < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if a == 0.0:
        return None
    if not math.isfinite(p) or not 0.0 < p < 1.0:
        return None
    return {0: 1.0, 1: ((1.0 - p) / p) ** a}


def lr_audit_config_ids() -> list[str]:
    return [config.config_id for config in _lr_audit_configs()]


def boosting_search_configs() -> list[ModelConfig]:
    configs: list[ModelConfig] = []
    seen: set[str] = set()

    def add(config_id: str, params: dict[str, Any], notes: str) -> None:
        _validate_boosting_search_params(params)
        key = json.dumps(params, sort_keys=True, separators=(",", ":"))
        if key in seen:
            return
        seen.add(key)
        configs.append(
            ModelConfig(
                config_id=config_id,
                model_family="boosting",
                feature_profile="baseline",
                params=dict(params),
                positive_weight="balanced",
                notes=notes,
            )
        )

    add("boosting_search_d_anchor", dict(_BOOSTING_CONFIG_D), "Config D protected search anchor.")
    add(
        "boosting_search_fast_anchor",
        {**_BOOSTING_CONFIG_D, "n_estimators": 150, "learning_rate": 0.10},
        "Faster staged anchor near Config D.",
    )
    add(
        "boosting_search_long_anchor",
        {**_BOOSTING_CONFIG_D, "n_estimators": 300, "learning_rate": 0.05},
        "Longer low-rate staged anchor near Config D.",
    )
    add(
        "boosting_search_depth4_anchor",
        {**_BOOSTING_CONFIG_D, "max_depth": 4},
        "Lower-depth anchor around Config D.",
    )

    for n_estimators, learning_rate in _ROUND_LR_PAIRS:
        for max_depth in (4, 5):
            for n_bins in (64, 96):
                add(
                    _boosting_id("capacity", n_estimators, learning_rate, max_depth, n_bins),
                    {
                        **_BOOSTING_CONFIG_D,
                        "n_estimators": n_estimators,
                        "learning_rate": learning_rate,
                        "max_depth": max_depth,
                        "n_bins": n_bins,
                    },
                    "Capacity sweep over protected round/rate pairs.",
                )

    for n_estimators, learning_rate in ((150, 0.10), (200, 0.08), (300, 0.05)):
        for max_depth in (3, 6):
            for n_bins in (64, 96, 128):
                if max_depth == 6 and n_bins == 128 and n_estimators >= 300:
                    continue
                child_weights = (10.0, 20.0) if max_depth == 6 else (1.0,)
                for min_child_weight in child_weights:
                    add(
                        _boosting_id(
                            f"boundary_mcw{_token(min_child_weight)}",
                            n_estimators,
                            learning_rate,
                            max_depth,
                            n_bins,
                        ),
                        {
                            **_BOOSTING_CONFIG_D,
                            "n_estimators": n_estimators,
                            "learning_rate": learning_rate,
                            "max_depth": max_depth,
                            "n_bins": n_bins,
                            "min_child_weight": min_child_weight,
                        },
                        "Boundary probe for depth, child weight, and histogram bins.",
                    )

    for subsample in (0.7, 0.8, 0.9, 1.0):
        for colsample in (0.6, 0.75, 0.9, 1.0):
            add(
                f"boosting_search_sampling_sub{_token(subsample)}_col{_token(colsample)}",
                {
                    **_BOOSTING_CONFIG_D,
                    "subsample": subsample,
                    "colsample": colsample,
                },
                "Sampling sweep around Config D.",
            )

    return configs


def _lr_audit_configs() -> list[ModelConfig]:
    configs: list[ModelConfig] = []

    for alpha, suffix in ((0.0, "0"), (0.5, "0_5"), (1.0, "1")):
        configs.append(
            ModelConfig(
                config_id=f"lr_alpha_{suffix}",
                model_family="lr",
                feature_profile="baseline",
                params=_LR_BASE,
                class_weight_alpha=alpha,
                notes=f"LR class-weight alpha audit at alpha={alpha:g}.",
            )
        )

    for l2 in (0.01, 0.05, 0.1, 0.2):
        configs.append(
            ModelConfig(
                config_id=f"lr_l2_{_token(l2)}",
                model_family="lr",
                feature_profile="baseline",
                params={**_LR_BASE, "l2": l2},
                class_weight_alpha=1.0,
                notes=f"LR regularization audit at l2={l2:g}.",
            )
        )

    for learning_rate in (0.03, 0.05, 0.08):
        for epochs in (12, 20, 30):
            configs.append(
                ModelConfig(
                    config_id=f"lr_epoch_{epochs}_lr_{_token(learning_rate)}",
                    model_family="lr",
                    feature_profile="baseline",
                    params={**_LR_BASE, "learning_rate": learning_rate, "epochs": epochs},
                    class_weight_alpha=1.0,
                    notes="LR learning-rate and epoch audit.",
                )
            )

    return configs


def _validate_boosting_search_params(params: dict[str, Any]) -> None:
    pair = (params["n_estimators"], params["learning_rate"])
    if pair not in _ROUND_LR_PAIRS:
        raise ValueError(f"unsupported boosting search round/rate pair: {pair}")
    if params["n_bins"] > 128:
        raise ValueError("boosting search n_bins must be <= 128")
    if params["max_depth"] == 6 and params["n_bins"] == 128 and params["n_estimators"] >= 300:
        raise ValueError("protected search excludes depth=6, n_bins=128, n_estimators>=300")


def _boosting_id(prefix: str, n_estimators: int, learning_rate: float, max_depth: int, n_bins: int) -> str:
    return f"boosting_search_{prefix}_n{n_estimators}_lr{_token(learning_rate)}_d{max_depth}_b{n_bins}"


def _token(value: float) -> str:
    return f"{value:g}".replace(".", "_")


def _copy_config(config: ModelConfig) -> ModelConfig:
    return ModelConfig(
        config_id=config.config_id,
        model_family=config.model_family,
        feature_profile=config.feature_profile,
        params=dict(config.params),
        class_weight_alpha=config.class_weight_alpha,
        positive_weight=config.positive_weight,
        notes=config.notes,
    )


__all__ = [
    "ModelConfig",
    "boosting_baseline",
    "boosting_config_d",
    "boosting_search_configs",
    "class_weight_from_alpha",
    "config_hash",
    "get_model_config",
    "lr_baseline",
    "lr_audit_config_ids",
    "registry",
]
