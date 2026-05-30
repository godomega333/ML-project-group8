"""From-scratch IEEE-CIS fraud modeling package."""

from __future__ import annotations

from fraud_model.configs import ModelConfig, config_hash, get_model_config

__version__ = "0.1.0"

__all__ = ["ModelConfig", "__version__", "config_hash", "get_model_config"]
