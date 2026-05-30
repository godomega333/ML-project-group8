from __future__ import annotations

import copy
import json
import unittest

from fraud_model import ModelConfig, config_hash as exported_config_hash, get_model_config as exported_get_model_config
from fraud_model.configs import (
    BOOSTING_BASELINE,
    BOOSTING_CONFIG_D,
    LR_BASE,
    boosting_config_d,
    boosting_search_configs,
    class_weight_from_alpha,
    config_hash,
    get_model_config,
    lr_audit_config_ids,
)


class ConfigRegistryTest(unittest.TestCase):
    def test_get_model_config_returns_deep_copies(self) -> None:
        first = get_model_config("boosting_config_d")
        second = get_model_config("boosting_config_d")

        self.assertIsNot(first, second)
        self.assertIsNot(first.params, second.params)
        with self.assertRaises(TypeError):
            first.params["max_depth"] = 99
        self.assertEqual(second.model_family, "boosting")
        self.assertEqual(second.feature_profile, "baseline")
        self.assertEqual(second.params["max_depth"], 5)

    def test_config_hash_is_stable_and_changes_when_params_change(self) -> None:
        first = get_model_config("lr_baseline")
        second = get_model_config("lr_baseline")

        stable_hash = config_hash(first)
        self.assertEqual(len(stable_hash), 16)
        self.assertEqual(stable_hash, config_hash(second))

        changed_params = dict(first.params)
        changed_params["learning_rate"] = 0.08
        changed = ModelConfig(
            config_id=first.config_id,
            model_family=first.model_family,
            feature_profile=first.feature_profile,
            params=changed_params,
            class_weight_alpha=first.class_weight_alpha,
            positive_weight=first.positive_weight,
            notes=first.notes,
        )

        self.assertNotEqual(config_hash(first), config_hash(changed))

    def test_config_hash_ignores_notes(self) -> None:
        config = get_model_config("lr_baseline")
        changed_notes = ModelConfig(
            config_id=config.config_id,
            model_family=config.model_family,
            feature_profile=config.feature_profile,
            params=config.params,
            class_weight_alpha=config.class_weight_alpha,
            positive_weight=config.positive_weight,
            notes="audit-only text change",
        )

        self.assertEqual(config_hash(config), config_hash(changed_notes))

    def test_package_exports_minimal_config_api(self) -> None:
        self.assertIs(exported_config_hash, config_hash)
        self.assertIs(exported_get_model_config, get_model_config)
        self.assertIsInstance(exported_get_model_config("lr_baseline"), ModelConfig)

    def test_canonical_config_params_are_immutable(self) -> None:
        with self.assertRaises(TypeError):
            boosting_config_d.params["max_depth"] = 99

        self.assertEqual(get_model_config("boosting_config_d").params["max_depth"], 5)

    def test_params_are_immutable_copyable_and_json_serializable(self) -> None:
        config = get_model_config("boosting_config_d")

        with self.assertRaises(TypeError):
            config.params["max_depth"] = 99

        shallow = copy.copy(config.params)
        deep = copy.deepcopy(config.params)

        self.assertEqual(shallow["max_depth"], 5)
        self.assertEqual(deep["max_depth"], 5)
        self.assertEqual(json.loads(json.dumps(config.params))["max_depth"], 5)

    def test_base_templates_cannot_corrupt_generated_configs(self) -> None:
        for template, key, bad_value in (
            (LR_BASE, "learning_rate", 99.0),
            (BOOSTING_BASELINE, "max_depth", 99),
            (BOOSTING_CONFIG_D, "max_depth", 99),
        ):
            with self.assertRaises(TypeError):
                template[key] = bad_value

        self.assertEqual(get_model_config("lr_baseline").params["learning_rate"], 0.05)
        self.assertEqual(get_model_config("boosting_baseline").params["max_depth"], 3)
        self.assertEqual(get_model_config("boosting_config_d").params["max_depth"], 5)
        self.assertTrue(
            any(
                config.config_id == "boosting_search_d_anchor" and config.params["max_depth"] == 5
                for config in boosting_search_configs()
            )
        )

    def test_class_weight_from_alpha(self) -> None:
        self.assertIsNone(class_weight_from_alpha(0.10, None))
        self.assertIsNone(class_weight_from_alpha(0.10, 0.0))
        self.assertIsNone(class_weight_from_alpha(0.0, 0.5))
        self.assertIsNone(class_weight_from_alpha(1.0, 0.5))

        alpha_half = class_weight_from_alpha(0.10, 0.5)
        self.assertEqual(alpha_half, {0: 1.0, 1: 3.0})

        alpha_one = class_weight_from_alpha(0.10, 1.0)
        self.assertEqual(alpha_one, {0: 1.0, 1: 9.0})

    def test_lr_audit_config_ids_includes_required_configs(self) -> None:
        config_ids = lr_audit_config_ids()

        self.assertIn("lr_alpha_0", config_ids)
        self.assertIn("lr_alpha_0_5", config_ids)
        self.assertIn("lr_alpha_1", config_ids)
        self.assertIn("lr_l2_0_01", config_ids)
        self.assertIn("lr_epoch_30_lr_0_08", config_ids)

    def test_boosting_search_configs_respects_protected_space(self) -> None:
        configs = boosting_search_configs()
        allowed_pairs = {
            (100, 0.12),
            (150, 0.10),
            (200, 0.08),
            (300, 0.05),
            (400, 0.03),
        }

        self.assertGreaterEqual(len(configs), 40)
        self.assertLessEqual(len(configs), 60)

        for config in configs:
            params = config.params
            pair = (params["n_estimators"], params["learning_rate"])
            self.assertIn(pair, allowed_pairs)
            self.assertLessEqual(params["n_bins"], 128)
            self.assertFalse(
                params["max_depth"] == 6
                and params["n_bins"] == 128
                and params["n_estimators"] >= 300
            )
            if params["max_depth"] == 6:
                self.assertIn(params["min_child_weight"], {10.0, 20.0})


if __name__ == "__main__":
    unittest.main()
