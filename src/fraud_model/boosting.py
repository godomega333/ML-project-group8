"""From-scratch sparsity-aware histogram gradient boosting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fraud_model.metrics import roc_auc


@dataclass
class TreeNode:
    value: float | None = None
    feature_index: int = -1
    bin_index: int = -1
    nan_go_left: bool = True
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


@dataclass
class HistogramGradientBoostingClassifier:
    n_estimators: int = 50
    max_depth: int = 3
    learning_rate: float = 0.05
    n_bins: int = 64
    l2: float = 1.0
    gamma: float = 0.0
    min_child_weight: float = 1.0
    subsample: float = 1.0
    colsample: float = 1.0
    early_stopping_rounds: int | None = None
    min_delta: float = 0.0
    positive_weight: float | None = None
    seed: int = 42

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
    ) -> "HistogramGradientBoostingClassifier":
        self._validate_hyperparameters()
        train_x = self._prepare_matrix(x)
        train_y = self._prepare_target(y, train_x.shape[0])
        valid_pair = self._prepare_validation(x_valid, y_valid, train_x.shape[1])

        self.n_features_in_ = train_x.shape[1]
        self.missing_bin_ = int(self.n_bins) - 1
        self.bin_edges_ = self._make_bin_edges(train_x)
        train_bins = self._bin_matrix(train_x)

        prior = np.clip(float(np.mean(train_y)), 1e-6, 1.0 - 1e-6)
        self.base_score_ = float(np.log(prior / (1.0 - prior)))
        self.trees_: list[TreeNode] = []
        self.history_: dict[str, list[float]] = {"train_auc": [], "valid_auc": [], "best_valid_auc": []}
        self.best_iteration_ = 0
        self.best_valid_auc_ = float("nan")
        self.early_stopping_enabled_ = self.early_stopping_rounds is not None and valid_pair is not None
        self.early_stopped_ = False

        train_logits = np.full(train_y.shape[0], self.base_score_, dtype=np.float64)
        valid_bins: np.ndarray | None = None
        valid_y: np.ndarray | None = None
        valid_logits: np.ndarray | None = None
        if valid_pair is not None:
            valid_x, valid_y = valid_pair
            valid_bins = self._bin_matrix(valid_x)
            valid_logits = np.full(valid_y.shape[0], self.base_score_, dtype=np.float64)

        rng = np.random.default_rng(self.seed)
        best_valid_auc = float("-inf")
        best_iteration = 0
        stale_rounds = 0
        early_stopping_rounds = (
            int(self.early_stopping_rounds)
            if self.early_stopping_enabled_
            else None
        )
        for _ in range(int(self.n_estimators)):
            proba = self._sigmoid(train_logits)
            gradients = proba - train_y
            hessians = proba * (1.0 - proba) + 1e-6
            if self.positive_weight is not None:
                positive_mask = train_y == 1.0
                gradients = gradients.copy()
                hessians = hessians.copy()
                gradients[positive_mask] *= float(self.positive_weight)
                hessians[positive_mask] *= float(self.positive_weight)

            row_indices = self._sample_rows(train_y.shape[0], rng)
            feature_indices = self._sample_features(train_x.shape[1], rng)
            tree = self._build_tree(train_bins, gradients, hessians, row_indices, feature_indices, depth=0)
            self.trees_.append(tree)

            train_logits += self.learning_rate * self._predict_tree_bins(tree, train_bins)
            self.history_["train_auc"].append(roc_auc(train_y, self._sigmoid(train_logits)))
            if valid_bins is not None and valid_y is not None and valid_logits is not None:
                valid_logits += self.learning_rate * self._predict_tree_bins(tree, valid_bins)
                current_valid_auc = roc_auc(valid_y, self._sigmoid(valid_logits))
                self.history_["valid_auc"].append(current_valid_auc)
                if np.isfinite(current_valid_auc) and current_valid_auc > best_valid_auc + float(self.min_delta):
                    best_valid_auc = current_valid_auc
                    best_iteration = len(self.trees_)
                    stale_rounds = 0
                else:
                    stale_rounds += 1
                self.history_["best_valid_auc"].append(
                    best_valid_auc if np.isfinite(best_valid_auc) else float("nan")
                )
                if early_stopping_rounds is not None and stale_rounds >= early_stopping_rounds:
                    self.early_stopped_ = True
                    break

        self.best_iteration_ = best_iteration if best_iteration > 0 else len(self.trees_)
        if self.early_stopped_ and best_iteration > 0:
            self.trees_ = self.trees_[:best_iteration]
        if np.isfinite(best_valid_auc):
            self.best_valid_auc_ = best_valid_auc

        return self

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "base_score_") or not hasattr(self, "trees_"):
            raise ValueError("model must be fit before prediction")
        bins = self._bin_matrix(x)
        logits = np.full(bins.shape[0], self.base_score_, dtype=np.float64)
        for tree in self.trees_:
            logits += self.learning_rate * self._predict_tree_bins(tree, bins)
        return logits

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid(self.predict_logits(x))

    def _make_bin_edges(self, x: np.ndarray) -> list[np.ndarray]:
        max_thresholds = max(0, int(self.n_bins) - 2)
        edges: list[np.ndarray] = []
        for feature_index in range(x.shape[1]):
            values = x[:, feature_index]
            finite_values = values[np.isfinite(values)]
            if finite_values.size <= 1 or max_thresholds == 0:
                edges.append(np.array([], dtype=np.float32))
                continue

            unique_values = np.unique(finite_values)
            n_thresholds = min(max_thresholds, unique_values.size - 1)
            if n_thresholds <= 0:
                edges.append(np.array([], dtype=np.float32))
                continue

            if unique_values.size - 1 <= max_thresholds:
                feature_edges = self._midpoints(unique_values)
            else:
                quantiles = np.linspace(0.0, 1.0, n_thresholds + 2, dtype=np.float64)[1:-1]
                quantile_values = np.quantile(finite_values, quantiles)
                feature_edges = self._quantiles_to_midpoints(unique_values, quantile_values)
            edges.append(feature_edges)
        return edges

    def _bin_matrix(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "bin_edges_") or not hasattr(self, "missing_bin_"):
            raise ValueError("bin edges are unavailable; fit the model first")
        matrix = self._prepare_matrix(x)
        if matrix.shape[1] != len(self.bin_edges_):
            raise ValueError("x must have the same number of columns used during fit")

        bins = np.empty(matrix.shape, dtype=np.int32)
        for feature_index, edges in enumerate(self.bin_edges_):
            values = matrix[:, feature_index]
            finite_mask = np.isfinite(values)
            bins[~finite_mask, feature_index] = self.missing_bin_
            bins[finite_mask, feature_index] = np.searchsorted(edges, values[finite_mask], side="right")
        return bins

    def _build_tree(
        self,
        bins: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        row_indices: np.ndarray,
        feature_indices: np.ndarray,
        depth: int,
    ) -> TreeNode:
        grad_sum = float(np.sum(gradients[row_indices]))
        hess_sum = float(np.sum(hessians[row_indices]))
        leaf_value = self._leaf_value(grad_sum, hess_sum)
        if (
            depth >= int(self.max_depth)
            or row_indices.shape[0] <= 1
            or (self.min_child_weight > 0.0 and hess_sum < 2.0 * self.min_child_weight)
        ):
            return TreeNode(value=leaf_value)

        split = self._best_split(bins, gradients, hessians, row_indices, feature_indices, grad_sum, hess_sum)
        if split is None:
            return TreeNode(value=leaf_value)

        feature_index, bin_index, nan_go_left, _ = split
        left_rows, right_rows = self._partition_rows(bins, row_indices, feature_index, bin_index, nan_go_left)
        if left_rows.shape[0] == 0 or right_rows.shape[0] == 0:
            return TreeNode(value=leaf_value)

        return TreeNode(
            feature_index=feature_index,
            bin_index=bin_index,
            nan_go_left=nan_go_left,
            left=self._build_tree(bins, gradients, hessians, left_rows, feature_indices, depth + 1),
            right=self._build_tree(bins, gradients, hessians, right_rows, feature_indices, depth + 1),
        )

    def _best_split(
        self,
        bins: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        row_indices: np.ndarray,
        feature_indices: np.ndarray,
        grad_sum: float,
        hess_sum: float,
    ) -> tuple[int, int, bool, float] | None:
        best: tuple[int, int, bool, float] | None = None
        parent_score = self._score_node(grad_sum, hess_sum)

        for feature_index in feature_indices:
            feature_bins = bins[row_indices, feature_index]
            grad_hist = np.bincount(feature_bins, weights=gradients[row_indices], minlength=int(self.n_bins))
            hess_hist = np.bincount(feature_bins, weights=hessians[row_indices], minlength=int(self.n_bins))

            last_finite_bin = len(self.bin_edges_[feature_index])
            finite_slice = slice(0, last_finite_bin + 1)
            finite_grad = grad_hist[finite_slice]
            finite_hess = hess_hist[finite_slice]
            finite_grad_total = float(np.sum(finite_grad))
            finite_hess_total = float(np.sum(finite_hess))

            missing_grad = float(grad_hist[self.missing_bin_])
            missing_hess = float(hess_hist[self.missing_bin_])

            prefix_grad = np.cumsum(finite_grad)
            prefix_hess = np.cumsum(finite_hess)

            for bin_index in range(-1, last_finite_bin + 1):
                if bin_index == -1:
                    left_grad_finite = 0.0
                    left_hess_finite = 0.0
                else:
                    left_grad_finite = float(prefix_grad[bin_index])
                    left_hess_finite = float(prefix_hess[bin_index])

                right_grad_finite = finite_grad_total - left_grad_finite
                right_hess_finite = finite_hess_total - left_hess_finite

                candidates = (
                    (
                        True,
                        left_grad_finite + missing_grad,
                        left_hess_finite + missing_hess,
                        right_grad_finite,
                        right_hess_finite,
                    ),
                    (
                        False,
                        left_grad_finite,
                        left_hess_finite,
                        right_grad_finite + missing_grad,
                        right_hess_finite + missing_hess,
                    ),
                )

                for nan_go_left, left_grad, left_hess, right_grad, right_hess in candidates:
                    if not self._valid_child_hessians(left_hess, right_hess):
                        continue
                    gain = (
                        0.5
                        * (
                            self._score_node(left_grad, left_hess)
                            + self._score_node(right_grad, right_hess)
                            - parent_score
                        )
                        - self.gamma
                    )
                    if gain <= 0.0:
                        continue
                    if best is None or gain > best[3] + 1e-12:
                        best = (int(feature_index), int(bin_index), bool(nan_go_left), float(gain))

        return best

    def _partition_rows(
        self,
        bins: np.ndarray,
        row_indices: np.ndarray,
        feature_index: int,
        bin_index: int,
        nan_go_left: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        feature_bins = bins[row_indices, feature_index]
        missing = feature_bins == self.missing_bin_
        left_mask = np.where(missing, nan_go_left, feature_bins <= bin_index)
        return row_indices[left_mask], row_indices[~left_mask]

    def _predict_tree_bins(self, tree: TreeNode, bins: np.ndarray) -> np.ndarray:
        predictions = np.empty(bins.shape[0], dtype=np.float64)
        for row_index in range(bins.shape[0]):
            node = tree
            while node.value is None:
                feature_bin = bins[row_index, node.feature_index]
                if feature_bin == self.missing_bin_:
                    node = node.left if node.nan_go_left else node.right
                elif feature_bin <= node.bin_index:
                    node = node.left
                else:
                    node = node.right
                if node is None:
                    raise ValueError("tree contains an incomplete split")
            predictions[row_index] = node.value
        return predictions

    def _sample_rows(self, n_rows: int, rng: np.random.Generator) -> np.ndarray:
        if self.subsample >= 1.0:
            return np.arange(n_rows, dtype=np.int32)
        sample_size = max(1, int(np.ceil(self.subsample * n_rows)))
        return np.sort(rng.choice(n_rows, size=sample_size, replace=False)).astype(np.int32, copy=False)

    def _sample_features(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        if self.colsample >= 1.0:
            return np.arange(n_features, dtype=np.int32)
        sample_size = max(1, int(np.ceil(self.colsample * n_features)))
        return np.sort(rng.choice(n_features, size=sample_size, replace=False)).astype(np.int32, copy=False)

    def _prepare_validation(
        self,
        x_valid: np.ndarray | None,
        y_valid: np.ndarray | None,
        n_features: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if x_valid is None and y_valid is None:
            return None
        if x_valid is None or y_valid is None:
            raise ValueError("x_valid and y_valid must be provided together")
        valid_x = self._prepare_matrix(x_valid)
        if valid_x.shape[1] != n_features:
            raise ValueError("x_valid must have the same number of columns as x")
        valid_y = self._prepare_target(y_valid, valid_x.shape[0])
        return valid_x, valid_y

    def _validate_hyperparameters(self) -> None:
        if int(self.n_estimators) != self.n_estimators or self.n_estimators < 0:
            raise ValueError("n_estimators must be a non-negative integer")
        if int(self.max_depth) != self.max_depth or self.max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if int(self.n_bins) != self.n_bins or self.n_bins < 3:
            raise ValueError("n_bins must be an integer greater than or equal to 3")
        if not np.isfinite(self.l2) or self.l2 < 0.0:
            raise ValueError("l2 must be finite and non-negative")
        if not np.isfinite(self.gamma) or self.gamma < 0.0:
            raise ValueError("gamma must be finite and non-negative")
        if not np.isfinite(self.min_child_weight) or self.min_child_weight < 0.0:
            raise ValueError("min_child_weight must be finite and non-negative")
        if not np.isfinite(self.subsample) or not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in the interval (0, 1]")
        if not np.isfinite(self.colsample) or not (0.0 < self.colsample <= 1.0):
            raise ValueError("colsample must be in the interval (0, 1]")
        if self.early_stopping_rounds is not None:
            rounds = self.early_stopping_rounds
            if (
                isinstance(rounds, (bool, np.bool_))
                or not isinstance(rounds, (int, np.integer))
                or rounds <= 0
            ):
                raise ValueError("early_stopping_rounds must be a positive integer or None")
        if not np.isfinite(self.min_delta) or self.min_delta < 0.0:
            raise ValueError("min_delta must be finite and non-negative")
        if self.positive_weight is not None and (
            not np.isfinite(self.positive_weight) or self.positive_weight <= 0.0
        ):
            raise ValueError("positive_weight must be finite and positive")

    def _valid_child_hessians(self, left_hess: float, right_hess: float) -> bool:
        return left_hess >= self.min_child_weight and right_hess >= self.min_child_weight

    def _leaf_value(self, grad_sum: float, hess_sum: float) -> float:
        denominator = hess_sum + self.l2
        if denominator <= 0.0:
            return 0.0
        return float(-grad_sum / denominator)

    def _score_node(self, grad_sum: float, hess_sum: float) -> float:
        denominator = hess_sum + self.l2
        if denominator <= 0.0:
            return 0.0
        return float((grad_sum * grad_sum) / denominator)

    @staticmethod
    def _midpoints(unique_values: np.ndarray) -> np.ndarray:
        values = np.asarray(unique_values, dtype=np.float64)
        return ((values[:-1] + values[1:]) / 2.0).astype(np.float32, copy=False)

    @staticmethod
    def _quantiles_to_midpoints(unique_values: np.ndarray, quantile_values: np.ndarray) -> np.ndarray:
        values = np.asarray(unique_values, dtype=np.float64)
        positions = np.searchsorted(values, quantile_values, side="right")
        positions = np.clip(positions, 1, values.shape[0] - 1)
        lower = values[positions - 1]
        upper = values[positions]
        return np.unique((lower + upper) / 2.0).astype(np.float32, copy=False)

    @staticmethod
    def _prepare_matrix(x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("x must be a two-dimensional matrix")
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("x must have at least one row and one column")
        if not np.all(np.isfinite(matrix) | np.isnan(matrix)):
            raise ValueError("x must contain only finite values or NaNs")
        return matrix

    @staticmethod
    def _prepare_target(y: np.ndarray, n_rows: int) -> np.ndarray:
        target = np.asarray(y, dtype=np.float64).reshape(-1)
        if target.shape[0] != n_rows:
            raise ValueError("y must have the same number of rows as x")
        if not np.all(np.isfinite(target)):
            raise ValueError("y must contain only finite values")
        if not np.all((target == 0.0) | (target == 1.0)):
            raise ValueError("y must contain only 0/1 labels")
        return target

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(logits, dtype=np.float64), -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))
