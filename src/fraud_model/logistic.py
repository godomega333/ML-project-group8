"""From-scratch matrix logistic regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fraud_model.metrics import roc_auc


@dataclass
class MatrixLogisticRegression:
    learning_rate: float = 0.05
    l2: float = 0.1
    epochs: int = 100
    batch_size: int | None = None
    class_weight: dict[int, float] | None = None
    seed: int = 42
    tolerance: float = 1e-7

    def __post_init__(self) -> None:
        self.coef_: np.ndarray | None = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        valid_x: np.ndarray | None = None,
        valid_y: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        train_x = self._prepare_matrix(x)
        train_y = self._prepare_target(y, train_x.shape[0])
        valid_pair = self._prepare_validation(valid_x, valid_y, train_x.shape[1])
        self._validate_hyperparameters()

        self.coef_ = np.zeros(train_x.shape[1], dtype=np.float64)
        regularized = self._regularization_mask(train_x)
        sample_weight = self._sample_weights(train_y)
        rng = np.random.default_rng(self.seed)
        history: dict[str, list[float]] = {
            "train_loss": [],
            "valid_loss": [],
            "train_auc": [],
            "valid_auc": [],
        }

        previous_loss: float | None = None
        for _ in range(int(self.epochs)):
            for batch_idx in self._batch_indices(train_x.shape[0], rng):
                self._gradient_step(train_x[batch_idx], train_y[batch_idx], sample_weight[batch_idx], regularized)

            train_proba = self.predict_proba(train_x)
            train_loss = self._loss(train_y, train_proba, sample_weight, regularized)
            history["train_loss"].append(train_loss)
            history["train_auc"].append(roc_auc(train_y, train_proba))

            if valid_pair is None:
                history["valid_loss"].append(float("nan"))
                history["valid_auc"].append(float("nan"))
            else:
                vx, vy = valid_pair
                valid_weight = self._sample_weights(vy)
                valid_proba = self.predict_proba(vx)
                history["valid_loss"].append(self._loss(vy, valid_proba, valid_weight, regularized))
                history["valid_auc"].append(roc_auc(vy, valid_proba))

            if (
                previous_loss is not None
                and self.tolerance > 0.0
                and 0.0 <= previous_loss - train_loss < self.tolerance
            ):
                break
            previous_loss = train_loss

        return history

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("model must be fit before prediction")
        matrix = self._prepare_matrix(x)
        if matrix.shape[1] != self.coef_.shape[0]:
            raise ValueError("x must have the same number of columns used during fit")
        return np.asarray(matrix @ self.coef_, dtype=np.float64).reshape(-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid(self.predict_logits(x))

    def _gradient_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        regularized: np.ndarray,
    ) -> None:
        if self.coef_ is None:
            raise ValueError("model must be initialized before training")
        proba = self._sigmoid(x @ self.coef_)
        weight_sum = float(np.sum(sample_weight))
        if weight_sum <= 0.0:
            raise ValueError("sample weights must sum to a positive value")

        errors = (proba - y) * sample_weight
        gradient = (x.T @ errors) / weight_sum
        if self.l2:
            regularization = np.zeros_like(self.coef_)
            regularization[regularized] = self.coef_[regularized]
            gradient = gradient + self.l2 * regularization
        self.coef_ = np.asarray(self.coef_ - self.learning_rate * gradient, dtype=np.float64)

    def _loss(
        self,
        y: np.ndarray,
        proba: np.ndarray,
        sample_weight: np.ndarray,
        regularized: np.ndarray,
    ) -> float:
        if self.coef_ is None:
            raise ValueError("model must be initialized before loss calculation")
        clipped = np.clip(proba, 1e-15, 1.0 - 1e-15)
        point_loss = -(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))
        weight_sum = float(np.sum(sample_weight))
        if weight_sum <= 0.0:
            raise ValueError("sample weights must sum to a positive value")

        data_loss = float(np.sum(sample_weight * point_loss) / weight_sum)
        penalty = 0.5 * self.l2 * float(np.sum(self.coef_[regularized] ** 2))
        return data_loss + penalty

    def _batch_indices(self, n_rows: int, rng: np.random.Generator) -> list[np.ndarray]:
        indices = np.arange(n_rows)
        if self.batch_size is None:
            return [indices]

        shuffled = rng.permutation(indices)
        size = min(int(self.batch_size), n_rows)
        return [shuffled[start : start + size] for start in range(0, n_rows, size)]

    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        weights = np.ones(y.shape[0], dtype=np.float64)
        if self.class_weight is None:
            return weights

        for label, weight in self.class_weight.items():
            if label not in (0, 1):
                raise ValueError("class_weight keys must be 0 or 1")
            if not np.isfinite(weight) or weight <= 0.0:
                raise ValueError("class_weight values must be positive")
            weights[y == float(label)] = float(weight)
        return weights

    def _regularization_mask(self, x: np.ndarray) -> np.ndarray:
        regularized = np.ones(x.shape[1], dtype=bool)
        if x.shape[1] > 0 and np.allclose(x[:, 0], 1.0):
            regularized[0] = False
        return regularized

    def _prepare_validation(
        self,
        valid_x: np.ndarray | None,
        valid_y: np.ndarray | None,
        n_features: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if valid_x is None and valid_y is None:
            return None
        if valid_x is None or valid_y is None:
            raise ValueError("valid_x and valid_y must be provided together")

        vx = self._prepare_matrix(valid_x)
        if vx.shape[1] != n_features:
            raise ValueError("valid_x must have the same number of columns as x")
        vy = self._prepare_target(valid_y, vx.shape[0])
        return vx, vy

    def _validate_hyperparameters(self) -> None:
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.l2) or self.l2 < 0.0:
            raise ValueError("l2 must be finite and non-negative")
        if int(self.epochs) != self.epochs or self.epochs < 0:
            raise ValueError("epochs must be a non-negative integer")
        if self.batch_size is not None and (int(self.batch_size) != self.batch_size or self.batch_size <= 0):
            raise ValueError("batch_size must be a positive integer or None")
        if not np.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")
        if self.class_weight is not None:
            for label, weight in self.class_weight.items():
                if label not in (0, 1):
                    raise ValueError("class_weight keys must be 0 or 1")
                if not np.isfinite(weight) or weight <= 0.0:
                    raise ValueError("class_weight values must be positive")

    @staticmethod
    def _prepare_matrix(x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("x must be a two-dimensional matrix")
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("x must have at least one row and one column")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("x must contain only finite values")
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
