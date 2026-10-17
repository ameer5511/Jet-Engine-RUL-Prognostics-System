"""Model construction for AeroGuard Phase 2.

Five models are defined by name; all classical scikit-learn trees or a constant
floor. No neural networks (see MASTER_SPEC.md scope).

Names
-----
constant            : predicts the training mean target (the floor)
random_forest       : RandomForestRegressor
gradient_boosting   : main model, squared error
safe_side           : quantile GB at alpha 0.30 (leans cautious)
quantile_10 / _90   : quantile GB at alpha 0.10 / 0.90 (prediction range)
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

MODEL_NAMES = (
    "constant",
    "random_forest",
    "gradient_boosting",
    "safe_side",
    "quantile_10",
    "quantile_90",
)

# The GB configuration shared by the point and quantile models.
_GB_KWARGS = dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=1.0)
_RF_KWARGS = dict(n_estimators=250, min_samples_leaf=1, n_jobs=-1)


class ConstantRegressor:
    """Predicts a single constant (the mean of the training targets)."""

    def fit(self, X, y):  # noqa: N803 - sklearn signature
        self.value_ = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X):  # noqa: N803 - sklearn signature
        return np.full(len(X), self.value_, dtype=float)


def build_model(name: str, seed: int = 0, **overrides: Any):
    """Return an unfitted estimator for ``name``.

    ``overrides`` are passed to the underlying constructor (or, for
    :class:`ConstantRegressor`, accepted and ignored).
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")

    if name == "constant":
        return ConstantRegressor()

    if name == "random_forest":
        kwargs = {**_RF_KWARGS, "random_state": seed, **overrides}
        return RandomForestRegressor(**kwargs)

    point_kwargs = {**_GB_KWARGS, "loss": "squared_error", "random_state": seed}
    quantile_alphas = {"safe_side": 0.30, "quantile_10": 0.10, "quantile_90": 0.90}

    if name == "gradient_boosting":
        return GradientBoostingRegressor(**(point_kwargs | overrides))

    alpha = quantile_alphas[name]
    kwargs = {**_GB_KWARGS, "loss": "quantile", "alpha": alpha, "random_state": seed}
    return GradientBoostingRegressor(**(kwargs | overrides))


def fit_predict(
    name: str, X_train, y_train, X_test, seed: int = 0, **overrides: Any
) -> np.ndarray:
    """Fit a fresh model and return its predictions on ``X_test``."""
    model = build_model(name, seed=seed, **overrides)
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_test), dtype=float)
