"""Tests for models.py: construction, fitting, determinism, quantile support."""
from __future__ import annotations

import numpy as np
import pytest

from models import MODEL_NAMES, ConstantRegressor, build_model


def test_constant_regressor_predicts_mean():
    model = ConstantRegressor().fit(None, [1.0, 2.0, 3.0])
    assert model.value_ == pytest.approx(2.0)
    np.testing.assert_allclose(model.predict([[0], [0], [0]]), [2.0, 2.0, 2.0])


def test_build_model_rejects_unknown():
    with pytest.raises(ValueError):
        build_model("nope")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_predicts_finite(name):
    rng = np.random.RandomState(0)
    X = rng.rand(60, 4)
    y = rng.rand(60) * 100
    estimator = build_model(name, seed=0)
    estimator.fit(X, y)
    predictions = estimator.predict(X)
    assert len(predictions) == len(y)
    assert np.all(np.isfinite(predictions))


@pytest.mark.parametrize(
    "name", ["random_forest", "gradient_boosting", "safe_side", "quantile_10", "quantile_90"]
)
def test_same_seed_is_deterministic(name):
    rng = np.random.RandomState(1)
    X = rng.rand(80, 4)
    y = rng.rand(80) * 100
    first = build_model(name, seed=3).fit(X, y).predict(X)
    second = build_model(name, seed=3).fit(X, y).predict(X)
    np.testing.assert_allclose(first, second)


def test_quantile_loss_api_supported():
    """Verifies the scikit-learn quantile API used for the safe-side model (B3)."""
    from sklearn.ensemble import GradientBoostingRegressor

    X = np.arange(40, dtype=float).reshape(-1, 1)
    y = np.arange(40, dtype=float)
    estimator = GradientBoostingRegressor(
        loss="quantile", alpha=0.3, n_estimators=20, random_state=0
    )
    estimator.fit(X, y)
    assert estimator.predict(np.arange(5, dtype=float).reshape(-1, 1)).shape == (5,)


def test_quantile_alpha_is_ordered_on_average():
    """On a monotone target, low alpha should not exceed high alpha on average."""
    rng = np.random.RandomState(2)
    X = rng.rand(200, 3)
    y = X[:, 0] * 100 + rng.rand(200)
    low = build_model("quantile_10", seed=0).fit(X, y).predict(X)
    high = build_model("quantile_90", seed=0).fit(X, y).predict(X)
    assert low.mean() <= high.mean()
