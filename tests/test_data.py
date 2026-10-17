"""Tests for data.py: loading, RUL labels, and engine-level splits."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data import (
    COLUMNS,
    SENSOR_COLS,
    SETTING_COLS,
    add_rul,
    assert_disjoint,
    build_train_val,
    engine_ids,
    load_official_test,
    load_raw,
    split_engines,
)


def test_column_layout():
    assert len(COLUMNS) == 26
    assert COLUMNS[:2] == ["unit", "cycle"]
    assert COLUMNS[2:5] == ["setting_1", "setting_2", "setting_3"]
    assert len(SETTING_COLS) == 3
    assert len(SENSOR_COLS) == 21
    assert SENSOR_COLS[0] == "s1" and SENSOR_COLS[-1] == "s21"


def test_add_rul_last_cycle_is_zero():
    df = pd.DataFrame({"unit": [1, 1, 1], "cycle": [1, 2, 3]})
    out = add_rul(df)
    assert out["RUL"].tolist() == [2.0, 1.0, 0.0]


def test_add_rul_uncapped_and_capped():
    df = pd.DataFrame({"unit": [1, 1, 1], "cycle": [1, 100, 300]})
    out = add_rul(df, cap=125)
    assert out["RUL"].tolist() == [299.0, 200.0, 0.0]
    assert out["RUL_capped"].tolist() == [125.0, 125.0, 0.0]


def test_add_rul_sorts_by_unit_and_cycle():
    df = pd.DataFrame({"unit": [2, 1, 2], "cycle": [1, 2, 2]})
    out = add_rul(df)
    assert out[["unit", "cycle"]].values.tolist() == [[1, 2], [2, 1], [2, 2]]


def test_split_engines_sizes_disjoint_deterministic():
    ids = np.arange(1, 101)
    train_ids, val_ids = split_engines(ids, val_fraction=0.2, seed=0)
    assert train_ids.size == 80
    assert val_ids.size == 20
    assert set(train_ids).isdisjoint(set(val_ids))
    assert set(train_ids.tolist()) | set(val_ids.tolist()) == set(range(1, 101))

    train_again, val_again = split_engines(ids, 0.2, 0)
    assert np.array_equal(train_ids, train_again)
    assert np.array_equal(val_ids, val_again)

    _, val_other = split_engines(ids, 0.2, 1)
    assert not np.array_equal(val_ids, val_other)


@pytest.mark.parametrize("bad_fraction", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_bad_fraction(bad_fraction):
    with pytest.raises(ValueError):
        split_engines(np.arange(1, 11), val_fraction=bad_fraction)


def test_assert_disjoint_raises_on_overlap():
    assert_disjoint([1, 2, 3], [4, 5])  # no raise
    with pytest.raises(ValueError):
        assert_disjoint([1, 2, 3], [3, 4])


def test_engine_ids_sorted_unique():
    df = pd.DataFrame({"unit": [3, 1, 2, 1]})
    assert engine_ids(df).tolist() == [1, 2, 3]


def test_load_raw_real(have_real_data, raw_dir):
    if not have_real_data:
        pytest.skip("real FD001 data not present")
    train_df, test_df, test_rul = load_raw("FD001", raw_dir)
    assert list(train_df.columns) == COLUMNS
    assert train_df.shape[1] == 26
    assert engine_ids(train_df).tolist() == list(range(1, 101))
    assert engine_ids(test_df).size == 100
    assert len(test_rul) == 100
    assert test_rul.index.tolist() == list(range(1, 101))


def test_build_train_val_real_no_overlap(have_real_data, raw_dir):
    if not have_real_data:
        pytest.skip("real FD001 data not present")
    train_split, val_split, meta = build_train_val("FD001", raw_dir, cap=125, seed=0)
    assert meta["n_train_engines"] == 80
    assert meta["n_val_engines"] == 20
    assert set(meta["train_ids"]).isdisjoint(meta["val_ids"])
    assert {"RUL", "RUL_capped"}.issubset(train_split.columns)
    # cap applied for training labels, uncapped RUL retained for metrics
    assert train_split["RUL_capped"].max() <= 125
    assert train_split["RUL"].max() > 125
    assert set(engine_ids(train_split)).isdisjoint(set(engine_ids(val_split)))


def test_add_rul_zero_on_last_cycle_real(have_real_data, raw_dir):
    if not have_real_data:
        pytest.skip("real FD001 data not present")
    train_df, _, _ = load_raw("FD001", raw_dir)
    out = add_rul(train_df)
    min_rul_per_engine = out.groupby("unit")["RUL"].min()
    assert (min_rul_per_engine == 0).all()


def test_official_test_rul_only_on_last_cycle(have_real_data, raw_dir):
    if not have_real_data:
        pytest.skip("real FD001 data not present")
    frame, test_rul = load_official_test("FD001", raw_dir)
    assert len(test_rul) == 100
    assert int(frame["RUL"].notna().sum()) == 100
    is_last = frame["cycle"] == frame.groupby("unit")["cycle"].transform("max")
    assert frame.loc[is_last, "RUL"].notna().all()
    assert frame.loc[~is_last, "RUL"].isna().all()
