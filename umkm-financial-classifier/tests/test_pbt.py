"""
test_pbt.py
Property-Based Tests menggunakan Hypothesis.
Memverifikasi invariant matematis untuk semua input valid (model 4 kelas).
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

DATASET_PATH = os.path.join("data", "raw", "dataset_umkm.csv")
MODEL_PATH = os.path.join("models", "gnb_pipeline.pkl")

_PIPELINE = None


def get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        if not os.path.exists(MODEL_PATH):
            pytest.skip(f"Model tidak ditemukan: {MODEL_PATH}. Jalankan train.py.")
        from src.model import load_pipeline
        _PIPELINE = load_pipeline(MODEL_PATH)
    return _PIPELINE


def valid_input_strategy():
    return st.fixed_dictionaries({
        "Net_Profit_Margin":      st.floats(min_value=-35.0, max_value=45.0,  allow_nan=False, allow_infinity=False),
        "Burn_Rate_Ratio":        st.floats(min_value=0.43,  max_value=1.55,  allow_nan=False, allow_infinity=False),
        "Avg_Historical_Rating":  st.floats(min_value=1.0,   max_value=5.0,   allow_nan=False, allow_infinity=False),
        "Business_Tenure_Months": st.integers(min_value=3,   max_value=179),
        "Repeat_Order_Rate":      st.floats(min_value=2.0,   max_value=54.06, allow_nan=False, allow_infinity=False),
    })


# ── Property 1: Probabilitas 4 kelas berjumlah 1 ─────────────────────────────

@given(valid_input_strategy())
@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
def test_prob_sum_to_one(data):
    """∀ input valid: Σ probabilitas 4 kelas == 1.0 ± 1e-3"""
    from src.model import predict_single
    hasil = predict_single(get_pipeline(), data)
    total = sum(hasil["probabilitas"].values())
    assert abs(total - 1.0) < 1e-3, f"Probabilitas tidak berjumlah 1: {total}"
    assert hasil["label"] in {0, 1, 2, 3}


# ── Property 2: Label konsisten dengan argmax proba ──────────────────────────

@given(
    npm    = st.floats(min_value=-35.0, max_value=45.0,  allow_nan=False, allow_infinity=False),
    burn   = st.floats(min_value=0.43,  max_value=1.55,  allow_nan=False, allow_infinity=False),
    rating = st.floats(min_value=1.0,   max_value=5.0,   allow_nan=False, allow_infinity=False),
    tenure = st.integers(min_value=3,   max_value=179),
    ror    = st.floats(min_value=2.0,   max_value=54.06, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_label_consistent_with_argmax(npm, burn, rating, tenure, ror):
    """∀ input: predict(X) == argmax(predict_proba(X))"""
    import pandas as pd
    from src.preprocessing import FITUR_TERPILIH
    pipeline = get_pipeline()
    X = pd.DataFrame([[npm, burn, rating, tenure, ror]], columns=FITUR_TERPILIH)
    label = int(pipeline.predict(X)[0])
    proba = pipeline.predict_proba(X)[0]
    assert proba.shape == (4,)
    assert label == int(pipeline.classes_[int(np.argmax(proba))])


# ── Property 5: predict_dataframe (utilitas model) preserves row count ───────

@given(st.integers(min_value=1, max_value=500))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_predict_dataframe_preserves_row_count(n_rows):
    """∀ n: len(predict_dataframe(df_valid)) == n"""
    import pandas as pd
    from src.model import predict_dataframe

    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "Net_Profit_Margin":      rng.uniform(-35, 45, n_rows),
        "Burn_Rate_Ratio":        rng.uniform(0.43, 1.55, n_rows),
        "Avg_Historical_Rating":  rng.uniform(1.0, 5.0, n_rows),
        "Business_Tenure_Months": rng.integers(3, 179, n_rows),
        "Repeat_Order_Rate":      rng.uniform(2, 54, n_rows),
    })

    result = predict_dataframe(get_pipeline(), df)
    assert len(result) == n_rows, f"Row count berubah: {len(result)} != {n_rows}"


# ── Property: Validasi input tidak mutasi dict ────────────────────────────────

@given(valid_input_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_validate_input_no_mutation(data):
    """validate_input tidak boleh mengubah dict input."""
    from src.preprocessing import validate_input
    import copy
    original = copy.deepcopy(data)
    validate_input(data)
    assert data == original
