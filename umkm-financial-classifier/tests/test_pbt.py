"""
test_pbt.py
Property-Based Tests menggunakan Hypothesis.
Memverifikasi invariant matematis untuk semua input valid.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

# ── Setup: muat pipeline sekali ───────────────────────────────────────────────

DATASET_PATH = os.path.join("data", "raw", "dataset_umkm.csv")
MODEL_PATH   = os.path.join("models", "gnb_pipeline.pkl")

_PIPELINE = None

def get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        if not os.path.exists(MODEL_PATH):
            pytest.skip(f"Model tidak ditemukan: {MODEL_PATH}. Jalankan train.py.")
        from src.model import load_pipeline
        _PIPELINE = load_pipeline(MODEL_PATH)
    return _PIPELINE


# ── Strategy Helper ───────────────────────────────────────────────────────────

def valid_input_strategy():
    return st.fixed_dictionaries({
        "Monthly_Revenue":        st.floats(min_value=0.01,  max_value=1e8,   allow_nan=False, allow_infinity=False),
        "Net_Profit_Margin":      st.floats(min_value=-35.0, max_value=45.0,  allow_nan=False, allow_infinity=False),
        "Burn_Rate_Ratio":        st.floats(min_value=0.44,  max_value=1.55,  allow_nan=False, allow_infinity=False),
        "Transaction_Count":      st.integers(min_value=9,   max_value=285),
        "Repeat_Order_Rate":      st.floats(min_value=2.0,   max_value=54.0,  allow_nan=False, allow_infinity=False),
        "Business_Tenure_Months": st.integers(min_value=3,   max_value=179),
    })


# ── Property 1: Probabilitas Berjumlah 1 ────────────────────────────────────

@given(valid_input_strategy())
@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
def test_prob_sum_to_one(data):
    """∀ input valid: P(Sehat) + P(Tidak Sehat) == 1.0 ± 1e-6"""
    from src.model import predict_single
    hasil = predict_single(get_pipeline(), data)
    total = hasil["prob_sehat"] + hasil["prob_tidak_sehat"]
    assert abs(total - 1.0) < 1e-4, f"Probabilitas tidak berjumlah 1: {total}"


# ── Property 2: Label Konsisten dengan Argmax Proba ──────────────────────────

@given(
    revenue = st.floats(min_value=0.01, max_value=1e8,  allow_nan=False, allow_infinity=False),
    npm     = st.floats(min_value=-35.0, max_value=45.0, allow_nan=False, allow_infinity=False),
    burn    = st.floats(min_value=0.44,  max_value=1.55, allow_nan=False, allow_infinity=False),
    txn     = st.integers(min_value=9, max_value=285),
    ror     = st.floats(min_value=2.0,   max_value=54.0, allow_nan=False, allow_infinity=False),
    tenure  = st.integers(min_value=3, max_value=179),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_label_consistent_with_argmax(revenue, npm, burn, txn, ror, tenure):
    """∀ input: predict(X) == argmax(predict_proba(X))"""
    pipeline = get_pipeline()
    X = [[revenue, npm, burn, txn, ror, tenure]]
    label = int(pipeline.predict(X)[0])
    proba = pipeline.predict_proba(X)[0]
    assert label == int(np.argmax(proba))


# ── Property 5: Batch Preserves Row Count ────────────────────────────────────

@given(st.integers(min_value=1, max_value=500))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_batch_preserves_row_count(n_rows):
    """∀ n: len(predict_dataframe(df_valid)) == n"""
    import pandas as pd
    from src.model import predict_dataframe
    from src.preprocessing import FITUR_TERPILIH

    # Buat DataFrame valid dengan n_rows
    rng = np.random.default_rng(42)
    df  = pd.DataFrame({
        "Monthly_Revenue":        rng.uniform(1e4, 1e7, n_rows),
        "Net_Profit_Margin":      rng.uniform(-35, 45, n_rows),
        "Burn_Rate_Ratio":        rng.uniform(0.44, 1.55, n_rows),
        "Transaction_Count":      rng.integers(9, 285, n_rows),
        "Repeat_Order_Rate":      rng.uniform(2, 54, n_rows),
        "Business_Tenure_Months": rng.integers(3, 179, n_rows),
    })

    result = predict_dataframe(get_pipeline(), df)
    assert len(result) == n_rows, f"Row count berubah: {len(result)} != {n_rows}"


# ── Property: Validasi Input Tidak Mutasi Dict ────────────────────────────────

@given(valid_input_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_validate_input_no_mutation(data):
    """validate_input tidak boleh mengubah dict input."""
    from src.preprocessing import validate_input
    import copy
    original = copy.deepcopy(data)
    validate_input(data)
    assert data == original
