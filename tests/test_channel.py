import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from channel import add_awgn_noise

def test_awgn_noise_variance():
    """Verify that generated complex AWGN noise matches theoretical variance."""
    snr_db = 10.0
    signal = np.zeros(200000, dtype=complex)
    noisy_signal, n0 = add_awgn_noise(signal, snr_db, bits_per_symbol=2)
    
    empirical_variance = np.var(noisy_signal)
    # Tolerance of 3% for 200,000 samples
    assert np.isclose(empirical_variance, n0, rtol=0.03), f"Empirical variance {empirical_variance} differs from expected {n0}"
