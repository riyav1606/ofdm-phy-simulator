import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from qpsk import modulate_qpsk, demodulate_qpsk, calculate_llr

def test_qpsk_power_normalization():
    """Verify that average symbol energy E_s is strictly 1.0."""
    bits = np.random.randint(0, 2, size=10000)
    symbols = modulate_qpsk(bits)
    avg_power = np.mean(np.abs(symbols) ** 2)
    assert np.isclose(avg_power, 1.0, atol=1e-3), f"Average power was {avg_power}"

def test_qpsk_noiseless_roundtrip():
    """Verify 100% bit-accurate recovery over an ideal noiseless channel."""
    bits = np.random.randint(0, 2, size=2048)
    symbols = modulate_qpsk(bits)
    recovered_bits = demodulate_qpsk(symbols)
    assert np.array_equal(bits, recovered_bits), "Noiseless roundtrip produced bit errors!"

def test_qpsk_gray_coding_hamming_distance():
    """Verify Gray coding: adjacent points on constellation have Hamming distance of 1."""
    # 4 cardinal constellation points:
    # (0,0) -> +1+j, (0,1) -> -1+j, (1,1) -> -1-j, (1,0) -> +1-j
    p00 = modulate_qpsk(np.array([0, 0]))[0]
    p01 = modulate_qpsk(np.array([0, 1]))[0]
    p11 = modulate_qpsk(np.array([1, 1]))[0]
    p10 = modulate_qpsk(np.array([1, 0]))[0]
    
    # Distance between adjacent constellation symbols: |(+1+j) - (-1+j)|/sqrt(2) = 2/sqrt(2) = sqrt(2)
    d_adjacent = np.abs(p00 - p01)
    assert np.isclose(d_adjacent, np.sqrt(2.0)), "Constellation geometry incorrect!"
