import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from ofdm import OFDMSystem
from qpsk import modulate_qpsk, demodulate_qpsk

def test_ofdm_cyclic_prefix_structure():
    """Verify cyclic prefix is exact replica of the tail of the OFDM time block."""
    ofdm = OFDMSystem(n_fft=64, n_data=48, n_cp=16)
    tx_symbols = np.ones(48, dtype=complex)
    
    freq_grid = ofdm.subcarrier_mapping(tx_symbols)
    time_stream = ofdm.modulate(freq_grid)
    
    assert len(time_stream) == 80  # 64 + 16
    # CP is first 16 samples, should match the last 16 samples of the 64-sample IFFT block (samples 64..80)
    cp_part = time_stream[:16]
    tail_part = time_stream[64:80]
    assert np.allclose(cp_part, tail_part, atol=1e-10), "Cyclic prefix does not match symbol tail!"

def test_ofdm_noiseless_recovery():
    """Verify end-to-end OFDM transmission and reception without noise."""
    ofdm = OFDMSystem(n_fft=64, n_data=48, n_cp=16)
    bits = np.random.randint(0, 2, size=48 * 2 * 10)  # 10 OFDM symbols
    
    tx_symbols = modulate_qpsk(bits)
    freq_grid = ofdm.subcarrier_mapping(tx_symbols)
    tx_stream = ofdm.modulate(freq_grid)
    
    rx_symbols = ofdm.demodulate(tx_stream, num_ofdm_symbols=10)
    rx_bits = demodulate_qpsk(rx_symbols)
    
    assert np.array_equal(bits, rx_bits), "OFDM noiseless recovery failed!"
