"""
Wireless Channel Impairments Module
Implements Additive White Gaussian Noise (AWGN) and multipath fading.
"""

import numpy as np

def add_awgn_noise(signal: np.ndarray, snr_db: float, bits_per_symbol: int = 2,
                   code_rate: float = 1.0, cp_ratio: float = 16.0 / 64.0) -> tuple[np.ndarray, float]:
    """
    Add complex zero-mean circular Additive White Gaussian Noise (AWGN).
    Calibrated accurately to Eb/N0.
    
    Formula:
      Eb/N0 (linear) = 10^(snr_db / 10)
      Noise variance per complex dimension:
      sigma^2 = (Es / (bits_per_symbol * code_rate * (1 + cp_ratio))) / (Eb/N0)
      For normalized uncoded baseband QPSK: Es = 1.0
    
    Returns:
      (noisy_signal, noise_variance)
    """
    eb_no_linear = 10.0 ** (snr_db / 10.0)
    
    # Calculate noise variance per complex symbol
    # Es / N0 = (Eb / N0) * bits_per_symbol * code_rate
    # Total complex noise variance = N0
    n0 = 1.0 / (eb_no_linear * bits_per_symbol * code_rate)
    
    # Complex circular AWGN: w = (w_real + 1j * w_imag) / sqrt(2) * sqrt(n0)
    noise = (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)) * np.sqrt(n0 / 2.0)
    
    return signal + noise, n0

def apply_multipath_channel(signal: np.ndarray, channel_taps: np.ndarray) -> np.ndarray:
    """
    Convolve signal with discrete multipath channel impulse response h.
    """
    return np.convolve(signal, channel_taps, mode="same")
