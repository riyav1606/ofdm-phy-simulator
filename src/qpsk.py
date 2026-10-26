"""
QPSK Modulation and Demodulation Module
Implements Gray-coded Quadrature Phase Shift Keying (QPSK) mapping,
minimum Euclidean distance hard demapping, and soft LLR calculation.
"""

import numpy as np

# Normalization factor so that average symbol energy E_s = 1.0
NORM_FACTOR = 1.0 / np.sqrt(2.0)

def modulate_qpsk(bits: np.ndarray) -> np.ndarray:
    """
    Modulate binary bits into Gray-coded QPSK complex symbols.
    Mapping:
      b0=0, b1=0 -> (+1 + 1j) / sqrt(2)
      b0=0, b1=1 -> (-1 + 1j) / sqrt(2)
      b0=1, b1=1 -> (-1 - 1j) / sqrt(2)
      b0=1, b1=0 -> (+1 - 1j) / sqrt(2)
    Adjacent symbols differ by exactly 1 bit (Gray code).
    """
    bits = np.asarray(bits, dtype=int)
    if len(bits) % 2 != 0:
        raise ValueError("Number of bits must be even for QPSK.")
    
    b0 = bits[0::2]
    b1 = bits[1::2]
    
    # Gray-coded constellation calculation
    # Let I and Q be mapped such that adjacent constellation points differ by 1 bit:
    I = np.where(b0 == 0, 1.0, -1.0) * NORM_FACTOR
    Q = np.where(b1 == 0, 1.0, -1.0) * NORM_FACTOR
    
    return I + 1j * Q

def demodulate_qpsk(symbols: np.ndarray) -> np.ndarray:
    """
    Demodulate complex QPSK symbols to hard bits using maximum likelihood
    minimum Euclidean distance decision boundaries (sign of I and Q).
    """
    symbols = np.asarray(symbols, dtype=complex)
    
    # For Gray mapping with I = +1 if b0=0 else -1, and Q = +1 if b1=0 else -1:
    b0 = (np.real(symbols) < 0).astype(int)
    b1 = (np.imag(symbols) < 0).astype(int)
    
    bits = np.empty(2 * len(symbols), dtype=int)
    bits[0::2] = b0
    bits[1::2] = b1
    return bits

def calculate_llr(symbols: np.ndarray, noise_variance: float) -> np.ndarray:
    """
    Calculate soft Log-Likelihood Ratios (LLR) for channel decoders (e.g., LDPC/Polar).
    LLR(b) = ln( P(b=0 | y) / P(b=1 | y) )
    For AWGN with variance sigma^2:
      LLR(b0) = (2 * sqrt(2) / sigma^2) * Re(y)
      LLR(b1) = (2 * sqrt(2) / sigma^2) * Im(y)
    """
    scale = (2.0 * np.sqrt(2.0)) / noise_variance
    llr0 = scale * np.real(symbols)
    llr1 = scale * np.imag(symbols)
    
    llrs = np.empty(2 * len(symbols), dtype=float)
    llrs[0::2] = llr0
    llrs[1::2] = llr1
    return llrs
