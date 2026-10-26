"""
OFDM Transceiver Module
Implements Subcarrier Mapping, 64-point IFFT/FFT modulation,
and Cyclic Prefix (CP) insertion / removal.
"""

import numpy as np

class OFDMSystem:
    def __init__(self, n_fft: int = 64, n_data: int = 48, n_cp: int = 16):
        """
        Initialize OFDM system parameters.
        Default configuration models IEEE 802.11a / standard baseline:
          n_fft = 64 subcarriers
          n_data = 48 active data subcarriers
          n_cp = 16 samples (25% CP overhead)
          DC null at index 0, 15 guard subcarriers.
        """
        self.n_fft = n_fft
        self.n_data = n_data
        self.n_cp = n_cp
        self.symbol_len = n_fft + n_cp
        
        # Subcarrier mapping:
        # Indices: [1..24] (positive frequencies) and [40..63] (negative frequencies)
        # DC null: index 0
        # Guard band nulls: indices [25..39] (15 subcarriers)
        self.data_indices = np.concatenate([
            np.arange(1, 25),
            np.arange(40, 64)
        ])
        assert len(self.data_indices) == self.n_data, "Data subcarriers mismatch!"

    def subcarrier_mapping(self, data_symbols: np.ndarray) -> np.ndarray:
        """
        Map data symbols into frequency-domain OFDM grid with DC and guard nulls.
        Shape: (num_ofdm_symbols, n_fft)
        """
        n_symbols = len(data_symbols) // self.n_data
        grid = np.zeros((n_symbols, self.n_fft), dtype=complex)
        reshaped_data = data_symbols[:n_symbols * self.n_data].reshape(n_symbols, self.n_data)
        grid[:, self.data_indices] = reshaped_data
        return grid

    def modulate(self, freq_grid: np.ndarray) -> np.ndarray:
        """
        Perform IFFT and insert Cyclic Prefix (CP).
        Uses 'ortho' norm to preserve Parseval energy conservation:
        sum(|x[n]|^2) == sum(|X[k]|^2)
        """
        # IFFT across each OFDM symbol (axis=1)
        time_domain = np.fft.ifft(freq_grid, axis=1, norm="ortho")
        
        # Cyclic Prefix: copy the last n_cp samples and prepend to the front
        cp = time_domain[:, -self.n_cp:]
        ofdm_symbols = np.concatenate([cp, time_domain], axis=1)
        
        # Serialize into continuous time-domain stream
        return ofdm_symbols.flatten()

    def demodulate(self, rx_stream: np.ndarray, num_ofdm_symbols: int) -> np.ndarray:
        """
        Remove Cyclic Prefix, perform FFT, and extract data subcarriers.
        """
        expected_len = num_ofdm_symbols * self.symbol_len
        rx_stream = rx_stream[:expected_len]
        
        # Reshape into symbols
        rx_blocks = rx_stream.reshape(num_ofdm_symbols, self.symbol_len)
        
        # Strip cyclic prefix
        time_domain = rx_blocks[:, self.n_cp:]
        
        # FFT to convert time domain back to frequency domain
        freq_grid = np.fft.fft(time_domain, axis=1, norm="ortho")
        
        # Extract active data subcarriers
        data_symbols = freq_grid[:, self.data_indices].flatten()
        return data_symbols
