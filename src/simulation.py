"""
End-to-End Monte Carlo Simulation Pipeline
Sweeps Eb/N0 from 0 to 10 dB, computes empirical Bit Error Rate (BER),
compares against theoretical Q-function curve, and saves verification plot.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from math import erfc

from qpsk import modulate_qpsk, demodulate_qpsk
from ofdm import OFDMSystem
from channel import add_awgn_noise

def theoretical_qpsk_ber(eb_no_db: np.ndarray) -> np.ndarray:
    """
    Theoretical BER for QPSK over AWGN:
      Pb = Q( sqrt(2 * Eb/N0) ) = 0.5 * erfc( sqrt(Eb/N0) )
    """
    eb_no_lin = 10.0 ** (eb_no_db / 10.0)
    return np.array([0.5 * erfc(np.sqrt(x)) for x in eb_no_lin])

def run_simulation(snr_db_range: np.ndarray, num_ofdm_symbols: int = 2000, seed: int = 42) -> dict:
    """
    Execute Monte Carlo physical layer simulation across specified SNR range.
    """
    np.random.seed(seed)
    ofdm = OFDMSystem(n_fft=64, n_data=48, n_cp=16)
    
    bits_per_symbol = 2  # QPSK
    total_bits = num_ofdm_symbols * ofdm.n_data * bits_per_symbol
    tx_bits = np.random.randint(0, 2, size=total_bits)
    
    # 1. Modulation
    tx_symbols = modulate_qpsk(tx_bits)
    
    # 2. OFDM Transmitter
    freq_grid = ofdm.subcarrier_mapping(tx_symbols)
    tx_time_stream = ofdm.modulate(freq_grid)
    
    ber_results = []
    
    for snr_db in snr_db_range:
        # 3. Channel (AWGN)
        rx_time_stream, n0 = add_awgn_noise(tx_time_stream, snr_db, bits_per_symbol=bits_per_symbol)
        
        # 4. OFDM Receiver
        rx_symbols = ofdm.demodulate(rx_time_stream, num_ofdm_symbols)
        
        # 5. Demodulation
        rx_bits = demodulate_qpsk(rx_symbols)
        
        # 6. BER Calculation
        bit_errors = np.sum(tx_bits != rx_bits)
        ber = bit_errors / total_bits
        ber_results.append(ber)
        print(f"SNR: {snr_db:4.1f} dB | Total Bits: {total_bits:,} | Bit Errors: {bit_errors:5d} | BER: {ber:.6e}")
        
    return {
        "snr_db": snr_db_range,
        "simulated_ber": np.array(ber_results),
        "theoretical_ber": theoretical_qpsk_ber(snr_db_range),
        "total_bits_per_point": total_bits
    }

def plot_and_save_results(results: dict, output_path: str):
    """
    Plot empirical vs theoretical BER on semilog scale.
    """
    plt.figure(figsize=(9, 6), dpi=150)
    plt.semilogy(results["snr_db"], results["theoretical_ber"], "k-", linewidth=2.0, label="Theoretical QPSK (AWGN)")
    plt.semilogy(results["snr_db"], results["simulated_ber"], "ro--", markersize=7, linewidth=1.5, label="Simulated 64-OFDM / QPSK")
    
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.xlabel("Eb/N0 (dB)", fontsize=12, fontweight="bold")
    plt.ylabel("Bit Error Rate (BER)", fontsize=12, fontweight="bold")
    plt.title("QPSK-OFDM Physical Layer: Empirical vs Theoretical BER Performance", fontsize=13, fontweight="bold")
    plt.legend(fontsize=11)
    plt.ylim([1e-6, 1.0])
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved BER curve to: {output_path}")

if __name__ == "__main__":
    snr_range = np.arange(0.0, 11.0, 1.0)
    print("=== Running QPSK-OFDM Baseband Simulation ===")
    results = run_simulation(snr_range, num_ofdm_symbols=2500)
    
    plot_path = os.path.join(os.path.dirname(__file__), "..", "plots", "ber_curve.png")
    plot_and_save_results(results, os.path.abspath(plot_path))
    print("=== Simulation Completed Successfully ===")
