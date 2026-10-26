"""
Top-level entry point for QPSK-OFDM Physical-Layer Baseband Simulation & Validation.
"""

import os
import sys

# Ensure src is in search path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from simulation import run_simulation, plot_and_save_results

def main():
    print("================================================================================")
    print("   64-SUBCARRIER QPSK-OFDM PHYSICAL-LAYER BASEBAND SIMULATOR")
    print("================================================================================")
    snr_range = np.arange(0.0, 11.0, 1.0)
    print(f"[*] Sweeping Eb/N0 across {snr_range[0]:.1f} dB to {snr_range[-1]:.1f} dB...")
    results = run_simulation(snr_range, num_ofdm_symbols=2500)
    
    plot_path = os.path.join(os.path.dirname(__file__), "plots", "ber_curve.png")
    plot_and_save_results(results, os.path.abspath(plot_path))
    
    print("================================================================================")
    print("   EMPIRICAL VALIDATION SUMMARY (VS THEORETICAL Q-FUNCTION)")
    print("================================================================================")
    for snr, sim_ber, theo_ber in zip(results["snr_db"], results["simulated_ber"], results["theoretical_ber"]):
        delta_pct = abs(sim_ber - theo_ber) / theo_ber * 100 if theo_ber > 0 else 0
        print(f"Eb/N0: {snr:4.1f} dB | Sim BER: {sim_ber:9.6e} | Theo BER: {theo_ber:9.6e} | Deviation: {delta_pct:5.2f}%")
    print("================================================================================")
    print(f"[+] Output BER validation curve saved to: {plot_path}")
    print("================================================================================")

if __name__ == "__main__":
    main()
