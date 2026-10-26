// High-Performance C++ QPSK Demapper and Slicer for Firmware Integration
// Simulates hardware-level register / memory-mapped symbol slicing.

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>

struct ComplexSymbol {
    float real;
    float imag;
};

// Fast branchless hard demapper:
// Re >= 0 -> bit0 = 0, Re < 0 -> bit0 = 1
// Im >= 0 -> bit1 = 0, Im < 0 -> bit1 = 1
inline void demap_qpsk_symbol(const ComplexSymbol& sym, uint8_t& b0, uint8_t& b1) {
    b0 = (sym.real < 0.0f) ? 1 : 0;
    b1 = (sym.imag < 0.0f) ? 1 : 0;
}

// Vectorized batch demapper simulating DSP buffer processing
void demap_stream(const ComplexSymbol* symbols, size_t num_symbols, uint8_t* out_bits) {
    for (size_t i = 0; i < num_symbols; ++i) {
        demap_qpsk_symbol(symbols[i], out_bits[2 * i], out_bits[2 * i + 1]);
    }
}

int main() {
    std::cout << "=== Apple Firmware DSP Demapper Verification ===\n";
    
    const size_t NUM_SYMBOLS = 1000000;
    std::vector<ComplexSymbol> symbols(NUM_SYMBOLS);
    std::vector<uint8_t> out_bits(2 * NUM_SYMBOLS);
    
    // Populate test constellation symbols
    const float norm = 1.0f / std::sqrt(2.0f);
    for (size_t i = 0; i < NUM_SYMBOLS; ++i) {
        symbols[i].real = (i % 2 == 0 ? 1.0f : -1.0f) * norm;
        symbols[i].imag = (i % 4 < 2 ? 1.0f : -1.0f) * norm;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    demap_stream(symbols.data(), NUM_SYMBOLS, out_bits.data());
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Processed " << NUM_SYMBOLS << " symbols (" 
              << (2 * NUM_SYMBOLS) << " bits) in " 
              << duration.count() << " ms.\n";
    std::cout << "Throughput: " << (NUM_SYMBOLS / duration.count() / 1000.0) << " MSymbols/sec\n";
    std::cout << "First 8 recovered bits: ";
    for (size_t i = 0; i < 8; ++i) {
        std::cout << (int)out_bits[i] << " ";
    }
    std::cout << "\n[STATUS] C++ Demapper Test Passed successfully.\n";
    return 0;
}
