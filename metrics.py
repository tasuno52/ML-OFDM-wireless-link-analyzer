import numpy as np


def calculate_ber(tx_bits, rx_bits):
    min_len = min(len(tx_bits), len(rx_bits))
    return np.mean(tx_bits[:min_len] != rx_bits[:min_len])


def calculate_evm(tx_symbols, rx_symbols):
    min_len = min(len(tx_symbols), len(rx_symbols))

    tx_symbols = tx_symbols[:min_len]
    rx_symbols = rx_symbols[:min_len]

    error = rx_symbols - tx_symbols

    evm_rms = np.sqrt(
        np.mean(np.abs(error) ** 2) / np.mean(np.abs(tx_symbols) ** 2)
    )

    return evm_rms * 100