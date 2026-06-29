import numpy as np


def ofdm_modulate(data_symbols, n_fft=64, n_cp=16):
    num_ofdm_symbols = len(data_symbols) // n_fft
    data_symbols = data_symbols[:num_ofdm_symbols * n_fft]

    data_matrix = data_symbols.reshape((num_ofdm_symbols, n_fft))

    time_domain = np.fft.ifft(data_matrix, axis=1)

    cyclic_prefix = time_domain[:, -n_cp:]

    ofdm_symbols = np.concatenate([cyclic_prefix, time_domain], axis=1)

    return ofdm_symbols.flatten()


def ofdm_demodulate(rx_signal, n_fft=64, n_cp=16):
    symbol_length = n_fft + n_cp
    num_symbols = len(rx_signal) // symbol_length

    rx_signal = rx_signal[:num_symbols * symbol_length]

    rx_matrix = rx_signal.reshape((num_symbols, symbol_length))

    rx_no_cp = rx_matrix[:, n_cp:]

    freq_domain = np.fft.fft(rx_no_cp, axis=1)

    return freq_domain.flatten()