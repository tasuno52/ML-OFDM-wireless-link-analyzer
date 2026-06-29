import os
import numpy as np
import matplotlib.pyplot as plt

from modulation import qpsk_mod, qpsk_demod, qam16_mod, qam64_mod
from ofdm import ofdm_modulate, ofdm_demodulate
from metrics import calculate_ber, calculate_evm

from channel import (
    add_awgn,
    rayleigh_channel,
    equalize_flat_fading,
    multipath_channel,
    get_channel_frequency_response
)


def plot_constellation(symbols, title, filename):
    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.scatter(np.real(symbols), np.imag(symbols), s=8, alpha=0.6)
    plt.grid(True)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.title(title)
    plt.axis("equal")
    plt.savefig(f"figures/{filename}", dpi=300)
    plt.show()


def test_constellations():
    bits_qpsk = np.random.randint(0, 2, 2000)
    bits_16qam = np.random.randint(0, 2, 4000)
    bits_64qam = np.random.randint(0, 2, 6000)

    qpsk_symbols = qpsk_mod(bits_qpsk)
    qam16_symbols = qam16_mod(bits_16qam)
    qam64_symbols = qam64_mod(bits_64qam)

    plot_constellation(qpsk_symbols, "QPSK Constellation", "constellation_qpsk.png")
    plot_constellation(qam16_symbols, "16-QAM Constellation", "constellation_16qam.png")
    plot_constellation(qam64_symbols, "64-QAM Constellation", "constellation_64qam.png")


def test_ofdm_no_noise():
    bits = np.random.randint(0, 2, 6400)

    tx_symbols = qpsk_mod(bits)
    tx_signal = ofdm_modulate(tx_symbols)
    rx_symbols = ofdm_demodulate(tx_signal)
    rx_bits = qpsk_demod(rx_symbols)

    ber = calculate_ber(bits, rx_bits)

    print("OFDM no-noise BER:", ber)


def test_ofdm_with_awgn():
    snr_db = 10

    bits = np.random.randint(0, 2, 6400)

    tx_symbols = qpsk_mod(bits)
    tx_signal = ofdm_modulate(tx_symbols)
    rx_signal = add_awgn(tx_signal, snr_db)
    rx_symbols = ofdm_demodulate(rx_signal)
    rx_bits = qpsk_demod(rx_symbols)

    ber = calculate_ber(bits, rx_bits)
    evm = calculate_evm(tx_symbols, rx_symbols)

    print("\nAWGN OFDM Test")
    print("SNR:", snr_db, "dB")
    print("BER:", ber)
    print("EVM:", evm, "%")

    plot_constellation(
        rx_symbols,
        f"Received QPSK OFDM Constellation at {snr_db} dB SNR",
        "received_qpsk_awgn_10db.png"
    )


def test_rayleigh_channel():
    snr_db = 15

    bits = np.random.randint(0, 2, 6400)

    tx_symbols = qpsk_mod(bits)
    tx_signal = ofdm_modulate(tx_symbols)

    rx_faded, h = rayleigh_channel(tx_signal)
    rx_noisy = add_awgn(rx_faded, snr_db)
    rx_equalized = equalize_flat_fading(rx_noisy, h)

    rx_symbols = ofdm_demodulate(rx_equalized)
    rx_bits = qpsk_demod(rx_symbols)

    ber = calculate_ber(bits, rx_bits)
    evm = calculate_evm(tx_symbols, rx_symbols)

    print("\nRayleigh OFDM Test")
    print("SNR:", snr_db, "dB")
    print("Channel coefficient h:", h)
    print("BER:", ber)
    print("EVM:", evm, "%")

    plot_constellation(
        rx_symbols,
        f"QPSK OFDM over Rayleigh + AWGN at {snr_db} dB",
        "received_qpsk_rayleigh_15db.png"
    )


def test_multipath_ofdm_equalization():
    os.makedirs("figures", exist_ok=True)

    n_fft = 64
    n_cp = 16
    snr_db = 20

    bits = np.random.randint(0, 2, 6400)

    tx_symbols = qpsk_mod(bits)
    tx_signal = ofdm_modulate(tx_symbols, n_fft=n_fft, n_cp=n_cp)

    rx_multipath, taps = multipath_channel(tx_signal)
    rx_noisy = add_awgn(rx_multipath, snr_db)

    rx_symbols_no_eq = ofdm_demodulate(rx_noisy, n_fft=n_fft, n_cp=n_cp)

    h_freq = get_channel_frequency_response(taps, n_fft=n_fft)

    num_ofdm_symbols = len(rx_symbols_no_eq) // n_fft

    rx_symbols_no_eq = rx_symbols_no_eq[:num_ofdm_symbols * n_fft]
    tx_symbols_trimmed = tx_symbols[:num_ofdm_symbols * n_fft]
    bits_trimmed = bits[:num_ofdm_symbols * n_fft * 2]

    rx_matrix = rx_symbols_no_eq.reshape(num_ofdm_symbols, n_fft)

    rx_equalized_matrix = rx_matrix / (h_freq + 1e-12)
    rx_symbols_eq = rx_equalized_matrix.flatten()

    rx_bits_no_eq = qpsk_demod(rx_symbols_no_eq)
    rx_bits_eq = qpsk_demod(rx_symbols_eq)

    ber_no_eq = calculate_ber(bits_trimmed, rx_bits_no_eq)
    ber_eq = calculate_ber(bits_trimmed, rx_bits_eq)

    evm_no_eq = calculate_evm(tx_symbols_trimmed, rx_symbols_no_eq)
    evm_eq = calculate_evm(tx_symbols_trimmed, rx_symbols_eq)

    print("\nMultipath OFDM Test")
    print("SNR:", snr_db, "dB")
    print("Channel taps:", taps)
    print("BER without equalization:", ber_no_eq)
    print("BER with equalization:", ber_eq)
    print("EVM without equalization:", evm_no_eq, "%")
    print("EVM with equalization:", evm_eq, "%")

    plot_constellation(
        rx_symbols_no_eq,
        "QPSK OFDM over Multipath Channel Before Equalization",
        "multipath_constellation_before_equalization.png"
    )

    plot_constellation(
        rx_symbols_eq,
        "QPSK OFDM over Multipath Channel After Equalization",
        "multipath_constellation_after_equalization.png"
    )

    plt.figure()
    plt.stem(np.arange(len(taps)), np.abs(taps))
    plt.xlabel("Tap index")
    plt.ylabel("Magnitude")
    plt.title("Multipath Channel Impulse Response")
    plt.grid(True)
    plt.savefig("figures/multipath_impulse_response.png", dpi=300)
    plt.show()

    plt.figure()
    plt.plot(20 * np.log10(np.abs(h_freq) + 1e-12))
    plt.xlabel("OFDM Subcarrier Index")
    plt.ylabel("Magnitude Response (dB)")
    plt.title("Channel Frequency Response Across OFDM Subcarriers")
    plt.grid(True)
    plt.savefig("figures/multipath_frequency_response.png", dpi=300)
    plt.show()


def ber_vs_snr():
    os.makedirs("figures", exist_ok=True)

    snr_values = np.arange(0, 31, 2)
    ber_values = []
    evm_values = []

    for snr_db in snr_values:
        bits = np.random.randint(0, 2, 64000)

        tx_symbols = qpsk_mod(bits)
        tx_signal = ofdm_modulate(tx_symbols)
        rx_signal = add_awgn(tx_signal, snr_db)
        rx_symbols = ofdm_demodulate(rx_signal)
        rx_bits = qpsk_demod(rx_symbols)

        ber = calculate_ber(bits, rx_bits)
        evm = calculate_evm(tx_symbols, rx_symbols)

        ber_values.append(ber)
        evm_values.append(evm)

        print(f"SNR = {snr_db} dB | BER = {ber:.6f} | EVM = {evm:.2f}%")

    plt.figure()
    plt.semilogy(snr_values, ber_values, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("BER vs SNR for QPSK OFDM over AWGN")
    plt.grid(True)
    plt.savefig("figures/ber_vs_snr_qpsk_ofdm.png", dpi=300)
    plt.show()

    plt.figure()
    plt.plot(snr_values, evm_values, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("EVM (%)")
    plt.title("EVM vs SNR for QPSK OFDM over AWGN")
    plt.grid(True)
    plt.savefig("figures/evm_vs_snr_qpsk_ofdm.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    test_constellations()
    test_ofdm_no_noise()
    test_ofdm_with_awgn()
    test_rayleigh_channel()
    test_multipath_ofdm_equalization()
    ber_vs_snr()
    