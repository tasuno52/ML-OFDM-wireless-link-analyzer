import numpy as np


def add_awgn(signal, snr_db):
    signal_power = np.mean(np.abs(signal) ** 2)

    snr_linear = 10 ** (snr_db / 10)

    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )

    return signal + noise


def rayleigh_channel(signal):
    h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    rx_signal = h * signal
    return rx_signal, h


def equalize_flat_fading(rx_signal, h):
    return rx_signal / h


def multipath_channel(signal, taps=None):
    """
    Simulates a frequency-selective multipath channel.

    signal: transmitted complex baseband signal
    taps: complex channel impulse response

    returns:
        rx_signal: signal after multipath channel
        taps: channel impulse response
    """

    if taps is None:
        taps = np.array([
            1.0 + 0.0j,
            0.5 * np.exp(1j * np.pi / 4),
            0.3 * np.exp(1j * np.pi / 2),
            0.15 * np.exp(1j * np.pi / 3),
        ])

    rx_signal = np.convolve(signal, taps, mode="same")

    return rx_signal, taps


def get_channel_frequency_response(taps, n_fft=64):
    """
    Converts the channel impulse response into a frequency response.
    This tells us how each OFDM subcarrier is affected.
    """

    h_freq = np.fft.fft(taps, n=n_fft)

    return h_freq
