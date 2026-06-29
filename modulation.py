import numpy as np
def qpsk_mod(bits):
    bits = bits[:len(bits)-len(bits)%2]
    bits = bits.reshape((-1, 2))    

    mapping = {
        (0, 0): 1 + 1j,
        (0, 1): -1 + 1j,
        (1, 0): 1 - 1j,
        (1, 1): -1 - 1j
    }
    symbols = np.array([mapping[tuple(b)] for b in bits])
    return symbols / np.sqrt(2)

def qpsk_demod(symbols):
    bits = []

    for s in symbols:
        real = np.real(s)
        imag = np.imag(s)

        if real >= 0 and imag >= 0:
            bits.extend([0, 0])
        elif real < 0 and imag >= 0:
            bits.extend([0, 1])
        elif real < 0 and imag < 0:
            bits.extend([1, 1])
        else:
            bits.extend([1, 0])

    return np.array(bits)


def qam16_mod(bits):
    bits = bits[:len(bits) - len(bits) % 4]
    bits = bits.reshape((-1, 4))

    levels = {
        (0, 0): -3,
        (0, 1): -1,
        (1, 1): 1,
        (1, 0): 3,
    }

    symbols = []

    for b in bits:
        i = levels[tuple(b[:2])]
        q = levels[tuple(b[2:])]
        symbols.append(i + 1j * q)

    return np.array(symbols) / np.sqrt(10)


def qam64_mod(bits):
    bits = bits[:len(bits) - len(bits) % 6]
    bits = bits.reshape((-1, 6))

    levels = {
        (0, 0, 0): -7,
        (0, 0, 1): -5,
        (0, 1, 1): -3,
        (0, 1, 0): -1,
        (1, 1, 0): 1,
        (1, 1, 1): 3,
        (1, 0, 1): 5,
        (1, 0, 0): 7,
    }

    symbols = []

    for b in bits:
        i = levels[tuple(b[:3])]
        q = levels[tuple(b[3:])]
        symbols.append(i + 1j * q)

    return np.array(symbols) / np.sqrt(42)
