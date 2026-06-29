# ML-Based OFDM Wireless Link Analyzer

Python project combining **OFDM simulation**, **RF channel modeling**, **constellation analysis**, and **machine learning** for wireless signal classification.

## Overview

This project implements an OFDM/LTE-inspired wireless link simulator. It generates QPSK, 16-QAM, and 64-QAM IQ signals, transmits them through impaired wireless channels, computes BER/EVM metrics, and applies ML algorithms to analyze received constellations.

## Features

- QPSK, 16-QAM, and 64-QAM modulation
- OFDM modulation/demodulation using IFFT/FFT
- Cyclic prefix insertion/removal
- AWGN channel
- Rayleigh fading channel
- Frequency-selective multipath channel
- Frequency-domain equalization
- BER and EVM calculation
- K-Means constellation clustering
- Random Forest and SVM modulation classification

## Signal Chain

```text
Random bits
→ QAM modulation
→ OFDM IFFT
→ Cyclic prefix
→ Wireless channel
→ FFT receiver
→ Equalization
→ BER/EVM analysis
→ ML classification
