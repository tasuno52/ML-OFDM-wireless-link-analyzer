import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from modulation import qpsk_mod, qam16_mod, qam64_mod
from ofdm import ofdm_modulate, ofdm_demodulate
from channel import add_awgn, rayleigh_channel, equalize_flat_fading


os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


def plot_iq_points(symbols, title, filename, labels=None):
    plt.figure(figsize=(6, 6))

    if labels is None:
        plt.scatter(np.real(symbols), np.imag(symbols), s=8, alpha=0.6)
    else:
        plt.scatter(np.real(symbols), np.imag(symbols), c=labels, s=8, alpha=0.7)

    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(f"figures/{filename}", dpi=300)
    plt.show()


def generate_received_symbols(modulation, snr_db, channel_type, num_bits=12000):
    if modulation == "QPSK":
        bits_per_symbol = 2
        bits = np.random.randint(0, 2, num_bits - num_bits % bits_per_symbol)
        tx_symbols = qpsk_mod(bits)

    elif modulation == "16QAM":
        bits_per_symbol = 4
        bits = np.random.randint(0, 2, num_bits - num_bits % bits_per_symbol)
        tx_symbols = qam16_mod(bits)

    elif modulation == "64QAM":
        bits_per_symbol = 6
        bits = np.random.randint(0, 2, num_bits - num_bits % bits_per_symbol)
        tx_symbols = qam64_mod(bits)

    else:
        raise ValueError("Unsupported modulation")

    tx_signal = ofdm_modulate(tx_symbols)

    if channel_type == "AWGN":
        rx_signal = add_awgn(tx_signal, snr_db)

    elif channel_type == "Rayleigh":
        rx_faded, h = rayleigh_channel(tx_signal)
        rx_noisy = add_awgn(rx_faded, snr_db)
        rx_signal = equalize_flat_fading(rx_noisy, h)

    else:
        raise ValueError("Unsupported channel type")

    rx_symbols = ofdm_demodulate(rx_signal)

    min_len = min(len(tx_symbols), len(rx_symbols))
    tx_symbols = tx_symbols[:min_len]
    rx_symbols = rx_symbols[:min_len]

    return tx_symbols, rx_symbols


def extract_constellation_features(symbols):
    i = np.real(symbols)
    q = np.imag(symbols)
    amplitude = np.abs(symbols)
    phase = np.angle(symbols)

    features = {
        "i_mean": np.mean(i),
        "q_mean": np.mean(q),
        "i_std": np.std(i),
        "q_std": np.std(q),
        "i_var": np.var(i),
        "q_var": np.var(q),
        "iq_cov": np.cov(i, q)[0, 1],
        "amplitude_mean": np.mean(amplitude),
        "amplitude_std": np.std(amplitude),
        "amplitude_var": np.var(amplitude),
        "phase_mean": np.mean(phase),
        "phase_std": np.std(phase),
        "real_kurtosis_like": np.mean((i - np.mean(i)) ** 4) / (np.std(i) ** 4 + 1e-12),
        "imag_kurtosis_like": np.mean((q - np.mean(q)) ** 4) / (np.std(q) ** 4 + 1e-12),
        "papr_db": 10 * np.log10(np.max(amplitude ** 2) / (np.mean(amplitude ** 2) + 1e-12)),
    }

    return features


def run_kmeans_on_constellation(modulation="16QAM", snr_db=20, channel_type="AWGN"):
    tx_symbols, rx_symbols = generate_received_symbols(
        modulation=modulation,
        snr_db=snr_db,
        channel_type=channel_type
    )

    x = np.column_stack((np.real(rx_symbols), np.imag(rx_symbols)))

    n_clusters = {
        "QPSK": 4,
        "16QAM": 16,
        "64QAM": 64,
    }[modulation]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(x)

    plot_iq_points(
        rx_symbols,
        f"K-Means Clustering of {modulation} Constellation, SNR={snr_db} dB",
        f"kmeans_{modulation}_{snr_db}db.png",
        labels=cluster_labels
    )

    print(f"K-Means finished for {modulation}")
    print("Number of clusters:", n_clusters)


def generate_modulation_classification_dataset(num_examples_per_class=500):
    rows = []

    modulations = ["QPSK", "16QAM", "64QAM"]
    channel_types = ["AWGN", "Rayleigh"]

    for modulation in modulations:
        for example in range(num_examples_per_class):
            snr_db = np.random.uniform(5, 30)
            channel_type = np.random.choice(channel_types)

            _, rx_symbols = generate_received_symbols(
                modulation=modulation,
                snr_db=snr_db,
                channel_type=channel_type,
                num_bits=12000
            )

            features = extract_constellation_features(rx_symbols)
            features["snr_db"] = snr_db
            features["channel_type"] = channel_type
            features["modulation"] = modulation

            rows.append(features)

            if example % 100 == 0:
                print(f"{modulation}: generated {example}/{num_examples_per_class}")

    df = pd.DataFrame(rows)
    df.to_csv("data/constellation_ml_dataset.csv", index=False)

    print("Saved dataset to data/constellation_ml_dataset.csv")
    print(df.head())
    print(df["modulation"].value_counts())

    return df


def train_modulation_classifier():
    df = pd.read_csv("data/constellation_ml_dataset.csv")

    feature_columns = [
        "i_mean",
        "q_mean",
        "i_std",
        "q_std",
        "i_var",
        "q_var",
        "iq_cov",
        "amplitude_mean",
        "amplitude_std",
        "amplitude_var",
        "phase_mean",
        "phase_std",
        "real_kurtosis_like",
        "imag_kurtosis_like",
        "papr_db",
        "snr_db",
    ]

    X = df[feature_columns]
    y = df["modulation"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        max_depth=None
    )

    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    print("\nRandom Forest Modulation Classification")
    print(classification_report(y_test, rf_pred))

    cm = confusion_matrix(y_test, rf_pred, labels=rf_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
    disp.plot()
    plt.title("Random Forest Modulation Classification")
    plt.savefig("figures/confusion_matrix_modulation_rf.png", dpi=300)
    plt.show()

    svm_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=10, gamma="scale"))
        ]
    )

    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)

    print("\nSVM Modulation Classification")
    print(classification_report(y_test, svm_pred))

    cm = confusion_matrix(y_test, svm_pred, labels=svm_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
    disp.plot()
    plt.title("SVM Modulation Classification")
    plt.savefig("figures/confusion_matrix_modulation_svm.png", dpi=300)
    plt.show()

    import joblib
    joblib.dump(rf_model, "models/modulation_classifier_rf.joblib")
    joblib.dump(svm_model, "models/modulation_classifier_svm.joblib")

    print("Saved classifiers to models/")


def plot_feature_importance():
    import joblib

    df = pd.read_csv("data/constellation_ml_dataset.csv")
    model = joblib.load("models/modulation_classifier_rf.joblib")

    feature_columns = [
        "i_mean",
        "q_mean",
        "i_std",
        "q_std",
        "i_var",
        "q_var",
        "iq_cov",
        "amplitude_mean",
        "amplitude_std",
        "amplitude_var",
        "phase_mean",
        "phase_std",
        "real_kurtosis_like",
        "imag_kurtosis_like",
        "papr_db",
        "snr_db",
    ]

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print(importance_df)

    plt.figure(figsize=(8, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance for Modulation Classification")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("figures/feature_importance_modulation.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run_kmeans_on_constellation(modulation="QPSK", snr_db=20, channel_type="AWGN")
    run_kmeans_on_constellation(modulation="16QAM", snr_db=20, channel_type="AWGN")
    run_kmeans_on_constellation(modulation="64QAM", snr_db=25, channel_type="AWGN")

    generate_modulation_classification_dataset(num_examples_per_class=500)

    train_modulation_classifier()

    plot_feature_importance()