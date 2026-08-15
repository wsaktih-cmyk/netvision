"""
Modul Explainable AI (XAI) untuk NetVision.

Berisi NetVisionExplainer yang menggunakan SHAP (TreeExplainer) untuk
menjelaskan hasil prediksi Random Forest secara human-readable, sehingga
setiap keputusan deteksi intrusi bisa diaudit dan tidak jadi "black box".
"""

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

FEATURE_NAMES = [
    "packet_count",
    "packet_rate",
    "total_bytes",
    "avg_packet_size",
    "unique_src_ips",
    "unique_dst_ips",
    "unique_src_ports",
    "unique_dst_ports",
    "port_diversity",
    "tcp_count",
    "udp_count",
    "icmp_count",
    "other_count",
    "tcp_ratio",
    "udp_ratio",
    "icmp_ratio",
    "syn_count",
    "fin_count",
    "rst_count",
    "syn_ratio",
    "bytes_per_second",
    "ip_diversity_ratio",
    "avg_entropy",
]


class NetVisionExplainer:
    """Menjelaskan prediksi RandomForestClassifier NetVision menggunakan SHAP."""

    def __init__(self, model: RandomForestClassifier):
        if not hasattr(model, "estimators_"):
            raise ValueError(
                "Model belum di-fit. Latih RandomForestClassifier terlebih "
                "dahulu sebelum membuat NetVisionExplainer."
            )
        self.explainer = shap.TreeExplainer(model)

    def explain_prediction(self, feature_vector: np.ndarray) -> str:
        """Jelaskan satu prediksi dalam bentuk narasi berbahasa Indonesia."""
        feature_vector = feature_vector.reshape(1, -1)
        shap_values = self.explainer.shap_values(feature_vector)

        # SHAP TreeExplainer untuk model multi-output (RandomForestClassifier)
        # mengembalikan list per kelas pada versi < 0.45, dan array tunggal
        # dengan dimensi kelas di axis terakhir pada versi >= 0.45. Tangani
        # keduanya agar kode tetap jalan di kedua versi shap.
        if isinstance(shap_values, list):
            class1_values = shap_values[1][0]
        else:
            class1_values = shap_values[0, :, 1]

        top_idx = np.argmax(np.abs(class1_values))

        feature_name = FEATURE_NAMES[top_idx]
        feature_value = feature_vector[0, top_idx]
        contribution = class1_values[top_idx]

        if contribution > 0:
            return (
                f"Fitur '{feature_name}' (nilai: {feature_value:.4f}) "
                f"meningkatkan kecurigaan anomali karena nilainya tergolong "
                f"tinggi/tidak biasa (kontribusi SHAP: {contribution:.4f})."
            )
        else:
            return (
                f"Fitur '{feature_name}' (nilai: {feature_value:.4f}) "
                f"menurunkan kecurigaan anomali karena nilainya masih dalam "
                f"batas wajar (kontribusi SHAP: {contribution:.4f})."
            )