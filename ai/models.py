"""
Modul model AI untuk NetVision.

Berisi class NetVisionAI yang menggabungkan model offline (Isolation
Forest untuk deteksi anomali dan Random Forest untuk klasifikasi
serangan) dengan model online (HalfSpaceTrees) untuk deteksi intrusi
secara real-time berbasis streaming data.
"""

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from river.anomaly import HalfSpaceTrees


class NetVisionAI:
    """Kumpulan model AI untuk deteksi intrusi pada NetVision."""

    def __init__(self):
        # Model offline: deteksi anomali tanpa label (unsupervised)
        self.isolation_forest = IsolationForest(
            contamination=0.05,
            random_state=42,
        )
        # Model offline: klasifikasi jenis traffic (butuh label)
        self.random_forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
        )
        # Model online: deteksi anomali secara streaming/real-time
        self.online_model = HalfSpaceTrees(
            n_trees=25,
            height=15,
            window_size=250,
            seed=42,
        )

        self.is_trained = False

    def train_offline_models(self, X_train: np.ndarray, y_train: np.ndarray = None):
        """Latih model offline (Isolation Forest, dan Random Forest jika ada label)."""
        print("[NetVisionAI] Melatih Isolation Forest...")
        self.isolation_forest.fit(X_train)

        if y_train is not None:
            print("[NetVisionAI] Melatih Random Forest Classifier...")
            self.random_forest.fit(X_train, y_train)

        self.is_trained = True
        print(f"[NetVisionAI] Training selesai. is_trained = {self.is_trained}")

    def predict_offline(self, feature_vector: np.ndarray) -> dict:
        """Prediksi satu feature vector menggunakan model offline (IF + RF)."""
        if not self.is_trained:
            print("[NetVisionAI] Model belum ditraining, panggil train_offline_models() terlebih dahulu.")
            return {
                "is_anomaly": False,
                "rf_prediction": None,
            }

        feature_vector = feature_vector.reshape(1, -1)

        # Isolation Forest: 1 = Normal, -1 = Anomali
        if_pred = self.isolation_forest.predict(feature_vector)[0]
        is_anomaly = bool(if_pred == -1)

        # Random Forest: 0 = Normal, 1 = Serangan
        rf_prediction = int(self.random_forest.predict(feature_vector)[0])

        return {
            "is_anomaly": is_anomaly,
            "rf_prediction": rf_prediction,
        }

    def process_online(self, feature_dict: dict) -> float:
        """Hitung anomaly score satu sample lalu update model HalfSpaceTrees (online learning)."""
        score = self.online_model.score_one(feature_dict)
        self.online_model.learn_one(feature_dict)
        return score