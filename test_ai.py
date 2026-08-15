import numpy as np
from ai.models import NetVisionAI
from ai.explainer import NetVisionExplainer, FEATURE_NAMES

def main():
    print("\n" + "=" * 60)
    print("   [HARI 3] NETVISION AI & XAI LAYER INTEGRATION TEST")
    print("=" * 60)

    # 1. Inisialisasi AI
    print("[*] Menginisialisasi model AI...")
    ai_engine = NetVisionAI()

    # 2. Simulasi Data Training (Untuk Scikit-Learn)
    print("[*] Membuat data dummy untuk melatih model offline...")
    dummy_X_train = np.random.rand(100, 23)  # 100 paket, 23 fitur
    dummy_y_train = np.zeros(100)            # Label 0 = Normal
    
    # Bikin sedikit data anomali (Label 1) biar AI bisa bedain
    dummy_X_train[-5:] = dummy_X_train[-5:] * 5
    dummy_y_train[-5:] = 1 
    
    # 3. Latih Model
    ai_engine.train_offline_models(dummy_X_train, dummy_y_train)

    # 4. Inisialisasi SHAP (XAI)
    xai = NetVisionExplainer(ai_engine.random_forest)

    # 5. Simulasi Paket Masuk (Data Baru)
    print("\n[*] Mensimulasikan 1 paket jaringan masuk dari Capture Layer...")
    new_feature_vector = np.random.rand(23)
    new_feature_dict = {FEATURE_NAMES[i]: new_feature_vector[i] for i in range(23)}

    # A. Prediksi Batch (Scikit-Learn)
    offline_result = ai_engine.predict_offline(new_feature_vector)
    
    # B. Prediksi Streaming & Belajar (River)
    online_anomaly_score = ai_engine.process_online(new_feature_dict)

    # C. Penjelasan (SHAP)
    explanation = xai.explain_prediction(new_feature_vector)

    # 6. Cetak Hasil Akhir
    print("\n" + "=" * 60)
    print("   HASIL KEPUTUSAN AI NETVISION")
    print("=" * 60)
    print(f"  [Isolation Forest] : {'ANOMALI' if offline_result['is_anomaly'] else 'NORMAL'}")
    print(f"  [Random Forest]    : {'SERANGAN' if offline_result['rf_prediction'] == 1 else 'NORMAL'}")
    print(f"  [River HST] Score  : {online_anomaly_score:.4f} (Mendekati 1.0 = Bahaya)")
    print("\n  [XAI / SHAP] Penjelasan:")
    print(f"  >> {explanation}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()