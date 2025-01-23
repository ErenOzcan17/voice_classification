import joblib
import numpy as np

def calculate_ram(feature_importances):
    """
    RAM (Rank Allocation Majority) skorlarını hesaplar.
    """
    total_importance = sum(feature_importances)
    ram_scores = [(i + 1, imp / total_importance) for i, imp in enumerate(feature_importances)]
    ram_scores.sort(key=lambda x: x[1], reverse=True)
    return ram_scores

def display_ram_scores(ram_scores):
    """
    RAM skorlarını ekrana yazdırır.
    """
    print("RAM Skorları (Rank Allocation Majority):")
    for feature, ram in ram_scores:
        print(f"MFCC Index {feature}: RAM Score = {ram:.4f}")

if __name__ == "__main__":
    # Kaydedilmiş modeli yükle
    model_filename = "emotion_recognition_model.joblib"  # Modelin adı
    try:
        model = joblib.load(model_filename)
        print(f"Model başarıyla yüklendi: {model_filename}")
    except Exception as e:
        print(f"Model yüklenirken bir hata oluştu: {e}")
        exit()

    # Modelin feature importance değerlerini al
    try:
        feature_importances = model.feature_importances_
        print("Feature importance değerleri başarıyla alındı.")
    except AttributeError:
        print("Bu model feature_importances_ özelliğini desteklemiyor.")
        exit()

    # RAM hesaplamasını yap
    ram_scores = calculate_ram(feature_importances)

    # RAM skorlarını göster
    display_ram_scores(ram_scores)
