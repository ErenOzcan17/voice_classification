import numpy as np
import os
import uuid
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def select_features(data, feature_indices):
    """
    Manuel olarak seçilen özelliklere göre veriyi filtreler.
    :param data: Tüm özellikleri içeren veri (n_samples x n_features)
    :param feature_indices: Seçilen özelliklerin indeksleri
    :return: Filtrelenmiş veri
    """
    return data[:, feature_indices]

def train_model(data, labels, selected_features=None):
    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Train-test split
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Manuel özellik seçimi
    if selected_features is not None:
        train_data = select_features(train_data, selected_features)
        test_data = select_features(test_data, selected_features)

    # Train the model
    model.fit(train_data, train_labels)


    # Evaluate the model
    predictions = model.predict(test_data)
    report = classification_report(test_labels, predictions)
    print("Model Evaluation Report (Manual Feature Selection):")
    print(report)

    return model

def display_selected_features(feature_indices, total_features):
    """
    Seçilen özelliklerin indekslerini ve anlamlarını görüntüler.
    :param feature_indices: Seçilen özelliklerin indeksleri
    :param total_features: Tüm özelliklerin toplam sayısı
    """
    feature_names = [f"MFCC_{i}" for i in range(total_features)]  # Örnek isimler MFCC_0, MFCC_1 ...
    selected_feature_names = [feature_names[i] for i in feature_indices]
    print("Seçilen Özellikler:")
    for idx, name in zip(feature_indices, selected_feature_names):
        print(f"İndeks: {idx}, Özellik Adı: {name}")

# Main script
if __name__ == "__main__":
    labels = ["mutlu", "sinirli", "uzgun"]
    data = []
    target_labels = []

    # Load data
    for label in labels:
        label_path = f"labels/{label}"
        for filename in sorted(os.listdir(label_path)):
            file_path = os.path.join(label_path, filename)
            if os.path.isfile(file_path):  # Only process files
                features = extract_features(file_path)
                if features is not None:
                    data.append(features)
                    target_labels.append(label)

    # Manuel seçilen özelliklerin indeksleri (örnek: ilk 10 özellik)
    selected_feature_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Seçilen özelliklerin anlamlarını görüntüle
    if data:
        display_selected_features(selected_feature_indices, total_features=len(data[0]))

    # Train the model
    if data and target_labels:
        model = train_model(data, target_labels, selected_features=selected_feature_indices)
    else:
        print("No data available for training.")

    # Save model
    import joblib
    def save_model(model, filename):
        try:
            joblib.dump(model, filename)
            print(f"Model başarıyla kaydedildi: {filename}")
        except Exception as e:
            print(f"Model kaydedilirken bir hata oluştu: {e}")

    # Save the trained model
    save_model(model, f"emotion_recognition_model_{uuid.uuid4()}.joblib")
