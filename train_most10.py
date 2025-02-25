import os
import uuid
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

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


def print_selected_feature_names(selector, feature_importance, feature_names):
    # Get the indices of the selected features
    selected_indices = selector.get_support(indices=True)

    # Özellikleri ve importans değerlerini bir listeye ekle
    selected_features = [(idx, feature_names[idx], feature_importance[idx]) for idx in selected_indices]

    # Importans değerine göre sıralama (büyükten küçüğe)
    selected_features.sort(key=lambda x: x[2], reverse=True)

    # Sıralı listeyi yazdır
    print("Top 10 Important Features (Index):")
    for idx, name, importance in selected_features:
        print(f"Feature {idx + 1} (MFCC index): {name} with importance: {importance}")


def train_model(data, labels):
    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Train-test split
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Train the model
    model.fit(train_data, train_labels)

    # Feature importance and selection
    feature_importance = model.feature_importances_
    print("Feature Importances:", feature_importance)

    # Select the top 10 most important features (can adjust based on your requirement)
    selector = SelectFromModel(model, max_features=10, importance_getter='auto')
    selector.fit(train_data, train_labels)

    # Get the selected features
    selected_train_data = selector.transform(train_data)
    selected_test_data = selector.transform(test_data)

    # Retrain the model using only the selected features
    model.fit(selected_train_data, train_labels)

    # Evaluate the model
    predictions = model.predict(selected_test_data)
    report = classification_report(test_labels, predictions)
    print("Model Evaluation Report (After Feature Selection):")
    print(report)

    # Feature names are MFCC 1, MFCC 2, ..., MFCC 40
    feature_names = [f"MFCC {i+1}" for i in range(40)]

    # Print selected features
    print_selected_feature_names(selector, feature_importance, feature_names)

    return model

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

    # Train the model
    if data and target_labels:
        model = train_model(data, target_labels)
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
