import os
import librosa
import numpy as np
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

def train_model(data, labels):
    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Train-test split
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Train the model
    model.fit(train_data, train_labels)

    # Evaluate the model
    predictions = model.predict(test_data)
    report = classification_report(test_labels, predictions)
    print("Model Evaluation Report:")
    print(report)


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
        train_model(data, target_labels)
    else:
        print("No data available for training.")

    #save model
    import joblib
    def save_model(model, filename):
        try:
            joblib.dump(model, filename)
            print(f"Model başarıyla kaydedildi: {filename}")
        except Exception as e:
            print(f"Model kaydedilirken bir hata oluştu: {e}")


    # Eğitilen modeli kaydediyoruz
    save_model(model, "emotion_recognition_model.joblib")


