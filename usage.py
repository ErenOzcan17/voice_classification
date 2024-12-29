import joblib
import numpy as np
import librosa

model = joblib.load("emotion_recognition_model.joblib")

def predict_emotion(file_path):
    feature = extract_features(file_path)
    if feature is not None:
        probabilities = model.predict_proba([feature])
        prediction = model.predict([feature])
        return prediction[0], probabilities[0]
    return None, None

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

new_audio_path = "audio.wav"
labels = []
for classes in model.classes_:
    labels.append(classes)
predicted_emotion, emotion_scores = predict_emotion(new_audio_path)
if predicted_emotion:
    print(f"Predicted emotion: {predicted_emotion}")
    print("Emotion scores:")
    for label, score in zip(labels, emotion_scores):
        print(f"  {label}: {score:.2f}")
else:
    print("Failed to predict emotion.")