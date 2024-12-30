import joblib
import numpy as np
import librosa

model = joblib.load("emotion_recognition_model.joblib")

new_audio_path = "labels/mutlu/5_oldu.wav"
labels = []
for classes in model.classes_:
    labels.append(classes)

audio, sr = librosa.load(new_audio_path, sr=None)
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
mfccs_mean = np.mean(mfccs.T, axis=0)
probabilities = model.predict_proba([mfccs_mean])
prediction = model.predict([mfccs_mean])
print(f"Predicted emotion: {prediction[0]}")
print("Emotion scores:")
for label, score in zip(labels, probabilities[0]):
    print(f"    {label}: {score:.2f}")
