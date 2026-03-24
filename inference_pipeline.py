# =============================================================
# INFERENCE PIPELINE - PD vs TBI PREDICTION
# =============================================================
# Course: Smart and Connected Health
# Project: PD vs TBI Classification using Speech Analysis
# =============================================================
# This module loads a trained model and predicts PD vs TBI
# from a new audio file.
# =============================================================


# =============================================================
# CELL 1: IMPORTS
# =============================================================
import numpy as np
import joblib
import os
import sys

from feature_extraction_enhanced import extract_all_features


# =============================================================
# CELL 2: LOAD MODEL ARTIFACTS
# =============================================================
def load_model(model_dir="."):
    """
    Load saved model, scaler, label encoder, and feature names.

    Parameters:
        model_dir: directory containing saved .joblib files

    Returns:
        model, scaler, label_encoder, feature_names
    """
    model = joblib.load(os.path.join(model_dir, "best_model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    le = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    feature_names = joblib.load(os.path.join(model_dir, "feature_names.joblib"))

    print(f"Model loaded: {type(model).__name__}")
    print(f"Features expected: {len(feature_names)}")
    print(f"Classes: {list(le.classes_)}")

    return model, scaler, le, feature_names


# =============================================================
# CELL 3: PREDICT FROM AUDIO FILE
# =============================================================
def predict(file_path, model_dir="."):
    """
    Predict PD vs TBI from a single audio file.

    Steps:
        1. Extract features from audio
        2. Align features with training feature order
        3. Scale features
        4. Predict class and probability

    Parameters:
        file_path: path to audio file (wav, m4a, mp4, mp3)
        model_dir: directory containing saved model artifacts

    Returns:
        dict with prediction, confidence, and all probabilities
    """
    # Load model artifacts
    model, scaler, le, feature_names = load_model(model_dir)

    # Extract features
    print(f"\nExtracting features from: {file_path}")
    features = extract_all_features(file_path)

    # Align feature vector to expected order
    feature_vector = []
    for fname in feature_names:
        val = features.get(fname, 0.0)
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            val = 0.0
        feature_vector.append(val)

    feature_vector = np.array(feature_vector).reshape(1, -1)

    # Scale
    feature_vector_scaled = scaler.transform(feature_vector)

    # Predict
    prediction = model.predict(feature_vector_scaled)[0]
    predicted_label = le.inverse_transform([prediction])[0]

    # Probabilities (if available)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(feature_vector_scaled)[0]
        confidence = probas.max()
        class_probas = dict(zip(le.classes_, probas))
    else:
        confidence = None
        class_probas = {}

    result = {
        "file": file_path,
        "prediction": predicted_label,
        "confidence": confidence,
        "probabilities": class_probas,
    }

    # Print result
    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"  File:       {os.path.basename(file_path)}")
    print(f"  Prediction: {predicted_label}")
    if confidence is not None:
        print(f"  Confidence: {confidence:.2%}")
        for cls, prob in class_probas.items():
            print(f"    P({cls}): {prob:.4f}")
    print("=" * 50)

    return result


# =============================================================
# CELL 4: BATCH PREDICT
# =============================================================
def batch_predict(folder_path, model_dir="."):
    """Predict PD vs TBI for all audio files in a folder."""
    supported_ext = (".wav", ".m4a", ".mp4", ".mp3")
    results = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(supported_ext):
            file_path = os.path.join(folder_path, filename)
            try:
                result = predict(file_path, model_dir)
                results.append(result)
            except Exception as e:
                print(f"  [FAIL] {filename}: {e}")

    return results


# =============================================================
# CELL 5: MAIN
# =============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference_pipeline.py <audio_file> [model_dir]")
        print("Example: python inference_pipeline.py recording.m4a ./models")
        exit(1)

    audio_file = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    if not os.path.exists(audio_file):
        print(f"File not found: {audio_file}")
        exit(1)

    predict(audio_file, model_dir)
