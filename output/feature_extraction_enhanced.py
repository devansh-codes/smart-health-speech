# =============================================================
# ENHANCED SPEECH FEATURE EXTRACTION
# =============================================================
# Course: Smart and Connected Health
# Project: PD vs TBI Classification using Speech Analysis
# =============================================================
# This module extracts 40+ speech features from audio files
# covering voice quality, spectral, and temporal categories.
# =============================================================


# =============================================================
# CELL 1: INSTALL LIBRARIES (Run once in Colab)
# =============================================================
# !pip install librosa praat-parselmouth scikit-learn xgboost
# !apt-get install -y ffmpeg


# =============================================================
# CELL 2: IMPORT LIBRARIES
# =============================================================
import librosa
import parselmouth
from parselmouth.praat import call
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")


# =============================================================
# CELL 3: HELPER - CONVERT M4A/MP4 TO WAV
# =============================================================
def convert_to_wav(input_path):
    """Convert m4a/mp4/mp3 file to wav format (16kHz mono)."""
    output_path = input_path.rsplit(".", 1)[0] + ".wav"
    command = f'ffmpeg -i "{input_path}" -ar 16000 -ac 1 "{output_path}" -y -loglevel quiet'
    os.system(command)
    return output_path


# =============================================================
# CELL 4: VOICE QUALITY FEATURES (Parselmouth / Praat)
# =============================================================
def extract_voice_quality_features(sound):
    """
    Extract voice quality features using Parselmouth (Praat).

    Returns dict with:
        - pitch_mean, pitch_std, pitch_min, pitch_max
        - jitter_local, jitter_rap, jitter_ppq5
        - shimmer_local, shimmer_apq3, shimmer_apq5
        - hnr
        - formant_f1, formant_f2, formant_f3, formant_f4
    """
    features = {}

    # --- Pitch ---
    pitch = call(sound, "To Pitch", 0.0, 75, 500)
    features["pitch_mean"] = call(pitch, "Get mean", 0, 0, "Hertz")
    features["pitch_std"] = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    features["pitch_min"] = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    features["pitch_max"] = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")

    # --- Jitter (3 variants) ---
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
    features["jitter_local"] = call(
        point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
    )
    features["jitter_rap"] = call(
        point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3
    )
    features["jitter_ppq5"] = call(
        point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3
    )

    # --- Shimmer (3 variants) ---
    features["shimmer_local"] = call(
        [sound, point_process],
        "Get shimmer (local)",
        0, 0, 0.0001, 0.02, 1.3, 1.6,
    )
    features["shimmer_apq3"] = call(
        [sound, point_process],
        "Get shimmer (apq3)",
        0, 0, 0.0001, 0.02, 1.3, 1.6,
    )
    features["shimmer_apq5"] = call(
        [sound, point_process],
        "Get shimmer (apq5)",
        0, 0, 0.0001, 0.02, 1.3, 1.6,
    )

    # --- HNR ---
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    features["hnr"] = call(harmonicity, "Get mean", 0, 0)

    # --- Formants (F1-F4) ---
    formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
    features["formant_f1"] = call(formant, "Get mean", 1, 0, 0, "Hertz")
    features["formant_f2"] = call(formant, "Get mean", 2, 0, 0, "Hertz")
    features["formant_f3"] = call(formant, "Get mean", 3, 0, 0, "Hertz")
    features["formant_f4"] = call(formant, "Get mean", 4, 0, 0, "Hertz")

    return features


# =============================================================
# CELL 5: SPECTRAL FEATURES (Librosa)
# =============================================================
def extract_spectral_features(audio, sr):
    """
    Extract spectral features using librosa.

    Returns dict with:
        - mfcc_1 through mfcc_13 (means)
        - delta_mfcc_1 through delta_mfcc_13 (means)
        - spectral_centroid, spectral_bandwidth
        - spectral_rolloff, spectral_flatness
        - spectral_contrast (mean across bands)
        - chroma_mean (mean across 12 chroma bins)
    """
    features = {}

    # --- MFCCs (13 coefficients) ---
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_stds = np.std(mfccs, axis=1)
    for i in range(13):
        features[f"mfcc_{i+1}"] = mfcc_means[i]

    # --- Delta MFCCs (rate of change) ---
    delta_mfccs = librosa.feature.delta(mfccs)
    delta_mfcc_means = np.mean(delta_mfccs, axis=1)
    for i in range(13):
        features[f"delta_mfcc_{i+1}"] = delta_mfcc_means[i]

    # --- Spectral Centroid ---
    features["spectral_centroid"] = float(
        np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    )

    # --- Spectral Bandwidth ---
    features["spectral_bandwidth"] = float(
        np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    )

    # --- Spectral Rolloff ---
    features["spectral_rolloff"] = float(
        np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    )

    # --- Spectral Flatness ---
    features["spectral_flatness"] = float(
        np.mean(librosa.feature.spectral_flatness(y=audio))
    )

    # --- Spectral Contrast ---
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features["spectral_contrast"] = float(np.mean(contrast))

    # --- Chroma Features ---
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features["chroma_mean"] = float(np.mean(chroma))

    return features


# =============================================================
# CELL 6: TEMPORAL FEATURES (Librosa)
# =============================================================
def extract_temporal_features(audio, sr):
    """
    Extract temporal features using librosa.

    Returns dict with:
        - duration, zcr, rms_energy
        - num_pauses, pause_ratio, speech_rate
        - tempo
    """
    features = {}

    # --- Duration ---
    features["duration"] = librosa.get_duration(y=audio, sr=sr)

    # --- Zero Crossing Rate ---
    features["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(audio)))

    # --- RMS Energy ---
    rms = librosa.feature.rms(y=audio)
    features["rms_energy"] = float(np.mean(rms))

    # --- Pause Detection ---
    non_silent = librosa.effects.split(audio, top_db=25)
    num_pauses = max(0, len(non_silent) - 1)
    total_speech_time = sum([(end - start) / sr for start, end in non_silent])
    total_pause_time = features["duration"] - total_speech_time

    features["num_pauses"] = num_pauses
    features["pause_ratio"] = (
        total_pause_time / features["duration"] if features["duration"] > 0 else 0
    )

    # --- Speech Rate (syllables proxy: non-silent segments per second) ---
    features["speech_rate"] = (
        len(non_silent) / features["duration"] if features["duration"] > 0 else 0
    )

    # --- Tempo ---
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    features["tempo"] = float(np.atleast_1d(tempo)[0])

    return features


# =============================================================
# CELL 7: MASTER EXTRACTION FUNCTION
# =============================================================
def extract_all_features(file_path):
    """
    Extract ALL speech features from an audio file.

    Combines voice quality (Parselmouth), spectral (librosa),
    and temporal (librosa) features into a single dictionary.

    Total features: 50+

    Parameters:
        file_path: path to audio file (wav, m4a, mp4, mp3)

    Returns:
        dictionary containing all extracted features
    """
    # Convert if needed
    if file_path.endswith((".m4a", ".mp4", ".mp3")):
        wav_path = convert_to_wav(file_path)
    else:
        wav_path = file_path

    # Load audio with both libraries
    audio, sr = librosa.load(wav_path, sr=16000)
    sound = parselmouth.Sound(wav_path)

    # Extract all feature categories
    features = {}
    features.update(extract_voice_quality_features(sound))
    features.update(extract_spectral_features(audio, sr))
    features.update(extract_temporal_features(audio, sr))

    return features


# =============================================================
# CELL 8: BATCH PROCESSING - MULTIPLE FILES
# =============================================================
def process_folder(folder_path, label):
    """Process all audio files in a folder with a given label."""
    all_data = []
    supported_ext = (".wav", ".m4a", ".mp4", ".mp3")

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(supported_ext):
            file_path = os.path.join(folder_path, filename)
            try:
                features = extract_all_features(file_path)
                features["filename"] = filename
                features["label"] = label
                all_data.append(features)
                print(f"  [OK] {filename} - {len(features)} features")
            except Exception as e:
                print(f"  [FAIL] {filename}: {e}")

    return all_data


def process_all_files(pd_folder, tbi_folder):
    """
    Process all audio files from PD and TBI folders.

    Returns:
        DataFrame with all features and labels
    """
    print("Processing PD files...")
    pd_data = process_folder(pd_folder, "PD")

    print("\nProcessing TBI files...")
    tbi_data = process_folder(tbi_folder, "TBI")

    df = pd.DataFrame(pd_data + tbi_data)
    print(f"\nTotal files processed: {len(df)}")
    print(f"  PD: {len(pd_data)}, TBI: {len(tbi_data)}")
    print(f"  Features per file: {len(df.columns) - 2}")  # minus filename, label
    return df


# =============================================================
# CELL 9: SINGLE FILE TEST
# =============================================================
if __name__ == "__main__":
    # -- Test on the provided sample file --
    test_file = "20250319_141117_HBOT_070_Grandfather.m4a"

    # Adjust path for Colab vs local
    if os.path.exists(f"/content/{test_file}"):
        test_file = f"/content/{test_file}"
    elif not os.path.exists(test_file):
        print(f"File not found: {test_file}")
        print("Please update the path to your audio file.")
        exit(1)

    print("=" * 60)
    print("ENHANCED SPEECH FEATURE EXTRACTION")
    print("=" * 60)

    features = extract_all_features(test_file)

    print(f"\n{'Feature':<30} {'Value':>15}")
    print("-" * 47)
    for name, value in features.items():
        if isinstance(value, float):
            print(f"  {name:<28} {value:>15.6f}")
        else:
            print(f"  {name:<28} {str(value):>15}")

    print("-" * 47)
    print(f"  Total features extracted: {len(features)}")
    print("=" * 60)
