# =============================================================
# SPEECH FEATURE EXTRACTION - FINAL CODE
# =============================================================
# Course: Smart and Connected Health
# Project: PD vs TBI Classification using Speech Analysis
# =============================================================


# =============================================================
# CELL 1: INSTALL LIBRARIES (Run once)
# =============================================================
# !pip install librosa praat-parselmouth
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


# =============================================================
# CELL 3: MOUNT GOOGLE DRIVE (If using Drive)
# =============================================================
# from google.colab import drive
# drive.mount('/content/drive')


# =============================================================
# CELL 4: UPLOAD FILE DIRECTLY (If not using Drive)
# =============================================================
# from google.colab import files
# uploaded = files.upload()


# =============================================================
# CELL 5: HELPER FUNCTION - CONVERT M4A/MP4 TO WAV
# =============================================================
def convert_to_wav(input_path):
    """
    Convert m4a/mp4/mp3 file to wav format.
    
    Parameters:
        input_path: path to audio file (m4a, mp4, mp3)
    
    Returns:
        path to converted wav file
    """
    output_path = input_path.rsplit('.', 1)[0] + '.wav'
    command = f'ffmpeg -i "{input_path}" -ar 16000 -ac 1 "{output_path}" -y -loglevel quiet'
    os.system(command)
    return output_path


# =============================================================
# CELL 6: MAIN FUNCTION - EXTRACT ALL FEATURES
# =============================================================
def extract_features(file_path):
    """
    Extract all speech features from an audio file.
    
    Parameters:
        file_path: path to audio file (wav, m4a, mp4, mp3)
    
    Returns:
        dictionary containing all extracted features
    
    Features extracted:
        - duration: total length in seconds
        - pitch_mean: average fundamental frequency (Hz)
        - pitch_std: pitch variation (Hz)
        - jitter: pitch instability (ratio)
        - shimmer: amplitude instability (ratio)
        - hnr: harmonics-to-noise ratio (dB)
        - mfcc_1 to mfcc_13: mel-frequency cepstral coefficients
        - num_pauses: number of pauses detected
        - pause_ratio: percentage of time spent pausing
        - spectral_centroid: center of mass of spectrum (Hz)
        - zcr: zero crossing rate
    """
    
    # ---------- CONVERT IF NEEDED ----------
    if file_path.endswith(('.m4a', '.mp4', '.mp3')):
        wav_path = convert_to_wav(file_path)
    else:
        wav_path = file_path
    
    features = {}
    
    # ---------- LOAD AUDIO ----------
    # librosa: for MFCCs, pauses, spectral features
    audio, sr = librosa.load(wav_path, sr=16000)
    
    # parselmouth: for pitch, jitter, shimmer, HNR
    sound = parselmouth.Sound(wav_path)
    
    # ---------- DURATION ----------
    features['duration'] = librosa.get_duration(y=audio, sr=sr)
    
    # ---------- PITCH ----------
    pitch = call(sound, "To Pitch", 0.0, 75, 500)
    features['pitch_mean'] = call(pitch, "Get mean", 0, 0, "Hertz")
    features['pitch_std'] = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    
    # ---------- JITTER ----------
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
    features['jitter'] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    
    # ---------- SHIMMER ----------
    features['shimmer'] = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    # ---------- HNR ----------
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    features['hnr'] = call(harmonicity, "Get mean", 0, 0)
    
    # ---------- MFCCs ----------
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)
    for i in range(13):
        features[f'mfcc_{i+1}'] = mfcc_means[i]
    
    # ---------- PAUSES ----------
    non_silent = librosa.effects.split(audio, top_db=25)
    num_pauses = len(non_silent) - 1
    total_speech_time = sum([(end - start) / sr for start, end in non_silent])
    total_pause_time = features['duration'] - total_speech_time
    
    features['num_pauses'] = num_pauses
    features['pause_ratio'] = total_pause_time / features['duration']
    
    # ---------- SPECTRAL FEATURES ----------
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(audio))
    
    return features


# =============================================================
# CELL 7: TEST ON ONE FILE
# =============================================================
# Change this path to your audio file
test_file = "/content/20250319_141117_HBOT_070_Grandfather.m4a"

# Extract features
features = extract_features(test_file)

# Print results
print("=" * 50)
print("EXTRACTED FEATURES")
print("=" * 50)
for name, value in features.items():
    if isinstance(value, float):
        print(f"  {name}: {value:.4f}")
    else:
        print(f"  {name}: {value}")
print("=" * 50)
print(f"Total features: {len(features)}")


# =============================================================
# CELL 8: PROCESS MULTIPLE FILES
# =============================================================
def process_all_files(pd_folder, tbi_folder):
    """
    Process all audio files from PD and TBI folders.
    
    Parameters:
        pd_folder: path to folder containing PD patient audio files
        tbi_folder: path to folder containing TBI patient audio files
    
    Returns:
        DataFrame with all features and labels
    """
    
    all_data = []
    
    # Process PD files
    print("Processing PD files...")
    for filename in os.listdir(pd_folder):
        if filename.endswith(('.wav', '.m4a', '.mp4', '.mp3')):
            file_path = os.path.join(pd_folder, filename)
            try:
                features = extract_features(file_path)
                features['filename'] = filename
                features['label'] = 'PD'
                all_data.append(features)
                print(f"  ✅ {filename}")
            except Exception as e:
                print(f"  ❌ {filename}: {e}")
    
    # Process TBI files
    print("\nProcessing TBI files...")
    for filename in os.listdir(tbi_folder):
        if filename.endswith(('.wav', '.m4a', '.mp4', '.mp3')):
            file_path = os.path.join(tbi_folder, filename)
            try:
                features = extract_features(file_path)
                features['filename'] = filename
                features['label'] = 'TBI'
                all_data.append(features)
                print(f"  ✅ {filename}")
            except Exception as e:
                print(f"  ❌ {filename}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    print(f"\n✅ Processed {len(df)} files total")
    return df


# =============================================================
# CELL 9: RUN ON ALL FILES & SAVE
# =============================================================
# Set paths to your data folders
# pd_folder = "/content/drive/MyDrive/SmartHealth/audio_data/PD"
# tbi_folder = "/content/drive/MyDrive/SmartHealth/audio_data/TBI"

# Process all files
# df = process_all_files(pd_folder, tbi_folder)

# View first few rows
# print(df.head())

# Save to CSV
# df.to_csv("/content/drive/MyDrive/SmartHealth/features.csv", index=False)
# print("Saved to features.csv")


# =============================================================
# CELL 10: VIEW RESULTS (Optional)
# =============================================================
# # Load saved features
# df = pd.read_csv("/content/drive/MyDrive/SmartHealth/features.csv")
#
# # Basic stats
# print("Dataset shape:", df.shape)
# print("\nLabel distribution:")
# print(df['label'].value_counts())
#
# # Feature summary
# print("\nFeature statistics:")
# print(df.describe())
