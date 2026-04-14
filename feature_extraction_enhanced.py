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
import re
import warnings
from scipy.signal import hilbert, butter, filtfilt
from scipy.fftpack import dct

warnings.filterwarnings("ignore")

try:
    import nolds
    _HAS_NOLDS = True
except Exception:
    _HAS_NOLDS = False

try:
    import antropy
    _HAS_ANTROPY = True
except Exception:
    _HAS_ANTROPY = False

try:
    import opensmile
    _HAS_OPENSMILE = True
except Exception:
    _HAS_OPENSMILE = False
_OPENSMILE_INSTANCE = None


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
# CELL 6.5: EXTENDED FEATURES (+100) — balanced across
#   1. Voice Quality       (25)
#   2. Prosody / Rhythm    (20)
#   3. Spectral / Cepstral (25)
#   4. Nonlinear Complexity (15)
#   5. eGeMAPS subset       (15)
# Any feature whose backing algorithm / library is unavailable
# is returned as NaN so the schema stays fixed at +100 columns.
# =============================================================

_VQ_EXT_KEYS = [
    "cpp_mean", "cpp_std", "cpp_median", "cpp_max",
    "cpps_mean", "cpps_std", "cpps_median", "cpps_max",
    "gne_mean", "gne_std", "gne_median", "gne_max",
    "jitter_ddp", "jitter_ppq11",
    "shimmer_apq11", "shimmer_dda",
    "voice_break_count", "voice_break_degree",
    "soft_phonation_index", "subharmonic_ratio",
    "vot_mean", "vot_std",
    "h1_h2", "h1_a1", "h1_a3",
]

_PROSODY_EXT_KEYS = [
    "speech_rate", "articulation_rate",
    "pause_duration_mean", "pause_duration_std",
    "pause_duration_min", "pause_duration_max",
    "pause_to_speech_ratio", "syllable_rate",
    "rpvi", "npvi",
    "intonation_slope_mean", "intonation_slope_std",
    "f0_contour_slope",
    "voiced_unvoiced_ratio",
    "voiced_segment_duration_mean", "voiced_segment_duration_std",
    "unvoiced_segment_duration_mean", "unvoiced_segment_duration_std",
    "rhythm_regularity", "stress_pattern_index",
]

_SPECTRAL_EXT_KEYS = (
    [f"lpc_{i+1}" for i in range(10)]
    + [f"lfcc_{i+1}_mean" for i in range(8)]
    + [
        "spectral_skewness", "spectral_kurtosis",
        "spectral_spread", "spectral_slope",
        "spectral_decrease", "spectral_crest",
        "spectral_slope_0_500hz",
    ]
)

_NONLINEAR_EXT_KEYS = [
    "rpde", "dfa", "ppe",
    "correlation_dimension_d2",
    "sample_entropy", "approximate_entropy",
    "permutation_entropy",
    "hurst_exponent", "lyapunov_exponent_max",
    "higuchi_fractal_dim", "katz_fractal_dim",
    "lempel_ziv_complexity",
    "teager_energy_mean", "teager_energy_std",
    "shannon_entropy",
]

_EGEMAPS_EXT_KEYS = [
    "loudness_mean", "loudness_std",
    "loudness_percentile20", "loudness_percentile50", "loudness_percentile80",
    "loudness_peaks_per_sec",
    "f0_semitone_range",
    "f0_semitone_mean_rising_slope", "f0_semitone_mean_falling_slope",
    "equivalent_sound_level_dbp",
    "hammarberg_index_mean", "hammarberg_index_std",
    "alpha_ratio_mean", "alpha_ratio_std",
    "voiced_segments_per_sec",
]

_EGEMAPS_NAME_MAP = {
    "loudness_sma3_amean": "loudness_mean",
    "loudness_sma3_stddevNorm": "loudness_std",
    "loudness_sma3_percentile20.0": "loudness_percentile20",
    "loudness_sma3_percentile50.0": "loudness_percentile50",
    "loudness_sma3_percentile80.0": "loudness_percentile80",
    "loudnessPeaksPerSec": "loudness_peaks_per_sec",
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2": "f0_semitone_range",
    "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope": "f0_semitone_mean_rising_slope",
    "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope": "f0_semitone_mean_falling_slope",
    "equivalentSoundLevel_dBp": "equivalent_sound_level_dbp",
    "hammarbergIndexV_sma3nz_amean": "hammarberg_index_mean",
    "hammarbergIndexV_sma3nz_stddevNorm": "hammarberg_index_std",
    "alphaRatioV_sma3nz_amean": "alpha_ratio_mean",
    "alphaRatioV_sma3nz_stddevNorm": "alpha_ratio_std",
    "VoicedSegmentsPerSec": "voiced_segments_per_sec",
}

_ALL_EXT_KEYS = (
    _VQ_EXT_KEYS
    + _PROSODY_EXT_KEYS
    + _SPECTRAL_EXT_KEYS
    + _NONLINEAR_EXT_KEYS
    + _EGEMAPS_EXT_KEYS
)
assert len(_ALL_EXT_KEYS) == 100, f"Expected 100 extended keys, got {len(_ALL_EXT_KEYS)}"


def _ext_safe_float(x, default=np.nan):
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except Exception:
        return default


def _ext_nan_dict(keys):
    return {k: np.nan for k in keys}


def _ext_dist_stats(values, prefix, which=("mean", "std", "median", "max")):
    out = {}
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    for s in which:
        key = f"{prefix}_{s}"
        if arr.size == 0:
            out[key] = np.nan
            continue
        if s == "mean":
            out[key] = _ext_safe_float(np.mean(arr))
        elif s == "std":
            out[key] = _ext_safe_float(np.std(arr))
        elif s == "median":
            out[key] = _ext_safe_float(np.median(arr))
        elif s == "max":
            out[key] = _ext_safe_float(np.max(arr))
        elif s == "min":
            out[key] = _ext_safe_float(np.min(arr))
    return out


def _ext_get_opensmile():
    global _OPENSMILE_INSTANCE
    if not _HAS_OPENSMILE:
        return None
    if _OPENSMILE_INSTANCE is None:
        _OPENSMILE_INSTANCE = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _OPENSMILE_INSTANCE


# -------- Voice quality helpers --------
def _ext_cpp_frame(frame, sr):
    if len(frame) < 64:
        return np.nan
    spec = np.abs(np.fft.rfft(frame * np.hamming(len(frame))))
    log_spec = 20.0 * np.log10(spec + 1e-12)
    cep = np.abs(np.fft.rfft(log_spec))
    lo = max(1, int(sr / 333))
    hi = min(len(cep) - 1, int(sr / 60))
    if hi - lo < 4:
        return np.nan
    q = np.arange(lo, hi)
    c = cep[lo:hi]
    if not np.all(np.isfinite(c)):
        return np.nan
    slope, intercept = np.polyfit(q, c, 1)
    baseline = slope * q + intercept
    peak = int(np.argmax(c))
    return _ext_safe_float(c[peak] - baseline[peak])


def _ext_cpp_series(audio, sr, frame_ms=40, hop_ms=10, smooth=False):
    n = int(frame_ms / 1000.0 * sr)
    hop = int(hop_ms / 1000.0 * sr)
    if n <= 0 or hop <= 0 or len(audio) < n:
        return np.array([])
    vals = []
    for start in range(0, len(audio) - n, hop):
        v = _ext_cpp_frame(audio[start:start + n], sr)
        if np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    if smooth and vals.size > 5:
        k = np.ones(5) / 5.0
        vals = np.convolve(vals, k, mode="valid")
    return vals


def _ext_gne_frame(frame, sr, bandwidth=1000, step=300, f_lo=500, f_hi=4500):
    if len(frame) < 256:
        return np.nan
    envelopes = []
    centers = list(range(f_lo, min(f_hi, int(sr / 2) - bandwidth), step))
    for f_c in centers:
        low = (f_c - bandwidth / 2) / (sr / 2)
        high = (f_c + bandwidth / 2) / (sr / 2)
        if low <= 0 or high >= 1:
            continue
        try:
            b, a = butter(4, [low, high], btype="band")
            filtered = filtfilt(b, a, frame)
            envelopes.append(np.abs(hilbert(filtered)))
        except Exception:
            continue
    if len(envelopes) < 2:
        return np.nan
    max_c = 0.0
    for i in range(len(envelopes)):
        for j in range(i + 1, len(envelopes)):
            a1 = envelopes[i] - np.mean(envelopes[i])
            a2 = envelopes[j] - np.mean(envelopes[j])
            d = np.sqrt(np.sum(a1 ** 2) * np.sum(a2 ** 2))
            if d > 0:
                c = np.max(np.correlate(a1, a2, mode="full")) / d
                if c > max_c:
                    max_c = c
    return _ext_safe_float(max_c)


def _ext_gne_series(audio, sr, frame_ms=30, hop_ms=20):
    n = int(frame_ms / 1000.0 * sr)
    hop = int(hop_ms / 1000.0 * sr)
    if n <= 0 or hop <= 0 or len(audio) < n:
        return np.array([])
    vals = []
    for start in range(0, len(audio) - n, hop):
        v = _ext_gne_frame(audio[start:start + n], sr)
        if np.isfinite(v):
            vals.append(v)
    return np.array(vals)


def _ext_ppq_from_periods(periods, N):
    if len(periods) < N or np.mean(periods) == 0:
        return np.nan
    half = N // 2
    diffs = []
    for i in range(half, len(periods) - half):
        window = periods[i - half:i + half + 1]
        diffs.append(abs(periods[i] - np.mean(window)))
    if not diffs:
        return np.nan
    return _ext_safe_float(np.mean(diffs) / np.mean(periods))


def _ext_soft_phonation_index(audio, sr):
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    low = spec[(freqs >= 70) & (freqs <= 1600)]
    high = spec[(freqs > 1600) & (freqs <= 4500)]
    low_e = float(np.sum(low ** 2))
    high_e = float(np.sum(high ** 2))
    if high_e <= 0 or low_e <= 0:
        return np.nan
    return _ext_safe_float(10.0 * np.log10(low_e / high_e))


def _ext_subharmonic_ratio(audio, sr, f0_mean):
    if not np.isfinite(f0_mean) or f0_mean <= 0:
        return np.nan
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)

    def peak_near(target, tol):
        mask = (freqs >= target - tol) & (freqs <= target + tol)
        if not np.any(mask):
            return 0.0
        return float(np.max(spec[mask]))

    tol_f0 = max(10.0, 0.15 * f0_mean)
    e_f0 = peak_near(f0_mean, tol_f0)
    e_sub = peak_near(f0_mean / 2.0, tol_f0) + peak_near(f0_mean / 3.0, tol_f0)
    if e_f0 <= 0:
        return np.nan
    return _ext_safe_float(e_sub / e_f0)


def _ext_h1_h2_a_tilts(audio, sr, f0_mean, f1_mean, f3_mean):
    if not all(np.isfinite([f0_mean, f1_mean, f3_mean])) or f0_mean <= 0:
        return np.nan, np.nan, np.nan
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)

    def peak_db(target, tol):
        mask = (freqs >= target - tol) & (freqs <= target + tol)
        if not np.any(mask):
            return np.nan
        return 20.0 * np.log10(float(np.max(spec[mask])) + 1e-12)

    h1 = peak_db(f0_mean, max(10.0, 0.2 * f0_mean))
    h2 = peak_db(2.0 * f0_mean, max(15.0, 0.2 * f0_mean))
    a1 = peak_db(f1_mean, 50.0)
    a3 = peak_db(f3_mean, 100.0)
    return _ext_safe_float(h1 - h2), _ext_safe_float(h1 - a1), _ext_safe_float(h1 - a3)


def _ext_extract_voice_quality(sound, audio, sr):
    features = _ext_nan_dict(_VQ_EXT_KEYS)

    try:
        cpp_vals = _ext_cpp_series(audio, sr, smooth=False)
        if cpp_vals.size:
            features.update(_ext_dist_stats(cpp_vals, "cpp"))
    except Exception:
        pass

    try:
        cpps_vals = _ext_cpp_series(audio, sr, smooth=True)
        if cpps_vals.size:
            features.update(_ext_dist_stats(cpps_vals, "cpps"))
    except Exception:
        pass

    try:
        gne_vals = _ext_gne_series(audio, sr)
        if gne_vals.size:
            features.update(_ext_dist_stats(gne_vals, "gne"))
    except Exception:
        pass

    f0_mean = np.nan
    try:
        pitch = call(sound, "To Pitch", 0.0, 75, 500)
        pv = pitch.selected_array["frequency"]
        pv = pv[pv > 0]
        if pv.size:
            f0_mean = float(np.mean(pv))
    except Exception:
        pass

    f1_mean = np.nan
    f3_mean = np.nan
    try:
        formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        times = np.arange(sound.xmin, sound.xmax, 0.01)
        f1_list, f3_list = [], []
        for t in times:
            v1 = _ext_safe_float(call(formant, "Get value at time", 1, float(t), "Hertz", "Linear"))
            v3 = _ext_safe_float(call(formant, "Get value at time", 3, float(t), "Hertz", "Linear"))
            if np.isfinite(v1):
                f1_list.append(v1)
            if np.isfinite(v3):
                f3_list.append(v3)
        if f1_list:
            f1_mean = float(np.mean(f1_list))
        if f3_list:
            f3_mean = float(np.mean(f3_list))
    except Exception:
        pass

    try:
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        features["jitter_ddp"] = _ext_safe_float(
            call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
        )
        features["shimmer_apq11"] = _ext_safe_float(
            call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        )
        features["shimmer_dda"] = _ext_safe_float(
            call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        )

        n_pts = int(call(point_process, "Get number of points"))
        if n_pts >= 12:
            times_pp = np.array([
                call(point_process, "Get time from index", i)
                for i in range(1, n_pts + 1)
            ])
            periods = np.diff(times_pp)
            periods = periods[(periods > 0.002) & (periods < 0.02)]
            if len(periods) >= 11:
                features["jitter_ppq11"] = _ext_ppq_from_periods(periods, 11)

        try:
            pitch_for_report = call(sound, "To Pitch (cc)", 0.0, 75, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 500)
            pulses = call([sound, pitch_for_report], "To PointProcess (cc)")
            vr = call(
                [sound, pitch_for_report, pulses],
                "Voice report", 0.0, 0.0, 75, 500, 1.3, 1.6, 0.03, 0.45,
            )
            m1 = re.search(r"Number of voice breaks:\s*(\d+)", vr)
            m2 = re.search(r"Degree of voice breaks:\s*([\d.]+)\s*%", vr)
            if m1:
                features["voice_break_count"] = float(m1.group(1))
            if m2:
                features["voice_break_degree"] = float(m2.group(1))
        except Exception:
            pass
    except Exception:
        pass

    try:
        features["soft_phonation_index"] = _ext_soft_phonation_index(audio, sr)
    except Exception:
        pass

    try:
        features["subharmonic_ratio"] = _ext_subharmonic_ratio(audio, sr, f0_mean)
    except Exception:
        pass

    try:
        h1h2, h1a1, h1a3 = _ext_h1_h2_a_tilts(audio, sr, f0_mean, f1_mean, f3_mean)
        features["h1_h2"] = h1h2
        features["h1_a1"] = h1a1
        features["h1_a3"] = h1a3
    except Exception:
        pass

    # VOT requires phone-level alignment; left as NaN by design.
    return features


# -------- Prosody / rhythm --------
def _ext_extract_prosody(sound, audio, sr):
    features = _ext_nan_dict(_PROSODY_EXT_KEYS)
    duration = float(librosa.get_duration(y=audio, sr=sr))
    if duration <= 0:
        return features

    non_silent = librosa.effects.split(audio, top_db=25)
    speech_segs = [(s / sr, e / sr) for s, e in non_silent]
    speech_durs = np.array([e - s for s, e in speech_segs])
    pause_durs = np.array([
        speech_segs[i + 1][0] - speech_segs[i][1]
        for i in range(len(speech_segs) - 1)
    ]) if len(speech_segs) > 1 else np.array([])

    total_speech = float(np.sum(speech_durs)) if speech_durs.size else 0.0
    total_pause = max(0.0, duration - total_speech)

    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    try:
        peaks = librosa.util.peak_pick(
            onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10,
        )
    except Exception:
        peaks = np.array([], dtype=int)
    syllable_count = int(len(peaks))

    features["syllable_rate"] = _ext_safe_float(syllable_count / duration)
    features["speech_rate"] = features["syllable_rate"]
    features["articulation_rate"] = _ext_safe_float(
        syllable_count / total_speech if total_speech > 0 else 0.0
    )

    if pause_durs.size:
        features["pause_duration_mean"] = _ext_safe_float(np.mean(pause_durs))
        features["pause_duration_std"] = _ext_safe_float(np.std(pause_durs))
        features["pause_duration_min"] = _ext_safe_float(np.min(pause_durs))
        features["pause_duration_max"] = _ext_safe_float(np.max(pause_durs))
    else:
        for k in ("pause_duration_mean", "pause_duration_std", "pause_duration_min", "pause_duration_max"):
            features[k] = 0.0

    features["pause_to_speech_ratio"] = _ext_safe_float(
        total_pause / total_speech if total_speech > 0 else 0.0
    )

    if syllable_count >= 3:
        iois = np.diff(librosa.frames_to_time(peaks, sr=sr))
        if iois.size >= 2:
            abs_diffs = np.abs(np.diff(iois))
            features["rpvi"] = _ext_safe_float(np.mean(abs_diffs) * 1000.0)
            mean_pairs = (iois[:-1] + iois[1:]) / 2.0
            mean_pairs = np.where(mean_pairs == 0, np.nan, mean_pairs)
            features["npvi"] = _ext_safe_float(100.0 * np.nanmean(abs_diffs / mean_pairs))
            features["rhythm_regularity"] = _ext_safe_float(1.0 / (np.std(iois) + 1e-6))

    try:
        pitch = call(sound, "To Pitch", 0.0, 75, 500)
        pitch_vals = pitch.selected_array["frequency"]
        times = np.arange(len(pitch_vals)) * pitch.dx + pitch.xmin
        voiced_mask = pitch_vals > 0

        slopes = []
        run_start = None
        for i, v in enumerate(voiced_mask):
            if v and run_start is None:
                run_start = i
            elif (not v) and run_start is not None:
                if i - run_start >= 5:
                    t = times[run_start:i]
                    p = pitch_vals[run_start:i]
                    s, _b = np.polyfit(t, p, 1)
                    slopes.append(s)
                run_start = None
        if run_start is not None and len(voiced_mask) - run_start >= 5:
            t = times[run_start:]
            p = pitch_vals[run_start:]
            s, _b = np.polyfit(t, p, 1)
            slopes.append(s)
        if slopes:
            features["intonation_slope_mean"] = _ext_safe_float(np.mean(slopes))
            features["intonation_slope_std"] = _ext_safe_float(np.std(slopes))

        vi = np.where(voiced_mask)[0]
        if vi.size > 5:
            s, _b = np.polyfit(times[vi], pitch_vals[vi], 1)
            features["f0_contour_slope"] = _ext_safe_float(s)

        voiced_runs, unvoiced_runs = [], []
        cur = 0
        last = None
        for v in voiced_mask:
            if last is None:
                cur = 1
            elif v == last:
                cur += 1
            else:
                (voiced_runs if last else unvoiced_runs).append(cur)
                cur = 1
            last = bool(v)
        if last is not None:
            (voiced_runs if last else unvoiced_runs).append(cur)
        v_sec = np.array(voiced_runs) * pitch.dx
        u_sec = np.array(unvoiced_runs) * pitch.dx
        if u_sec.sum() > 0:
            features["voiced_unvoiced_ratio"] = _ext_safe_float(v_sec.sum() / u_sec.sum())
        if v_sec.size:
            features["voiced_segment_duration_mean"] = _ext_safe_float(np.mean(v_sec))
            features["voiced_segment_duration_std"] = _ext_safe_float(np.std(v_sec))
        if u_sec.size:
            features["unvoiced_segment_duration_mean"] = _ext_safe_float(np.mean(u_sec))
            features["unvoiced_segment_duration_std"] = _ext_safe_float(np.std(u_sec))
    except Exception:
        pass

    try:
        if len(peaks) >= 3:
            heights = onset_env[peaks]
            features["stress_pattern_index"] = _ext_safe_float(
                np.std(heights) / (np.mean(heights) + 1e-9)
            )
    except Exception:
        pass

    return features


# -------- Spectral / cepstral --------
def _ext_lfcc(audio, sr, n_cep=8, n_filters=26, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sr / 2.0
    n_fft = 512
    hop = 256
    stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop))
    power = stft ** 2
    freqs = np.linspace(fmin, fmax, n_filters + 2)
    bin_idx = np.floor((n_fft + 1) * freqs / sr).astype(int)
    bin_idx = np.clip(bin_idx, 0, power.shape[0] - 1)
    filters = np.zeros((n_filters, power.shape[0]))
    for m in range(1, n_filters + 1):
        left, center, right = bin_idx[m - 1], bin_idx[m], bin_idx[m + 1]
        if center > left:
            for k in range(left, center):
                filters[m - 1, k] = (k - left) / (center - left)
        if right > center:
            for k in range(center, right):
                filters[m - 1, k] = (right - k) / (right - center)
    mel_e = np.maximum(filters @ power, 1e-10)
    log_e = np.log(mel_e)
    cep = dct(log_e, type=2, axis=0, norm="ortho")[:n_cep]
    return cep


def _ext_extract_spectral(audio, sr):
    features = _ext_nan_dict(_SPECTRAL_EXT_KEYS)

    try:
        lpc = librosa.lpc(audio.astype(np.float32), order=10)
        for i in range(10):
            features[f"lpc_{i+1}"] = _ext_safe_float(lpc[i + 1])
    except Exception:
        pass

    try:
        lfcc_mat = _ext_lfcc(audio, sr, n_cep=8)
        for i in range(8):
            features[f"lfcc_{i+1}_mean"] = _ext_safe_float(np.mean(lfcc_mat[i]))
    except Exception:
        pass

    try:
        stft = np.abs(librosa.stft(audio))
        spec = np.mean(stft, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2 * (len(spec) - 1))
        total = float(np.sum(spec))
        if total <= 0 or not np.all(np.isfinite(spec)):
            return features
        p = spec / (total + 1e-12)
        centroid = float(np.sum(freqs * p))
        spread = float(np.sqrt(np.sum(p * (freqs - centroid) ** 2)))
        m3 = float(np.sum(p * (freqs - centroid) ** 3))
        m4 = float(np.sum(p * (freqs - centroid) ** 4))
        features["spectral_spread"] = _ext_safe_float(spread)
        features["spectral_skewness"] = _ext_safe_float(m3 / (spread ** 3 + 1e-12))
        features["spectral_kurtosis"] = _ext_safe_float(m4 / (spread ** 4 + 1e-12) - 3.0)

        log_spec = np.log(spec + 1e-10)
        slope, _b = np.polyfit(freqs, log_spec, 1)
        features["spectral_slope"] = _ext_safe_float(slope)

        k = np.arange(1, len(spec))
        sk = spec[1:]
        s0 = spec[0]
        denom = float(np.sum(sk)) + 1e-12
        features["spectral_decrease"] = _ext_safe_float(np.sum((sk - s0) / k) / denom)
        features["spectral_crest"] = _ext_safe_float(np.max(spec) / (np.mean(spec) + 1e-12))

        mask500 = freqs <= 500
        if np.sum(mask500) >= 2:
            s500, _b = np.polyfit(freqs[mask500], log_spec[mask500], 1)
            features["spectral_slope_0_500hz"] = _ext_safe_float(s500)
    except Exception:
        pass

    return features


# -------- Nonlinear / complexity --------
def _ext_teager(x):
    if len(x) < 3:
        return np.array([])
    return x[1:-1] ** 2 - x[:-2] * x[2:]


def _ext_rpde(x, m=4, tau=2, radius=None, max_points=400, seed=0):
    N = len(x) - (m - 1) * tau
    if N < 20:
        return np.nan
    embed = np.array([x[i:i + m * tau:tau] for i in range(N)])
    if N > max_points:
        idx = np.random.default_rng(seed).choice(N, max_points, replace=False)
        idx.sort()
        embed = embed[idx]
        N = max_points
    dists = np.linalg.norm(embed[:, None, :] - embed[None, :, :], axis=-1)
    if radius is None:
        radius = 0.1 * float(np.std(x))
    if radius <= 0:
        return np.nan
    rec = dists < radius
    lens = []
    for off in range(1, N):
        diag = np.diag(rec, k=off)
        run = 0
        for v in diag:
            if v:
                run += 1
            else:
                if run >= 2:
                    lens.append(run)
                run = 0
        if run >= 2:
            lens.append(run)
    if len(lens) < 2:
        return np.nan
    arr = np.array(lens, dtype=int)
    probs = np.bincount(arr) / len(arr)
    probs = probs[probs > 0]
    H = -float(np.sum(probs * np.log(probs)))
    maxH = float(np.log(len(probs))) if len(probs) > 1 else 1.0
    if maxH <= 0:
        return np.nan
    return _ext_safe_float(H / maxH)


def _ext_ppe(pitch_vals):
    pv = np.asarray(pitch_vals, dtype=float)
    pv = pv[pv > 0]
    if pv.size < 20:
        return np.nan
    st = 12.0 * np.log2(pv / np.mean(pv))
    hist, _edges = np.histogram(st, bins=30, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return np.nan
    return _ext_safe_float(-float(np.sum(hist * np.log(hist + 1e-12))))


def _ext_lempel_ziv(x):
    if len(x) < 10:
        return np.nan
    median = float(np.median(x))
    s = "".join("1" if v >= median else "0" for v in x)
    n = len(s)
    i, c, k = 0, 1, 1
    while i + k <= n:
        if s[i:i + k] in s[0:i] + s[i:i + k - 1]:
            k += 1
            if i + k > n:
                c += 1
                break
        else:
            c += 1
            i += k
            k = 1
    return _ext_safe_float(c * np.log2(n) / n)


def _ext_extract_nonlinear(audio, sr, pitch_vals=None):
    features = _ext_nan_dict(_NONLINEAR_EXT_KEYS)

    max_len = sr * 5
    sig = audio[:max_len] if len(audio) > max_len else audio
    sig_ds = sig[::4] if len(sig) > 1000 else sig

    try:
        features["rpde"] = _ext_rpde(sig_ds)
    except Exception:
        pass

    if _HAS_NOLDS:
        try:
            features["dfa"] = _ext_safe_float(nolds.dfa(sig_ds))
        except Exception:
            pass
        try:
            features["correlation_dimension_d2"] = _ext_safe_float(
                nolds.corr_dim(sig_ds[:1000], emb_dim=5)
            )
        except Exception:
            pass
        try:
            features["hurst_exponent"] = _ext_safe_float(nolds.hurst_rs(sig_ds))
        except Exception:
            pass
        try:
            features["lyapunov_exponent_max"] = _ext_safe_float(
                nolds.lyap_r(sig_ds[:1000], emb_dim=5)
            )
        except Exception:
            pass

    if _HAS_ANTROPY:
        try:
            features["sample_entropy"] = _ext_safe_float(antropy.sample_entropy(sig_ds))
        except Exception:
            pass
        try:
            features["approximate_entropy"] = _ext_safe_float(antropy.app_entropy(sig_ds))
        except Exception:
            pass
        try:
            features["permutation_entropy"] = _ext_safe_float(
                antropy.perm_entropy(sig_ds, order=3, normalize=True)
            )
        except Exception:
            pass
        try:
            features["higuchi_fractal_dim"] = _ext_safe_float(antropy.higuchi_fd(sig_ds))
        except Exception:
            pass
        try:
            features["katz_fractal_dim"] = _ext_safe_float(antropy.katz_fd(sig_ds))
        except Exception:
            pass
    elif _HAS_NOLDS:
        try:
            features["sample_entropy"] = _ext_safe_float(nolds.sampen(sig_ds))
        except Exception:
            pass

    try:
        features["lempel_ziv_complexity"] = _ext_lempel_ziv(sig_ds)
    except Exception:
        pass

    try:
        te = _ext_teager(sig_ds)
        if te.size:
            features["teager_energy_mean"] = _ext_safe_float(np.mean(te))
            features["teager_energy_std"] = _ext_safe_float(np.std(te))
    except Exception:
        pass

    try:
        hist, _e = np.histogram(sig_ds, bins=64, density=True)
        hist = hist[hist > 0]
        if hist.size:
            features["shannon_entropy"] = _ext_safe_float(
                -float(np.sum(hist * np.log2(hist)))
            )
    except Exception:
        pass

    try:
        if pitch_vals is not None:
            features["ppe"] = _ext_ppe(pitch_vals)
    except Exception:
        pass

    return features


# -------- eGeMAPS (openSMILE) --------
def _ext_extract_egemaps(wav_path):
    features = _ext_nan_dict(_EGEMAPS_EXT_KEYS)
    if not _HAS_OPENSMILE or wav_path is None:
        return features
    try:
        smile = _ext_get_opensmile()
        if smile is None:
            return features
        df = smile.process_file(str(wav_path))
        if len(df) == 0:
            return features
        row = df.iloc[0]
        for src_name, dst_name in _EGEMAPS_NAME_MAP.items():
            if src_name in row.index:
                features[dst_name] = _ext_safe_float(row[src_name])
    except Exception:
        pass
    return features


def extract_extended_features(sound, audio, sr, wav_path=None):
    """Return exactly 100 additional features across 5 categories.

    Unavailable features are returned as NaN so the output schema is fixed.
    """
    features = _ext_nan_dict(_ALL_EXT_KEYS)

    try:
        features.update(_ext_extract_voice_quality(sound, audio, sr))
    except Exception:
        pass
    try:
        features.update(_ext_extract_prosody(sound, audio, sr))
    except Exception:
        pass
    try:
        features.update(_ext_extract_spectral(audio, sr))
    except Exception:
        pass

    pitch_vals = None
    try:
        pitch = call(sound, "To Pitch", 0.0, 75, 500)
        pv = pitch.selected_array["frequency"]
        pitch_vals = pv[pv > 0]
    except Exception:
        pass

    try:
        features.update(_ext_extract_nonlinear(audio, sr, pitch_vals))
    except Exception:
        pass

    try:
        features.update(_ext_extract_egemaps(wav_path))
    except Exception:
        pass

    # Ensure schema shape
    for k in _ALL_EXT_KEYS:
        features.setdefault(k, np.nan)
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
    features.update(extract_extended_features(sound, audio, sr, wav_path=wav_path))

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
