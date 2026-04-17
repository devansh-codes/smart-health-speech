"""Batch-extract PD features into features_PD.csv with the 428-col schema
used by features_full_updated.csv.

The feature extraction logic mirrors the notebook (PD_TBI_Classification_
Notebook.ipynb, cell 8), including the per-stat summaries (_mean/_std/
_median/_min/_max) produced by _distribution_stats/_matrix_row_stats. The
extended-feature block is reused from feature_extraction_enhanced.

Whisper-based passage trimming is skipped (Whisper isn't installed and the
trim_* columns aren't present in the target schema).
"""

from __future__ import annotations

import csv
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extraction_enhanced import extract_extended_features

TARGET_SR = 16000

PD_DATA_DIR = Path("/Users/devanshchaudhary/Downloads/PD_Data")
SCHEMA_CSV = Path(__file__).parent / "features_full_updated.csv"
OUTPUT_CSV = Path(__file__).parent / "features_PD.csv"

FILENAME_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<hash>[0-9a-f]{64})_(?P<passage>.+?)_Grandfather.*\.wav$",
    re.IGNORECASE,
)


def safe_float(value, default=np.nan) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except Exception:
        return default


def _clean_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _distribution_stats(values, prefix: str) -> Dict[str, float]:
    arr = _clean_array(values)
    stats = {
        f"{prefix}_mean": np.nan,
        f"{prefix}_std": np.nan,
        f"{prefix}_median": np.nan,
        f"{prefix}_min": np.nan,
        f"{prefix}_max": np.nan,
    }
    if arr.size == 0:
        return stats
    stats[f"{prefix}_mean"] = safe_float(np.mean(arr))
    stats[f"{prefix}_std"] = safe_float(np.std(arr))
    stats[f"{prefix}_median"] = safe_float(np.median(arr))
    stats[f"{prefix}_min"] = safe_float(np.min(arr))
    stats[f"{prefix}_max"] = safe_float(np.max(arr))
    return stats


def _matrix_row_stats(matrix, prefix: str) -> Dict[str, float]:
    x = np.asarray(matrix, dtype=float)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    features = {}
    for idx, row in enumerate(x, start=1):
        rp = f"{prefix}_{idx}"
        features[f"{rp}_mean"] = safe_float(np.mean(row))
        features[f"{rp}_std"] = safe_float(np.std(row))
        features[f"{rp}_median"] = safe_float(np.median(row))
        features[f"{rp}_min"] = safe_float(np.min(row))
        features[f"{rp}_max"] = safe_float(np.max(row))
    return features


def extract_voice_quality_features(sound) -> Dict[str, float]:
    features: Dict[str, float] = {}

    pitch = call(sound, "To Pitch", 0.0, 75, 500)
    pitch_values = pitch.selected_array["frequency"]
    pitch_values = pitch_values[pitch_values > 0]
    features.update(_distribution_stats(pitch_values, "pitch_hz"))
    features["pitch_range"] = safe_float(
        features.get("pitch_hz_max", np.nan) - features.get("pitch_hz_min", np.nan)
    )

    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
    features["jitter_local"] = safe_float(call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
    features["jitter_rap"] = safe_float(call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3))
    features["jitter_ppq5"] = safe_float(call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3))

    features["shimmer_local"] = safe_float(call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    features["shimmer_apq3"] = safe_float(call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    features["shimmer_apq5"] = safe_float(call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6))

    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    try:
        harmonicity_values = harmonicity.values.flatten()
    except Exception:
        harmonicity_values = np.array([safe_float(call(harmonicity, "Get mean", 0, 0))])
    features.update(_distribution_stats(harmonicity_values, "hnr"))

    intensity = call(sound, "To Intensity", 75.0, 0.0, "yes")
    try:
        intensity_values = intensity.values.flatten()
    except Exception:
        intensity_values = np.array([safe_float(call(intensity, "Get mean", 0, 0, "energy"))])
    features.update(_distribution_stats(intensity_values, "intensity_db"))
    features["intensity_range"] = safe_float(
        features.get("intensity_db_max", np.nan) - features.get("intensity_db_min", np.nan)
    )

    formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
    times = np.arange(sound.xmin, sound.xmax, 0.01)
    for formant_idx in range(1, 5):
        values = [
            safe_float(call(formant, "Get value at time", formant_idx, float(t), "Hertz", "Linear"))
            for t in times
        ]
        features.update(_distribution_stats(values, f"formant_f{formant_idx}"))

    return features


def extract_librosa_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    features: Dict[str, float] = {}

    stft = np.abs(librosa.stft(audio))
    power_spectrogram = stft ** 2
    harmonic_audio = librosa.effects.harmonic(audio)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    features.update(_matrix_row_stats(mfcc, "mfcc"))
    features.update(_matrix_row_stats(delta_mfcc, "delta_mfcc"))

    chroma_stft = librosa.feature.chroma_stft(S=power_spectrogram, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(S=power_spectrogram, sr=sr)
    features.update(_distribution_stats(chroma_stft.flatten(), "chroma_stft"))
    features.update(_distribution_stats(spectral_contrast.flatten(), "spectral_contrast"))

    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    features.update(_matrix_row_stats(delta2_mfcc, "delta2_mfcc"))

    framewise_features = {
        "spectral_centroid": librosa.feature.spectral_centroid(S=stft, sr=sr).flatten(),
        "spectral_bandwidth_p2": librosa.feature.spectral_bandwidth(S=stft, sr=sr, p=2).flatten(),
        "spectral_flatness": librosa.feature.spectral_flatness(S=power_spectrogram).flatten(),
        "rms_energy": librosa.feature.rms(S=stft).flatten(),
        "zcr": librosa.feature.zero_crossing_rate(audio).flatten(),
    }
    for name, values in framewise_features.items():
        features.update(_distribution_stats(values, name))

    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr).flatten()
    features.update(_distribution_stats(rolloff, "spectral_rolloff"))

    flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
    features.update(_distribution_stats(flux, "spectral_flux"))

    S_norm = stft / (np.sum(stft, axis=0, keepdims=True) + 1e-10)
    entropy = -np.sum(S_norm * np.log(S_norm + 1e-10), axis=0)
    features.update(_distribution_stats(entropy, "spectral_entropy"))

    tonnetz = librosa.feature.tonnetz(y=harmonic_audio, sr=sr)
    features.update(_matrix_row_stats(tonnetz, "tonnetz"))

    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    features["tempo_bpm"] = safe_float(np.median(tempo))

    duration = librosa.get_duration(y=audio, sr=sr)
    non_silent = librosa.effects.split(audio, top_db=25)
    speech_segments = [(start / sr, end / sr) for start, end in non_silent]
    speech_durations = [end - start for start, end in speech_segments]
    pause_durations = (
        [speech_segments[i + 1][0] - speech_segments[i][1] for i in range(len(speech_segments) - 1)]
        if len(speech_segments) > 1
        else []
    )
    total_speech_time = float(np.sum(speech_durations)) if speech_durations else 0.0
    total_pause_time = max(0.0, duration - total_speech_time)
    features["duration_sec"] = safe_float(duration)
    features["num_pauses"] = safe_float(len(pause_durations), default=0.0)
    features["pause_fraction"] = safe_float(total_pause_time / duration if duration > 0 else 0.0, default=0.0)

    return features


def extract_all_features(wav_path: Path) -> Dict[str, float]:
    audio, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    if audio is None or len(audio) == 0:
        raise RuntimeError(f"empty audio: {wav_path}")
    sound = parselmouth.Sound(str(wav_path))

    features: Dict[str, float] = {}
    try:
        features.update(extract_voice_quality_features(sound))
    except Exception as e:
        print(f"  [warn] voice_quality: {e}", file=sys.stderr)
    try:
        features.update(extract_librosa_features(audio, sr))
    except Exception as e:
        print(f"  [warn] librosa: {e}", file=sys.stderr)
    try:
        features.update(extract_extended_features(sound, audio, sr, wav_path=str(wav_path)))
    except Exception as e:
        print(f"  [warn] extended: {e}", file=sys.stderr)
    return features


def parse_filename_meta(filename: str) -> Dict[str, str]:
    m = FILENAME_RE.match(filename)
    if not m:
        return {"date": "", "time": "", "hash": "", "passage": ""}
    return {
        "date": m.group("date"),
        "time": str(int(m.group("time"))),
        "hash": m.group("hash"),
        "passage": m.group("passage"),
    }


def load_schema(path: Path) -> list[str]:
    with path.open() as f:
        return next(csv.reader(f))


def load_processed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {
            os.path.basename(row["audio_path"].replace("\\", "/"))
            for row in csv.DictReader(f)
            if row.get("audio_path")
        }


def main() -> int:
    columns = load_schema(SCHEMA_CSV)
    print(f"Target schema: {len(columns)} columns (from {SCHEMA_CSV.name})")

    files = sorted(p for p in PD_DATA_DIR.iterdir() if p.suffix.lower() in {".wav", ".m4a", ".mp4", ".mp3"})
    print(f"Found {len(files)} audio files in {PD_DATA_DIR}")

    # Assign integer patient_ids deterministically from sorted unique hashes
    hashes = sorted({parse_filename_meta(p.name)["hash"] for p in files if parse_filename_meta(p.name)["hash"]})
    hash_to_pid = {h: str(i + 1) for i, h in enumerate(hashes)}
    print(f"Unique patient hashes: {len(hashes)} -> integer patient_ids 1..{len(hashes)}")

    # key by file basename for incremental resume
    processed = load_processed(OUTPUT_CSV)
    write_header = not OUTPUT_CSV.exists()

    start = time.time()
    ok = fail = 0

    with OUTPUT_CSV.open("a", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
            out.flush()

        for i, path in enumerate(files, 1):
            # file_name column stores the passage (e.g., 'Grandfather'), same as TBI; use basename for de-dup
            if path.name in processed:
                continue
            meta = parse_filename_meta(path.name)
            t0 = time.time()
            try:
                feats = extract_all_features(path)
                row = {c: feats.get(c, np.nan) for c in columns}
                row["label"] = "PD"
                row["patient_id"] = hash_to_pid.get(meta["hash"], "")
                # Match TBI convention: file_name = passage short name (Grandfather) when discoverable
                row["file_name"] = "Grandfather"
                row["date"] = meta["date"]
                row["time"] = meta["time"]
                row["audio_path"] = f"pd_audio_input\\{path.name}"
                # Store the full original filename as a de-dup key via a comment field?
                # We track de-dup against the actual wav basename through the processed set re-parsed below.
                writer.writerow(row)
                out.flush()
                processed.add(path.name)
                ok += 1
                dt = time.time() - t0
                elapsed = time.time() - start
                eta = (len(files) - i) * (elapsed / max(i, 1)) / 60
                print(f"[{i}/{len(files)}] OK  {path.name}  ({dt:.1f}s, eta {eta:.1f}min)")
            except Exception as e:
                fail += 1
                print(f"[{i}/{len(files)}] FAIL {path.name}: {e}", file=sys.stderr)

    print(f"\nDone. ok={ok} fail={fail} elapsed={(time.time()-start)/60:.1f}min")
    print(f"Output: {OUTPUT_CSV}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
