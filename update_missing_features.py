"""
Update a features CSV by re-extracting the 44 previously-failing features.

Usage:
    python update_missing_features.py \
        --csv features_full.csv \
        --audio-root /path/to/folder/containing/tbi_audio_input \
        --out features_full_updated.csv

The script reads each row's `audio_path`, resolves it relative to --audio-root
(handles Windows-style backslashes), runs feature_extraction_enhanced on the
file, and writes the 44 target features back into the matching columns. All
other columns are left untouched.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

TARGET_FEATURES = [
    "gne_mean", "gne_std", "gne_median", "gne_max",
    "vot_mean", "vot_std",
    "lfcc_1_mean", "lfcc_2_mean", "lfcc_3_mean", "lfcc_4_mean",
    "lfcc_5_mean", "lfcc_6_mean", "lfcc_7_mean", "lfcc_8_mean",
    "rpde", "dfa", "ppe",
    "correlation_dimension_d2", "sample_entropy", "approximate_entropy",
    "permutation_entropy", "hurst_exponent", "lyapunov_exponent_max",
    "higuchi_fractal_dim", "katz_fractal_dim", "lempel_ziv_complexity",
    "teager_energy_mean", "teager_energy_std", "shannon_entropy",
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


def resolve_audio(path_from_csv, audio_root):
    p = path_from_csv.replace("\\", "/")
    candidates = [
        os.path.join(audio_root, p),
        os.path.join(audio_root, os.path.basename(p)),
        p,
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--audio-root", required=True,
                    help="Folder containing tbi_audio_input/ (or the wavs themselves)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None,
                    help="Optional substring filter on audio_path to process a subset")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from feature_extraction_enhanced import extract_all_features

    df = pd.read_csv(args.csv)
    for col in TARGET_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    ok = fail = skip = 0
    for idx, row in df.iterrows():
        audio_rel = str(row["audio_path"])
        if args.only and args.only not in audio_rel:
            continue
        audio = resolve_audio(audio_rel, args.audio_root)
        if audio is None:
            print(f"[SKIP] file not found: {audio_rel}")
            skip += 1
            continue
        try:
            feats = extract_all_features(audio)
            for col in TARGET_FEATURES:
                if col in feats:
                    df.at[idx, col] = feats[col]
            ok += 1
            print(f"[OK {ok}/{len(df)}] {os.path.basename(audio)}")
        except Exception as exc:
            print(f"[FAIL] {audio}: {exc}")
            fail += 1

    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} — ok={ok}, fail={fail}, skipped={skip}")


if __name__ == "__main__":
    main()
