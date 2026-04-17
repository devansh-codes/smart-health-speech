"""Batch-extract features from PD audio files into features_PD.csv.

Reads features.csv to lock the column schema, processes every audio file in
PD_DATA_DIR, extracts features via feature_extraction_enhanced, and writes rows
with the same column order. Incremental writes let you resume after interrupts.
"""

from __future__ import annotations

import csv
import os
import re
import sys
import time
from pathlib import Path

from feature_extraction_enhanced import extract_all_features

PD_DATA_DIR = Path("/Users/devanshchaudhary/Downloads/PD_Data")
SCHEMA_CSV = Path(__file__).parent / "features.csv"
OUTPUT_CSV = Path(__file__).parent / "features_PD.csv"

HASH_RE = re.compile(r"_([0-9a-f]{64})_")


def patient_id_from_filename(filename: str) -> str:
    m = HASH_RE.search(filename)
    return m.group(1) if m else ""


def load_schema(path: Path) -> list[str]:
    with path.open() as f:
        return next(csv.reader(f))


def load_processed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as f:
        reader = csv.DictReader(f)
        return {row["filename"] for row in reader if row.get("filename")}


def main() -> int:
    columns = load_schema(SCHEMA_CSV)
    processed = load_processed(OUTPUT_CSV)

    files = sorted(p for p in PD_DATA_DIR.iterdir() if p.suffix.lower() in {".wav", ".m4a", ".mp4", ".mp3"})
    print(f"Found {len(files)} audio files in {PD_DATA_DIR}")
    print(f"Already processed: {len(processed)}")
    print(f"Schema columns: {len(columns)}")

    write_header = not OUTPUT_CSV.exists()
    start = time.time()
    ok = fail = 0

    with OUTPUT_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
            f.flush()

        for i, path in enumerate(files, 1):
            if path.name in processed:
                continue
            t0 = time.time()
            try:
                feats = extract_all_features(str(path))
                feats["filename"] = path.name
                feats["patient_id"] = patient_id_from_filename(path.name)
                writer.writerow(feats)
                f.flush()
                ok += 1
                dt = time.time() - t0
                elapsed = time.time() - start
                remaining = (len(files) - i) * (elapsed / max(i, 1))
                print(f"[{i}/{len(files)}] OK  {path.name}  ({dt:.1f}s, eta {remaining/60:.1f}min)")
            except Exception as e:
                fail += 1
                print(f"[{i}/{len(files)}] FAIL {path.name}: {e}", file=sys.stderr)

    print(f"\nDone. ok={ok} fail={fail} elapsed={(time.time()-start)/60:.1f}min")
    print(f"Output: {OUTPUT_CSV}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
