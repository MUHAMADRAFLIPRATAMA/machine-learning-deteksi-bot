"""Periksa semua file data di folder data/ dan simpan laporan ke data_report.txt.

Jalankan dari folder proyek:  python src\\inspect_data.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
OUT = Path("data_report.txt")
LABEL_HINTS = ("label", "bot", "class", "target", "is_", "fake", "spam", "type")

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def load(path):
    ext = path.suffix.lower()
    if ext in (".csv", ".tsv", ".txt"):
        sep = "\t" if ext == ".tsv" else None
        for enc in ("utf-8", "latin-1"):
            try:
                return pd.read_csv(path, sep=sep, engine="python", encoding=enc, nrows=5000)
            except Exception as e:  # noqa: BLE001
                err = e
        raise err
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, nrows=5000)
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            return pd.json_normalize(json.load(f)[:5000])
    return None


def main():
    if not DATA_DIR.exists():
        sys.exit("Folder data/ tidak ditemukan. Jalankan dari folder proyek.")

    files = [p for p in sorted(DATA_DIR.rglob("*")) if p.is_file()]
    log(f"Total file: {len(files)}\n")

    for p in files:
        mb = p.stat().st_size / 1e6
        log("=" * 70)
        log(f"{p}  ({mb:.2f} MB)")
        try:
            df = load(p)
        except Exception as e:  # noqa: BLE001
            log(f"  GAGAL dibaca: {e}")
            continue
        if df is None:
            log("  (format tidak dibaca, dilewati)")
            continue

        log(f"  Baris (maks 5000 dibaca): {len(df)}  | Kolom: {df.shape[1]}")
        log("  Kolom & tipe:")
        for c in df.columns:
            log(f"    - {c}  [{df[c].dtype}]  contoh: {str(df[c].iloc[0])[:60]!r}")

        cands = [c for c in df.columns if any(h in str(c).lower() for h in LABEL_HINTS)]
        for c in cands:
            if df[c].nunique() <= 10:
                log(f"  Kandidat label '{c}': {df[c].value_counts(dropna=False).to_dict()}")
        log()

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nLaporan tersimpan di {OUT.resolve()}")


if __name__ == "__main__":
    main()