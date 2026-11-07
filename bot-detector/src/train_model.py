"""Latih dan simpan model profil untuk deployment.

Contoh:
  python src/train_model.py ../data/dataset1/train.json
"""
import argparse
from pathlib import Path
import joblib
from pipeline import prepare, get_models


def train(reference, output):
    d = prepare(reference)
    model = get_models().get("XGBoost", get_models()["RandomForest"])
    model.fit(d["X"], d["y"])
    artifact = {
        "model": model,
        "columns": list(d["X"].columns),
        "reference": str(reference),
        "features": "profile",
        "warning": "Skor statistik, bukan bukti identitas atau niat akun.",
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    return artifact


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", nargs="?", default="../data/dataset1/train.json")
    ap.add_argument("--output", default="models/bot_model.joblib")
    args = ap.parse_args()
    a = train(args.reference, args.output)
    print(f"Model tersimpan: {Path(args.output).resolve()}")
    print(f"Fitur: {len(a['columns'])}; data referensi: {a['reference']}")
