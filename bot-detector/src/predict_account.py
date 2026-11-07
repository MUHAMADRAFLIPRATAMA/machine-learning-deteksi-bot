"""Prediksi sinyal bot untuk CSV akun tanpa kolom label.

Contoh:
  python src/predict_account.py ../data/akun_uji.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pipeline import prepare, read_any, detect, build_features, get_models


def train_reference(path):
    d = prepare(path)
    # Profil-only: aman untuk input akun yang hanya punya data profil.
    tr, te = train_test_split(np.arange(d["n"]), test_size=.2,
                              stratify=d["y"], random_state=42)
    model = get_models().get("XGBoost") or get_models()["RandomForest"]
    model.fit(d["X"].iloc[tr], d["y"].iloc[tr])
    return model, list(d["X"].columns)


def predict(reference, target):
    model, columns = train_reference(reference)
    raw = read_any(target)
    mapping = detect(raw)
    if "label" in mapping:
        raw = raw.drop(columns=[mapping["label"]])
        mapping = detect(raw)
    f, _, _, notes = build_features_without_label(raw, mapping)
    f = f.reindex(columns=columns, fill_value=0).replace([np.inf, -np.inf], 0).fillna(0)
    p = model.predict_proba(f)[:, 1]
    out = pd.DataFrame({"bot_probability": p.round(4)})
    if "screen_name" in mapping:
        out.insert(0, "username", raw[mapping["screen_name"]].astype(str).values)
    else:
        out.insert(0, "row", np.arange(len(raw)))
    out["signal_level"] = pd.cut(p, [-1, .35, .65, 1.01], labels=["rendah", "sedang", "tinggi"])
    return out, notes


def build_features_without_label(df, mapping):
    # build_features only needs a label to return y; use a temporary neutral label,
    # then discard it. This keeps feature definitions identical to training.
    from pipeline import build_features
    tmp = df.copy(); tmp["__temporary_label"] = 0
    m = dict(mapping); m["label"] = "__temporary_label"
    return build_features(tmp, m, "target")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="CSV/JSON akun baru tanpa label")
    ap.add_argument("--reference", default="../data/dataset1/train.json")
    ap.add_argument("--out", default="results/account_predictions.csv")
    args = ap.parse_args()
    result, notes = predict(args.reference, args.target)
    Path(args.out).parent.mkdir(exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))
    print(f"\nTersimpan: {Path(args.out).resolve()}")
    print("Catatan: skor adalah sinyal statistik, bukan bukti bot/buzzer.")


if __name__ == "__main__":
    main()
