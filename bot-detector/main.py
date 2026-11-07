"""Railway/Hugging Face app: upload CSV, get bot-like scores."""
import os
import sys
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from pipeline import build_features, detect  # noqa: E402

ARTIFACT = joblib.load(ROOT / "models" / "bot_model.joblib")


def predict_csv(file):
    if file is None:
        raise gr.Error("Unggah file CSV terlebih dahulu.")
    source = file.name if hasattr(file, "name") else file
    raw = pd.read_csv(source)
    mapping = detect(raw)
    if "label" in mapping:
        raw = raw.drop(columns=[mapping["label"]])
        mapping = detect(raw)
    work = raw.copy()
    work["__temporary_label"] = 0
    mapping["label"] = "__temporary_label"
    features, _, _, _ = build_features(work, mapping, "upload")
    features = features.reindex(columns=ARTIFACT["columns"], fill_value=0)
    features = features.replace([float("inf"), float("-inf")], 0).fillna(0)
    probability = ARTIFACT["model"].predict_proba(features)[:, 1]
    usernames = (raw[mapping["screen_name"]].astype(str).values
                 if "screen_name" in mapping else range(1, len(raw) + 1))
    result = pd.DataFrame({"username": usernames, "bot_probability": probability.round(4)})
    result["signal_level"] = pd.cut(
        probability, [-1, 0.35, 0.65, 1.01], labels=["rendah", "sedang", "tinggi"]
    )
    return result


demo = gr.Interface(
    fn=predict_csv,
    inputs=gr.File(file_types=[".csv"], type="filepath", label="CSV akun tanpa label"),
    outputs=gr.Dataframe(label="Hasil prediksi"),
    title="NusaBotGuard-ML",
    description="Sinyal statistik akun bot-like; bukan bukti identitas atau niat.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
