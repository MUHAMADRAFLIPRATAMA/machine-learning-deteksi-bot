import io
import joblib
import gradio as gr
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from pipeline import detect, build_features

ARTIFACT = joblib.load(Path(__file__).parent / "models" / "bot_model.joblib")


def predict_csv(file):
    if file is None:
        raise gr.Error("Unggah file CSV terlebih dahulu.")
    source = file.name if hasattr(file, "name") else file
    raw = pd.read_csv(source)
    mapping = detect(raw)
    if "label" in mapping:
        raw = raw.drop(columns=[mapping["label"]])
        mapping = detect(raw)
    tmp = raw.copy()
    tmp["__temporary_label"] = 0
    mapping["label"] = "__temporary_label"
    features, _, _, notes = build_features(tmp, mapping, "upload")
    features = features.reindex(columns=ARTIFACT["columns"], fill_value=0)
    features = features.replace([float("inf"), float("-inf")], 0).fillna(0)
    probability = ARTIFACT["model"].predict_proba(features)[:, 1]
    result = pd.DataFrame({
        "username": raw[mapping["screen_name"]].astype(str).values
        if "screen_name" in mapping else range(1, len(raw) + 1),
        "bot_probability": probability.round(4),
    })
    result["signal_level"] = pd.cut(probability, [-1, .35, .65, 1.01], labels=["rendah", "sedang", "tinggi"])
    return result


demo = gr.Interface(
    fn=predict_csv,
    inputs=gr.File(file_types=[".csv"], label="CSV akun tanpa label"),
    outputs=gr.Dataframe(label="Hasil prediksi"),
    title="NusaBotGuard-ML",
    description="Analisis sinyal statistik akun bot-like. Hasil bukan bukti identitas atau niat akun.",
)

if __name__ == "__main__":
    demo.launch()
