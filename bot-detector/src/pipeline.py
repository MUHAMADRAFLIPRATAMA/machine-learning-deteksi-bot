"""Pipeline deteksi akun bot - lengkap.

Jalankan dari folder proyek (bot-detector):
    python src\\pipeline.py data\\dataset1\\file_a.csv data\\dataset2\\file_b.csv

Contoh: dataset 1 = CSV, dataset 2 = JSON:
    python src\\pipeline.py data\\dataset1\\akun.csv data\\dataset2\\akun.json

Format yang didukung: CSV, TSV, Excel, JSON (list, {"data": [...]}, {id: {...}},
bertingkat), dan JSON Lines (.jsonl). Kalau satu dataset terdiri dari beberapa file
(misal node.json + label.csv), berikan FOLDER-nya; file digabung lewat kolom id:
    python src\\pipeline.py data\\dataset1\\akun.csv data\\dataset2

Atau cukup satu dataset (tanpa uji lintas dataset):
    python src\\pipeline.py data\\dataset1\\file_a.csv

Keluaran (folder results\\):
    results.csv          tabel perbandingan semua model & skenario
    pr_curve_*.png       kurva precision-recall
    shap_summary.png     (jika shap & xgboost terpasang)
    column_mapping.txt   kolom yang terdeteksi per dataset -> CEK INI
"""
import re
import sys
import warnings
from itertools import permutations

from features import coordination_features
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             f1_score, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
OUT = Path("results")
OUT.mkdir(exist_ok=True)
SEED = 42

# ---------------------------------------------------------------- 1. KOLOM
# Alias nama kolom yang umum di dataset bot Twitter. Tambah sendiri jika perlu.
ALIASES = {
    "label": ["label", "bot", "is_bot", "isbot", "class", "target", "account_type", "type"],
    "followers": ["followers_count", "followers", "follower_count", "num_followers"],
    "friends": ["friends_count", "following", "friends", "following_count", "followees_count"],
    "statuses": ["statuses_count", "tweets", "tweet_count", "statuses", "num_tweets", "posts"],
    "favourites": ["favourites_count", "favorites_count", "likes", "favourites"],
    "listed": ["listed_count", "listed"],
    "verified": ["verified", "is_verified"],
    "default_profile": ["default_profile", "default_profile_image_flag"],
    "default_image": ["default_profile_image", "default_image"],
    "has_url": ["url", "has_url", "profile_url"],
    "created_at": ["created_at", "account_creation", "creation_date", "created"],
    "screen_name": ["screen_name", "username", "user_name", "handle"],
    "name": ["name", "display_name"],
    "description": ["description", "bio", "profile_description"],
    "text": ["text", "tweet", "tweets_text", "content"],
}
BOT_WORDS = {"bot", "automated", "spam", "fake", "sybil", "cyborg"}


ID_COLS = ["id", "user_id", "id_str", "userid", "account_id", "profile_id"]


def _read_json_file(p):
    """JSON list/dict/dict-of-dicts, JSON Lines, atau JSON bertingkat -> DataFrame datar."""
    import json
    raw = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            raw = p.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:  # kemungkinan JSON Lines
        obj = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    if isinstance(obj, dict):
        # {"data": [...]} atau {"users": [...]}
        lists = [v for v in obj.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if lists:
            obj = max(lists, key=len)
        elif all(isinstance(v, dict) for v in obj.values()):
            obj = [{"id": k, **v} for k, v in obj.items()]  # {id: {...}}
        else:
            obj = [obj]
    return pd.json_normalize(obj)


def _flatten(df):
    """Nama kolom bertingkat 'profile.followers_count' -> 'followers_count'; list -> teks."""
    new, seen = [], set()
    for c in df.columns:
        short = str(c).split(".")[-1]
        if short in seen:
            short = str(c).replace(".", "_")
        seen.add(short)
        new.append(short)
    df.columns = new
    for c in df.columns:
        if df[c].map(lambda v: isinstance(v, list)).any():
            df[c] = df[c].map(lambda v: " ".join(map(str, v)) if isinstance(v, list) else v)
    return df.loc[:, ~df.columns.duplicated()]


def _read_one(p):
    ext = p.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return _flatten(pd.read_excel(p))
    if ext in (".json", ".jsonl", ".ndjson"):
        return _flatten(_read_json_file(p))
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return _flatten(pd.read_csv(p, sep=None, engine="python", encoding=enc))
        except Exception:
            pass
    raise ValueError(f"Tidak bisa membaca {p}")


def read_any(path):
    """Baca satu file, atau satu FOLDER berisi beberapa file (digabung lewat kolom id)."""
    p = Path(path)
    if not p.is_dir():
        return _read_one(p)
    exts = (".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xlsx", ".xls")
    files = [f for f in sorted(p.rglob("*")) if f.suffix.lower() in exts]
    if not files:
        raise ValueError(f"Tidak ada file data di folder {p}")
    frames = [(f, _read_one(f)) for f in files]
    if len(frames) == 1:
        return frames[0][1]

    def idcol(df):
        low = {c.lower(): c for c in df.columns}
        return next((low[i] for i in ID_COLS if i in low), None)

    # mulai dari tabel terbesar, gabungkan tabel lain yang punya kolom id
    frames.sort(key=lambda t: -len(t[1]))
    base = frames[0][1]
    bid = idcol(base)
    if bid is None:
        print(f"  [{p.name}] kolom id tidak ada di {frames[0][0].name}; hanya file itu dipakai")
        return base
    base[bid] = base[bid].astype(str)
    for f, df in frames[1:]:
        did = idcol(df)
        if did is None:
            print(f"  (dilewati, tidak ada kolom id: {f.name})")
            continue
        df = df.copy()
        df[did] = df[did].astype(str)
        df = df.rename(columns={did: bid}).drop_duplicates(bid)
        new_cols = [c for c in df.columns if c == bid or c not in base.columns]
        base = base.merge(df[new_cols], on=bid, how="left")
        print(f"  digabung: {f.name} (+{len(new_cols) - 1} kolom)")
    return base


def detect(df):
    # Normalisasi nama kolom agar format seperti ``Bot Label`` dan
    # ``bot_label`` diperlakukan sama.
    def norm(value):
        return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")

    lower = {norm(c): c for c in df.columns}
    aliases = {key: [norm(n) for n in names] for key, names in ALIASES.items()}
    aliases["label"] += ["bot_label"]
    aliases["followers"] += ["follower_count"]
    aliases["text"] += ["tweet"]
    aliases["created_at"] += ["created_at"]
    mapping = {}
    for key, names in aliases.items():
        for n in names:
            if n in lower:
                mapping[key] = lower[n]
                break
    return mapping


def to_binary_label(s):
    """Ubah label apapun (0/1, 'bot'/'human', True/False) jadi 1=bot, 0=manusia."""
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        u = sorted(s.dropna().unique())
        if set(u) <= {0, 1}:
            return s.astype(int)
        return (s == max(u)).astype(int)
    t = s.astype(str).str.lower().str.strip()
    return t.apply(lambda v: int(any(w in v for w in BOT_WORDS) or v in {"1", "true", "yes"}))


def to_flag(s):
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0) > 0).astype(int)
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"]).astype(int)


# ------------------------------------------------------------- 2. FITUR
def build_features(df, m, name):
    """Kembalikan (X_numerik DataFrame, teks Series, y Series, catatan)."""
    notes = []
    if "label" not in m:
        raise SystemExit(f"[{name}] Kolom label tidak terdeteksi. Isi ALIASES['label'] "
                         f"dengan nama kolom label: {list(df.columns)}")
    y = to_binary_label(df[m["label"]])
    f = pd.DataFrame(index=df.index)

    def num(key):
        return pd.to_numeric(df[m[key]], errors="coerce").fillna(0) if key in m else None

    for key in ("followers", "friends", "statuses", "favourites", "listed"):
        v = num(key)
        if v is not None:
            f[key] = v
        else:
            notes.append(f"kolom '{key}' tidak ada")

    if {"followers", "friends"} <= set(f):
        f["follow_ratio"] = f["followers"] / (f["friends"] + 1)
        f["log_followers"] = np.log1p(f["followers"])
        f["log_friends"] = np.log1p(f["friends"])
    if "statuses" in f:
        f["log_statuses"] = np.log1p(f["statuses"])

    for key in ("verified", "default_profile", "default_image"):
        if key in m:
            f[key] = to_flag(df[m[key]])
    if "has_url" in m:
        f["has_url"] = df[m["has_url"]].notna().astype(int) & \
            (df[m["has_url"]].astype(str).str.len() > 3).astype(int)

    # umur akun & tweet per hari
    if "created_at" in m:
        dt = pd.to_datetime(df[m["created_at"]], errors="coerce", utc=True)
        age = (dt.max() - dt).dt.days.fillna(dt.dropna().pipe(lambda x: (dt.max() - x).dt.days.median()))
        f["age_days"] = age.clip(lower=1)
        if "statuses" in f:
            f["tweets_per_day"] = f["statuses"] / f["age_days"]
        if "followers" in f:
            f["followers_per_day"] = f["followers"] / f["age_days"]
    else:
        notes.append("kolom tanggal pembuatan tidak ada")

    # fitur dari nama & bio (profil)
    if "screen_name" in m:
        sn = df[m["screen_name"]].fillna("").astype(str)
        f["sn_len"] = sn.str.len()
        f["sn_digits"] = sn.apply(lambda s: sum(c.isdigit() for c in s))
        f["sn_digit_ratio"] = f["sn_digits"] / (f["sn_len"] + 1)
    desc = df[m["description"]].fillna("").astype(str) if "description" in m else None
    if desc is not None:
        f["bio_len"] = desc.str.len()
        f["bio_empty"] = (desc.str.strip() == "").astype(int)
        f["bio_urls"] = desc.str.count(r"https?://")
        f["bio_hashtags"] = desc.str.count(r"#\w+")
        f["bio_mentions"] = desc.str.count(r"@\w+")

    # teks untuk TF-IDF (bio + tweet kalau ada)
    parts = []
    if desc is not None:
        parts.append(desc)
    if "text" in m:
        parts.append(df[m["text"]].fillna("").astype(str))
    text = (parts[0] if len(parts) == 1 else parts[0] + " " + parts[1]) if parts else None
    if text is not None:
        text = text.apply(lambda t: re.sub(r"https?://\S+", " URL ", t))
    else:
        notes.append("tidak ada kolom teks -> TF-IDF dilewati")

    return f, text, y, notes


# ------------------------------------------------------------- 3. MODEL
def get_models():
    models = {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               n_jobs=-1, random_state=SEED),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                                          subsample=0.9, colsample_bytree=0.9,
                                          eval_metric="logloss", random_state=SEED)
    except ImportError:
        print("  (xgboost tidak terpasang, dilewati)")
    return models


def evaluate(model, Xte, yte):
    p = model.predict_proba(Xte)[:, 1]
    return {
        "PR-AUC": average_precision_score(yte, p),
        "ROC-AUC": roc_auc_score(yte, p),
        "F1": f1_score(yte, (p >= 0.5).astype(int)),
    }, p


def fit_eval(name_train, Xtr, ytr, name_test, Xte, yte, rows, curves, tag):
    for mn, model in get_models().items():
        model.fit(Xtr, ytr)
        r, p = evaluate(model, Xte, yte)
        rows.append({"skenario": tag, "train": name_train, "test": name_test,
                     "model": mn, **{k: round(v, 4) for k, v in r.items()}})
        pr, rc, _ = precision_recall_curve(yte, p)
        curves.setdefault(tag, []).append((mn, rc, pr, r["PR-AUC"]))
        print(f"  {tag:<22} {mn:<13} PR-AUC={r['PR-AUC']:.3f}  F1={r['F1']:.3f}")


def plot_curves(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for tag, items in curves.items():
        plt.figure(figsize=(5.5, 4.5))
        for mn, rc, pr, ap in items:
            plt.plot(rc, pr, label=f"{mn} (AP={ap:.3f})")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(tag); plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / f"pr_curve_{re.sub(r'[^A-Za-z0-9]+', '_', tag)}.png", dpi=130)
        plt.close()


def shap_plot(Xtr, ytr, Xte):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
        from xgboost import XGBClassifier
    except ImportError:
        print("  (shap/xgboost tidak terpasang, SHAP dilewati)")
        return
    m = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, eval_metric="logloss",
                      random_state=SEED).fit(Xtr, ytr)
    sv = shap.TreeExplainer(m).shap_values(Xte.iloc[:1000])
    shap.summary_plot(sv, Xte.iloc[:1000], show=False)
    plt.tight_layout(); plt.savefig(OUT / "shap_summary.png", dpi=130); plt.close()


# ----------------------------------------------------------------- MAIN
def prepare(path):
    name = Path(path).parent.name + "/" + Path(path).stem
    df = read_any(path)
    m = detect(df)
    f, text, y, notes = build_features(df, m, name)
    base = f.copy()
    # Sinyal koordinasi dihitung dari seluruh observasi yang tersedia, lalu
    # digabung sebagai fitur tambahan. Ini bukan label dan tidak mengubah y.
    if "text" in m:
        raw = pd.DataFrame({"text": df[m["text"]].fillna("").astype(str)})
        time_col = m.get("created_at")
        if time_col:
            raw["created_at"] = df[time_col]
        coord = coordination_features(raw, "text", "created_at" if time_col else None)
        f = pd.concat([f.reset_index(drop=True), coord.reset_index(drop=True)], axis=1)
    return {"name": name, "X": base, "X_coord": f, "text": text, "y": y, "map": m, "notes": notes, "n": len(df)}


def with_text(train, test, max_feat=300):
    """Tambahkan fitur TF-IDF (dilatih HANYA dari data train)."""
    if train["text"] is None or test["text"] is None:
        return train["X"], test["X"], False
    tf = TfidfVectorizer(max_features=max_feat, min_df=3, ngram_range=(1, 2), stop_words="english")
    a = tf.fit_transform(train["text"]).toarray()
    b = tf.transform(test["text"]).toarray()
    cols = [f"tfidf_{w}" for w in tf.get_feature_names_out()]
    return (pd.concat([train["X"].reset_index(drop=True), pd.DataFrame(a, columns=cols)], axis=1),
            pd.concat([test["X"].reset_index(drop=True), pd.DataFrame(b, columns=cols)], axis=1), True)


def align(a, b):
    cols = [c for c in a.columns if c in b.columns]
    return a[cols], b[cols]


def main(paths):
    if not paths:
        sys.exit(__doc__)
    ds = [prepare(p) for p in paths]

    with open(OUT / "column_mapping.txt", "w", encoding="utf-8") as fh:
        for d in ds:
            fh.write(f"{d['name']}  ({d['n']} baris, bot={d['y'].mean():.1%})\n")
            for k, v in d["map"].items():
                fh.write(f"   {k:<16} <- {v}\n")
            fh.write(f"   catatan: {d['notes'] or '-'}\n\n")
    print(open(OUT / "column_mapping.txt", encoding="utf-8").read())

    rows, curves = [], {}

    # A. dalam-dataset (split stratified)
    for tag_suffix, use_text, use_coord in (("profil", False, False), ("profil+teks", True, False), ("profil+teks+koordinasi", True, True)):
        for d in ds:
            idx_tr, idx_te = train_test_split(np.arange(d["n"]), test_size=0.25,
                                              stratify=d["y"], random_state=SEED)
            sub = lambda i: {"X": (d["X_coord"] if use_coord else d["X"]).iloc[i], "text": None if d["text"] is None else d["text"].iloc[i],
                             "y": d["y"].iloc[i]}
            tr, te = sub(idx_tr), sub(idx_te)
            if use_text:
                Xtr, Xte, ok = with_text(tr, te)
                if not ok:
                    continue
            else:
                Xtr, Xte = tr["X"], te["X"]
            fit_eval(d["name"], Xtr, tr["y"], d["name"], Xte, te["y"], rows, curves,
                     f"dalam {d['name'][:14]} {tag_suffix}")

    # B. LINTAS dataset (hasil terpenting untuk klaim generalisasi)
    for a0, b0 in permutations(ds, 2):
        for tag_suffix, use_text, use_coord in (("profil", False, False), ("profil+teks", True, False), ("profil+teks+koordinasi", True, True)):
            a = {**a0, "X": a0["X_coord"] if use_coord else a0["X"]}
            b = {**b0, "X": b0["X_coord"] if use_coord else b0["X"]}
            if use_text:
                Xtr, Xte, ok = with_text(a, b)
                if not ok:
                    continue
            else:
                Xtr, Xte = a["X"], b["X"]
            Xtr, Xte = align(Xtr, Xte)
            fit_eval(a["name"], Xtr, a["y"].reset_index(drop=True), b["name"], Xte,
                     b["y"].reset_index(drop=True), rows, curves,
                     f"LINTAS {a['name'][:10]}->{b['name'][:10]} {tag_suffix}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results.csv", index=False)
    plot_curves(curves)
    d0 = ds[0]
    Xtr, Xte = align(d0["X"], d0["X"])
    shap_plot(Xtr, d0["y"], Xte)

    print("\n=== RINGKASAN (PR-AUC) ===")
    print(res.pivot_table(index=["skenario"], columns="model", values="PR-AUC").round(3).to_string())
    print(f"\nTersimpan di folder: {OUT.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1:])