"""Unsupervised Indonesian content/coordination report (no bot labels assumed)."""
import sys
from pathlib import Path
import pandas as pd
from features import coordination_features, content_features

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../data/data_bot3/techbro_tweets.csv")
d = pd.read_csv(p)
text = next(c for c in ("text", "tweet", "content") if c in d)
out = pd.concat([content_features(d[text]), coordination_features(d, text, "created_at" if "created_at" in d else None)], axis=1)
out.insert(0, "user_handle", d["user_handle"] if "user_handle" in d else "unknown")
out.to_csv("results/indonesia_coordination_features.csv", index=False)
summary = out.groupby("user_handle", dropna=False).agg({"copy_paste_count":"max","co_hashtag_count":"max","co_retweet_count":"max","same_window_count":"max","text_len":"mean"}).sort_values(["copy_paste_count","co_hashtag_count","same_window_count"], ascending=False)
summary.to_csv("results/indonesia_coordination_summary.csv")
print(f"records={len(d)} accounts={summary.shape[0]}")
print(summary.head(10).to_string())
