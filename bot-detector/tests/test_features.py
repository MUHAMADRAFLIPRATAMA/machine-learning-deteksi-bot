import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import pandas as pd
from features import coordination_features


def test_duplicate_text_is_counted_per_row():
    d = pd.DataFrame({"text": ["same", "same", "other"], "created_at": ["2024-01-01T00:00:00Z"] * 3})
    got = coordination_features(d, "text", "created_at")
    assert got["copy_paste_count"].tolist() == [1, 1, 0]


def test_missing_timestamp_keeps_coordination_columns():
    got = coordination_features(pd.DataFrame({"text": ["a"]}), "text")
    assert set(got) == {"copy_paste_count", "co_hashtag_count", "co_retweet_count", "same_window_count"}
