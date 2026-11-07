"""Run lightweight end-to-end checks without pytest."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import prepare

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../data/dataset1/train.json")
d = prepare(p)
assert d["n"] > 0
assert len(d["X"]) == len(d["y"]) == d["n"]
assert len(d["X_coord"]) == d["n"]
assert d["y"].isin([0, 1]).all()
print(f"OK: {d['name']} rows={d['n']} profile_features={d['X'].shape[1]} coordination_features={d['X_coord'].shape[1]}")
