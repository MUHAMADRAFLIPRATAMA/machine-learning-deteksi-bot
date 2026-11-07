"""Local demo server: static dashboard + CSV prediction endpoint."""
import json, tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

ROOT = Path(__file__).parent
SRC = ROOT.parent / "src"
sys.path.insert(0, str(SRC))
from predict_account import predict

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)
    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404); return
        n = int(self.headers.get("Content-Length", 0))
        if n > 2_000_000:
            self.send_error(413, "CSV terlalu besar"); return
        body = self.rfile.read(n)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                f.write(body); tmp = f.name
            result, _ = predict("../data/dataset1/train.json", tmp)
            payload = {"rows": len(result), "predictions": json.loads(result.to_json(orient="records"))}
            raw = json.dumps(payload).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
        except Exception as e:
            raw = json.dumps({"error": str(e)}).encode(); self.send_response(400); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
        finally:
            if tmp: Path(tmp).unlink(missing_ok=True)

if __name__ == "__main__":
    print("Dashboard: http://127.0.0.1:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
