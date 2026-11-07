"""Streaming inspection for TwiBot-20 support.json inside a ZIP.
Does not extract the 5+ GB file or load it into RAM.
"""
import json, sys, zipfile
from collections import Counter


def objects(stream, chunk_size=1024 * 1024):
    decoder = json.JSONDecoder(); buf = ""; first = True
    while True:
        part = stream.read(chunk_size)
        if not part: break
        buf += part.decode("utf-8", "replace")
        pos = 0
        while True:
            while pos < len(buf) and buf[pos] in "[\r\n \t,": pos += 1
            if pos >= len(buf): break
            try: obj, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError: break
            yield obj; pos = end
        buf = buf[pos:]


def main(path):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("support.json"))
        count = 0; domains = Counter(); has_neighbor = 0; keys = set()
        with z.open(name) as stream:
            for item in objects(stream):
                count += 1; keys.update(item)
                domains.update(item.get("domain") or [])
                has_neighbor += bool(item.get("neighbor"))
        print(json.dumps({"file": name, "records": count,
                          "top_domains": domains.most_common(20),
                          "records_with_neighbor": has_neighbor,
                          "keys": sorted(keys)}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("usage: python src/inspect_support.py data_bot4.zip")
    main(sys.argv[1])
