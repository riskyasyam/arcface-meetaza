import base64, pathlib

def to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

ref_b64 = to_b64("dataset_static/billie/dummy1.jpg")
tgt_b64 = to_b64("dataset_static/madison/dummy2.jpg")

pathlib.Path("ref_b64.txt").write_text(ref_b64, encoding="utf-8")
pathlib.Path("tgt_b64.txt").write_text(tgt_b64, encoding="utf-8")

print("Wrote ref_b64.txt & tgt_b64.txt (full base64).")