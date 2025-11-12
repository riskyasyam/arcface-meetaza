import base64, pathlib

def to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

current_b64 = to_b64("../dataset/current/dummy2.jpg")
profile_b64 = to_b64("../dataset/profile/dummy1.jpg")

pathlib.Path("output/current_b64.txt").write_text(current_b64, encoding="utf-8")
pathlib.Path("output/profile_b64.txt").write_text(profile_b64, encoding="utf-8")
print("Success writing base64 files")