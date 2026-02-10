import json
import os


VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"]


def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)

def write_json(path, results):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def list_videos(input_folder):
    paths = []
    for name in os.listdir(input_folder):
        full = os.path.join(input_folder, name)
        if not os.path.isfile(full):
            continue
        low = name.lower()
        if any(low.endswith(ext) for ext in VIDEO_EXTS):
            paths.append(full)
    paths = sorted(paths)
    return paths
