import csv
import os

CSV_TEST_FIELDS = [
    "video_name",
    "model",
    "frames",
    "transformation",
    "total_num",
    "RR_avg",
    "RI_avg",
    "IE_avg"
]

CSV_ATTACK_FIELDS = [
    "video_name",
    "transformation",
    "success",
    "query",
]

PRINT_ORDER = [
    "add_one_frame",
    "add_two_frames",
    "delete_one_frame",
    "delete_two_frames",
    "replace_one_frame",
    "replace_two_frames",
    "reverse",
    "random_shuffle",
]

def load_existing_transformations(csv_path, model_name, frames):
    tested = set()
    if not os.path.exists(csv_path):
        return tested

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"] == model_name and int(row["frames"]) == frames:
                tested.add(row["transformation"])
    return tested

def append_csv(csv_path, row):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_TEST_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def init_attack_csv(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_ATTACK_FIELDS)
            writer.writeheader()

def load_finished_attacks(csv_path):
    finished = set()
    if not os.path.exists(csv_path):
        return finished

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            finished.add((row["video_name"], row["transformation"]))
    return finished

def append_attack_csv(csv_path, record):
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_ATTACK_FIELDS)
        writer.writerow(record)

def compute_asr_aq(csv_path):
    stats = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["transformation"]
            success = int(row["success"])
            query = int(row["query"])

            if name not in stats:
                stats[name] = {"total": 0, "success": 0, "query": 0}

            stats[name]["total"] += 1
            stats[name]["success"] += success
            if success:
                stats[name]["query"] += query

    for name in PRINT_ORDER:
        if name not in stats:
            continue

        s = stats[name]
        asr = s["success"] / s["total"]
        aq = s["query"] / s["success"] if s["success"] > 0 else 0

        print(name, "ASR:", round(asr, 4), "Average Query:", round(aq, 2))
