from pathlib import Path
from collections import Counter

BASE_DIR = Path(r"C:\Users\ragha\OneDrive\سطح المكتب\industrial_safety_dataset")
MERGED = BASE_DIR / "merged_dataset"

CLASS_NAMES = {
    0: "person",
    1: "helmet",
    2: "no_helmet",
    3: "vest",
    4: "no_vest",
    5: "goggles",
    6: "no_goggles",
    7: "gloves",
    8: "no_gloves",
}


def count_split(split_name: str):
    lbl_dir = MERGED / "labels" / split_name
    counter = Counter()

    for label_file in lbl_dir.glob("*.txt"):
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                counter[class_id] += 1

    print(f"\n===== {split_name.upper()} =====")
    total = 0
    for class_id in sorted(CLASS_NAMES.keys()):
        count = counter[class_id]
        total += count
        print(f"{class_id} - {CLASS_NAMES[class_id]}: {count}")

    print(f"Total objects: {total}")


def main():
    for split in ["train", "val", "test"]:
        count_split(split)


if __name__ == "__main__":
    main()