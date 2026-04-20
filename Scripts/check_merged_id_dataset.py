from pathlib import Path

BASE_DIR = Path(r"C:\Users\ragha\OneDrive\سطح المكتب\industrial_safety_dataset")
MERGED = BASE_DIR / "merged_id_dataset"

ALLOWED_CLASS_IDS = {0}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_split(split_name: str):
    img_dir = MERGED / "images" / split_name
    lbl_dir = MERGED / "labels" / split_name

    image_files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    label_files = [p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]

    image_stems = {p.stem for p in image_files}
    label_stems = {p.stem for p in label_files}

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    invalid_class_lines = []
    empty_label_files = []

    for label_file in label_files:
        with open(label_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if not lines:
            empty_label_files.append(label_file.name)
            continue

        for line_num, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) < 5:
                invalid_class_lines.append((label_file.name, line_num, line, "less than 5 values"))
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                invalid_class_lines.append((label_file.name, line_num, line, "class id is not int"))
                continue

            if class_id not in ALLOWED_CLASS_IDS:
                invalid_class_lines.append((label_file.name, line_num, line, "class id out of range"))

    print(f"\n===== {split_name.upper()} =====")
    print(f"Images: {len(image_files)}")
    print(f"Labels: {len(label_files)}")
    print(f"Images without labels: {len(missing_labels)}")
    print(f"Labels without images: {len(missing_images)}")
    print(f"Empty label files: {len(empty_label_files)}")
    print(f"Invalid label lines: {len(invalid_class_lines)}")

    if missing_labels[:10]:
        print("\nSample images without labels:")
        for x in missing_labels[:10]:
            print(" ", x)

    if missing_images[:10]:
        print("\nSample labels without images:")
        for x in missing_images[:10]:
            print(" ", x)

    if empty_label_files[:10]:
        print("\nSample empty label files:")
        for x in empty_label_files[:10]:
            print(" ", x)

    if invalid_class_lines[:10]:
        print("\nSample invalid label lines:")
        for item in invalid_class_lines[:10]:
            print(" ", item)


def check_data_yaml():
    yaml_path = MERGED / "data.yaml"
    print("\n===== DATA.YAML =====")
    if yaml_path.exists():
        print("data.yaml exists")
        print(yaml_path)
    else:
        print("data.yaml is missing")


def main():
    check_data_yaml()
    for split in ["train", "val", "test"]:
        check_split(split)


if __name__ == "__main__":
    main()