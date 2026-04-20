from pathlib import Path
import shutil

BASE_DIR = Path(r"C:\Users\ragha\OneDrive\سطح المكتب\industrial_safety_dataset")

SOURCE = BASE_DIR / "raw_data" / "ladder_main"
MERGED = BASE_DIR / "merged_ladder_dataset"

SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}


def copy_split_files():
    for src_split, dst_split in SPLIT_MAP.items():
        src_images = SOURCE / src_split / "images"
        src_labels = SOURCE / src_split / "labels"

        if not src_images.exists() or not src_labels.exists():
            continue

        dst_images = MERGED / "images" / dst_split
        dst_labels = MERGED / "labels" / dst_split

        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        image_files = list(src_images.glob("*"))
        copied_count = 0

        for img_path in image_files:
            if not img_path.is_file():
                continue

            stem = img_path.stem
            suffix = img_path.suffix

            label_path = src_labels / f"{stem}.txt"
            if not label_path.exists():
                continue

            dst_img_path = dst_images / img_path.name
            dst_lbl_path = dst_labels / label_path.name

            shutil.copy2(img_path, dst_img_path)
            shutil.copy2(label_path, dst_lbl_path)
            copied_count += 1

        print(f"{src_split} -> {dst_split} | copied {copied_count} pairs")


def write_data_yaml():
    yaml_text = """train: ../merged_ladder_dataset/images/train
val: ../merged_ladder_dataset/images/val
test: ../merged_ladder_dataset/images/test

nc: 1
names:
  0: ladder
"""
    yaml_path = MERGED / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)


def main():
    copy_split_files()
    write_data_yaml()
    print("\nDone. Ladder dataset prepared at:")
    print(MERGED)


if __name__ == "__main__":
    main()