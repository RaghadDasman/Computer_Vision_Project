from pathlib import Path
import shutil

BASE_DIR = Path(r"C:\Users\ragha\OneDrive\سطح المكتب\industrial_safety_dataset")

DATASET_1 = BASE_DIR / "raw_data" / "id_bzx4h"
DATASET_2 = BASE_DIR / "raw_data" / "id_rdler"
MERGED = BASE_DIR / "merged_id_dataset"

DATASETS = [
    {"path": DATASET_1, "prefix": "id1"},
    {"path": DATASET_2, "prefix": "id2"},
]

SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}


def copy_split_files(dataset_path: Path, prefix: str):
    for src_split, dst_split in SPLIT_MAP.items():
        src_images = dataset_path / src_split / "images"
        src_labels = dataset_path / src_split / "labels"

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

            new_name = f"{prefix}_{stem}"

            dst_img_path = dst_images / f"{new_name}{suffix}"
            dst_lbl_path = dst_labels / f"{new_name}.txt"

            shutil.copy2(img_path, dst_img_path)
            shutil.copy2(label_path, dst_lbl_path)
            copied_count += 1

        print(f"{dataset_path.name} | {src_split} -> {dst_split} | copied {copied_count} pairs")


def write_data_yaml():
    yaml_text = """train: ../merged_id_dataset/images/train
val: ../merged_id_dataset/images/val
test: ../merged_id_dataset/images/test

nc: 1
names:
  0: worker_id_tag
"""
    yaml_path = MERGED / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)


def main():
    for ds in DATASETS:
        copy_split_files(ds["path"], ds["prefix"])

    write_data_yaml()
    print("\nDone. Merged ID dataset created at:")
    print(MERGED)


if __name__ == "__main__":
    main()