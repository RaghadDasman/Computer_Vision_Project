from pathlib import Path

BASE_DIR = Path(r"C:\Users\ragha\OneDrive\سطح المكتب\industrial_safety_dataset")
MERGED = BASE_DIR / "merged_id_dataset"
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def remove_empty_labels(split_name: str):
    img_dir = MERGED / "images" / split_name
    lbl_dir = MERGED / "labels" / split_name

    removed = 0

    for label_file in lbl_dir.glob("*.txt"):
        content = label_file.read_text(encoding="utf-8").strip()

        if content == "":
            stem = label_file.stem

            label_file.unlink(missing_ok=True)

            for ext in IMAGE_EXTS:
                img_file = img_dir / f"{stem}{ext}"
                if img_file.exists():
                    img_file.unlink()
                    break

            removed += 1

    print(f"{split_name}: removed {removed} empty-label pairs")


def main():
    for split in ["train", "val", "test"]:
        remove_empty_labels(split)


if __name__ == "__main__":
    main()