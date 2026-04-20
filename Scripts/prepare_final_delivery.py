from pathlib import Path
import shutil

BASE_DIR = Path(r"C:\Users\ragha\OneDrive\سطح المكتب\industrial_safety_dataset")

DEST = BASE_DIR / "final_delivery"

PPE = BASE_DIR / "merged_dataset"
ID = BASE_DIR / "merged_id_dataset"
LADDER = BASE_DIR / "merged_ladder_dataset"
DOCS = BASE_DIR / "docs"


def copy_folder(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"Copied: {src.name} -> {dst}")


def main():
    # create main folder
    DEST.mkdir(exist_ok=True)

    # copy datasets
    copy_folder(PPE, DEST / "ppe" / "merged_dataset")
    copy_folder(ID, DEST / "id" / "merged_id_dataset")
    copy_folder(LADDER, DEST / "ladder" / "merged_ladder_dataset")

    # copy docs
    copy_folder(DOCS, DEST / "docs")

    print("\n🔥 Final delivery folder is ready:")
    print(DEST)


if __name__ == "__main__":
    main()