from pathlib import Path

# =========================
# CHANGE THESE PATHS ONLY
# =========================
DATASET_DIR = Path(r"C:\Users\ragha\OneDrive\سطح المكتب\industrial_safety_dataset\raw_data\ppe_goggles_extra")

# old_id -> new_id
CLASS_MAPPING = {
    0: 7,   # gloves -> gloves
    1: 5,   # goggles -> goggles
    2: 8,   # no_gloves -> no_gloves
    3: 6,   # no_goggles -> no_goggles
}

NEW_NAMES = [
    "person",
    "helmet",
    "no_helmet",
    "vest",
    "no_vest",
    "goggles",
    "no_goggles",
    "gloves",
    "no_gloves",
]


def remap_label_file(label_path: Path, mapping: dict):
    if not label_path.exists():
        return

    new_lines = []

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        try:
            old_class_id = int(parts[0])
        except:
            continue

        if old_class_id not in mapping:
            continue

        new_class_id = mapping[old_class_id]
        new_line = " ".join([str(new_class_id)] + parts[1:])
        new_lines.append(new_line)

    with open(label_path, "w", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line + "\n")


def update_data_yaml(dataset_dir: Path, new_names: list):
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        print(f"data.yaml not found in {dataset_dir}")
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    updated_lines = []
    names_written = False
    nc_written = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("nc:"):
            updated_lines.append(f"nc: {len(new_names)}\n")
            nc_written = True
        elif stripped.startswith("names:"):
            updated_lines.append("names:\n")
            for i, name in enumerate(new_names):
                updated_lines.append(f"  {i}: {name}\n")
            names_written = True
        else:
            updated_lines.append(line)

    if not nc_written:
        updated_lines.append(f"\nnc: {len(new_names)}\n")

    if not names_written:
        updated_lines.append("names:\n")
        for i, name in enumerate(new_names):
            updated_lines.append(f"  {i}: {name}\n")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.writelines(updated_lines)


def main():
    label_files = list(DATASET_DIR.rglob("*.txt"))

    for label_file in label_files:
        remap_label_file(label_file, CLASS_MAPPING)

    update_data_yaml(DATASET_DIR, NEW_NAMES)

    print(f"Done. Remapped {len(label_files)} label files in:")
    print(DATASET_DIR)


if __name__ == "__main__":
    main()