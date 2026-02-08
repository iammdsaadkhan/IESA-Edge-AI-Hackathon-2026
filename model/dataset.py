import os
from collections import Counter

DATASET_DIR = "dataset"

splits = ["train", "validation", "test"]

def count_images(split):
    split_path = os.path.join(DATASET_DIR, split)
    class_counts = {}

    for cls in sorted(os.listdir(split_path)):
        cls_path = os.path.join(split_path, cls)
        if os.path.isdir(cls_path):
            class_counts[cls] = len([
                f for f in os.listdir(cls_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
    return class_counts


if __name__ == "__main__":
    for split in splits:
        counts = count_images(split)
        print(f"\n📂 {split.upper()} SET")
        total = 0
        for cls, cnt in counts.items():
            print(f"{cls:20s}: {cnt}")
            total += cnt
        print(f"Total images ({split}): {total}")
