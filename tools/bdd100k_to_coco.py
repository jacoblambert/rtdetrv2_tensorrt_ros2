import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm

parser = argparse.ArgumentParser(description='bdd100k -> COCO (custom 8-class mapping)')
parser.add_argument('--bdd_dir', type=str, default='E:\\bdd100k')
parser.add_argument('--train_json', type=str, default=None, help='Path to bdd100k_labels_images_train.json (overrides default)')
parser.add_argument('--val_json', type=str, default=None, help='Path to bdd100k_labels_images_val.json (overrides default)')
parser.add_argument('--out_dir', type=str, default=None, help='Output dir (default: <bdd_dir>/labels_coco)')
cfg = parser.parse_args()

# Resolve input/output paths
src_train = cfg.train_json or os.path.join(cfg.bdd_dir, 'labels', 'bdd100k_labels_images_train.json')
src_val   = cfg.val_json   or os.path.join(cfg.bdd_dir, 'labels', 'bdd100k_labels_images_val.json')
out_dir   = cfg.out_dir    or os.path.join(cfg.bdd_dir, 'labels_coco')
os.makedirs(out_dir, exist_ok=True)
dst_train = os.path.join(out_dir, 'bdd100k_labels_images_train_coco.json')
dst_val   = os.path.join(out_dir, 'bdd100k_labels_images_val_coco.json')

TARGET_CATEGORIES = [
    {"id": 1, "name": "car",         "supercategory": "vehicle"},
    {"id": 2, "name": "truck",       "supercategory": "vehicle"},
    {"id": 3, "name": "bus",         "supercategory": "vehicle"},
    {"id": 4, "name": "trailer",     "supercategory": "vehicle"},
    {"id": 5, "name": "motorcycle",  "supercategory": "vehicle"},
    {"id": 6, "name": "bicycle",     "supercategory": "vehicle"},
    {"id": 7, "name": "pedestrian",  "supercategory": "person"},
    {"id": 8, "name": "animal",      "supercategory": "animal"},
    {"id": 9, "name": "traffic light","supercategory": "traffic"},
    {"id":10, "name": "traffic sign", "supercategory": "traffic"},
]
NAME2ID = {c["name"]: c["id"] for c in TARGET_CATEGORIES}

BDD_TO_TARGET = {
    "car":           "car",
    "truck":         "truck",
    "bus":           "bus",
    "trailer":       "trailer",     # keep if present in your BDD labels; otherwise safely ignored
    "motor":         "motorcycle",
    "motorcycle":    "motorcycle",  # defensive alias
    "bike":          "bicycle",
    "bicycle":       "bicycle",     # defensive alias
    "person":        "pedestrian",
    "pedestrian":    "pedestrian",  # defensive alias
    "rider":         "pedestrian",  # choice: fold rider into pedestrian
    "animal":        "animal",      # not actually in bdd100k
    "other person":  "pedestrian",
    "other vehicle": "truck",
    "traffic light":  "traffic light",
    "traffic sign":   "traffic sign",
}

def bdd2coco_detection(labeled_images, save_path):
    coco = {
        "images": [],
        "annotations": [],
        "categories": TARGET_CATEGORIES,
        "type": "instances",
    }

    ignored_categories = set()
    class_counts = defaultdict(int)

    ann_id = 1
    img_id = 1
    n_none_labels = 0
    n_no_kept = 0

    for img in tqdm(labeled_images, desc=f"Converting -> {os.path.basename(save_path)}"):
        # BDD image record
        labels = img.get("labels") or []
        if labels is None:
            n_none_labels += 1
            labels = []

        image = {
            "file_name": img.get("name"),
            "height": 720,   # BDD100K images are 720x1280 by default
            "width": 1280,
            "id": img_id,
        }

        has_kept_annotations = False

        for lab in labels:
            cat_raw = lab.get("category")
            if not cat_raw:
                continue
            cat = str(cat_raw).strip().lower()

            if cat not in BDD_TO_TARGET:
                ignored_categories.add(cat)
                continue

            box2d = lab.get("box2d")
            if not box2d:
                # skip labels without boxes (e.g., only attributes or polys)
                continue

            x1 = float(box2d["x1"])
            y1 = float(box2d["y1"])
            x2 = float(box2d["x2"])
            y2 = float(box2d["y2"])
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue

            target_name = BDD_TO_TARGET[cat]
            category_id = NAME2ID[target_name]

            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "iscrowd": 0,
                "bbox": [x1, y1, w, h],
                "area": float(w * h),
                "ignore": 0,
                "segmentation": [[x1, y1, x1, y2, x2, y2, x2, y1]],
            }
            coco["annotations"].append(ann)
            ann_id += 1
            has_kept_annotations = True
            class_counts[target_name] += 1

        if has_kept_annotations:
            coco["images"].append(image)
            img_id += 1
        else:
            n_no_kept += 1  # images with labels=None or only unmapped/invalid boxes

    # Summary & save
    print("\nKept class counts:")
    for k in sorted(class_counts.keys()):
        print(f"  {k}: {class_counts[k]}")

    if ignored_categories:
        print("\nIgnored source categories (not mapped):")
        print(" ", sorted(ignored_categories))

    print(f"\nSaving COCO to: {save_path}")
    with open(save_path, "w") as f:
        json.dump(coco, f)
    print("Done.")

def main():
    # Training set
    print('Loading training set...')
    with open(src_train, "r") as f:
        train_labels = json.load(f)
    print('Converting training set...')
    bdd2coco_detection(train_labels, dst_train)

    # Validation set
    print('\nLoading validation set...')
    with open(src_val, "r") as f:
        val_labels = json.load(f)
    print('Converting validation set...')
    bdd2coco_detection(val_labels, dst_val)

if __name__ == '__main__':
    main()
