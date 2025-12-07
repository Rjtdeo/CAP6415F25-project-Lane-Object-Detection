"""
generate_best_from_nuscenes_multi.py
=============================================================
This script:

1. Loads ALL SIX nuScenes camera folders:
       CAM_FRONT
       CAM_FRONT_LEFT
       CAM_FRONT_RIGHT
       CAM_BACK
       CAM_BACK_LEFT
       CAM_BACK_RIGHT

2. Creates a full YOLO dataset:
       images/train, images/val
       labels/train, labels/val

3. Auto-labels all images using YOLOv8x

4. Creates dataset.yaml

5. Trains YOLOv8n for 40 epochs

6. Produces best.pt:
   C:/Users/Rajat/Downloads/runs/vehicle_detector_multi/weights/best.pt
=============================================================
"""

import os
import random
import shutil
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO


# CAMERA FOLDERS (FULL nuScenes multi-camera)
# ============================================================

CAMERA_FOLDERS = [
    r"C:\Users\Rajat\Downloads\archive\samples\CAM_FRONT",
    r"C:\Users\Rajat\Downloads\archive\samples\CAM_FRONT_LEFT",
    r"C:\Users\Rajat\Downloads\archive\samples\CAM_FRONT_RIGHT",
    r"C:\Users\Rajat\Downloads\archive\samples\CAM_BACK",
    r"C:\Users\Rajat\Downloads\archive\samples\CAM_BACK_LEFT",
    r"C:\Users\Rajat\Downloads\archive\samples\CAM_BACK_RIGHT",
]


# DATASET OUTPUT
# ============================================================

BASE = Path(r"C:\Users\Rajat\Downloads\vehicle_dataset_multi")
IMG_TRAIN = BASE / "images" / "train"
IMG_VAL   = BASE / "images" / "val"
LAB_TRAIN = BASE / "labels" / "train"
LAB_VAL   = BASE / "labels" / "val"
DATASET_YAML = BASE / "dataset.yaml"

for p in [IMG_TRAIN, IMG_VAL, LAB_TRAIN, LAB_VAL]:
    p.mkdir(parents=True, exist_ok=True)


# STEP 1 — LOAD ALL IMAGES FROM ALL CAMERAS
# ============================================================
print("\n[STEP 1] Gathering ALL camera images...")

all_images = []
for folder in CAMERA_FOLDERS:
    p = Path(folder)
    imgs = sorted([x for x in p.iterdir() if x.suffix.lower() in (".jpg",".jpeg",".png")])
    all_images.extend(imgs)

print(f"[ OK ] Total images found: {len(all_images)}")


# STEP 2 — TRAIN/VAL SPLIT
# ============================================================
print("\n[STEP 2] Splitting into train/val 80/20...")

random.shuffle(all_images)
split_idx = int(len(all_images) * 0.8)

train_imgs = all_images[:split_idx]
val_imgs   = all_images[split_idx:]

for img in train_imgs:
    shutil.copy(img, IMG_TRAIN / img.name)
for img in val_imgs:
    shutil.copy(img, IMG_VAL / img.name)

print(f"[ OK ] Train images: {len(train_imgs)}")
print(f"[ OK ] Val images  : {len(val_imgs)}")



# STEP 3 — AUTO-LABEL WITH YOLOv8x
# ============================================================
print("\n[STEP 3] Auto-labeling with YOLOv8x (COCO)...")

autolabel_model = YOLO("yolov8x.pt")

def autolabel(images_list, label_folder):
    for img_path in tqdm(images_list):
        results = autolabel_model(img_path, verbose=False)[0]
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        label_path = label_folder / f"{img_path.stem}.txt"
        with open(label_path, "w") as f:
            for box in results.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0]

                xc = float((x1 + x2) / (2 * w))
                yc = float((y1 + y2) / (2 * h))
                bw = float((x2 - x1) / w)
                bh = float((y2 - y1) / h)

                f.write(f"{cls} {xc} {yc} {bw} {bh}\n")


print("Auto-labeling TRAIN images...")
autolabel(IMG_TRAIN.iterdir(), LAB_TRAIN)

print("Auto-labeling VAL images...")
autolabel(IMG_VAL.iterdir(), LAB_VAL)

print("[ OK ] Auto-labeling complete.")


# STEP 4 — CREATE dataset.yaml
# ============================================================
print("\n[STEP 4] Creating dataset.yaml...")

yaml_content = f"""
path: {BASE}

train: images/train
val: images/val

names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: bus
  5: truck
  6: traffic_light
  7: stop_sign
  8: fire_hydrant
"""

with open(DATASET_YAML, "w") as f:
    f.write(yaml_content)

print("[ OK ] dataset.yaml created.")


# STEP 5 — TRAIN YOLO MODEL
# ============================================================
print("\n[STEP 5] Training YOLOv8n on ALL CAMERA DATA...")

train_model = YOLO("yolov8n.pt")

train_model.train(
    data=str(DATASET_YAML),
    imgsz=640,
    epochs=40,
    batch=8,
    workers=2,
    name="vehicle_detector_multi",
    project="runs",
    device=0,   # GPU if available
)

print("\n TRAINING COMPLETE!")
print("BEST model is saved at:")
print(r"C:\Users\Rajat\Downloads\runs\vehicle_detector_multi\weights\best.pt")
