"""
part3_infer_and_video_multi.py
----------------------------------------------------------
Final full pipeline that uses ALL nuScenes camera folders:

    CAM_FRONT
    CAM_FRONT_LEFT
    CAM_FRONT_RIGHT
    CAM_BACK
    CAM_BACK_LEFT
    CAM_BACK_RIGHT

This produces a long video (depends on how many images exist).

----------------------------------------------------------
"""

import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

CAM_FOLDERS = [
    r"C:\Users\Rajat\Downloads\archive\samples\CAM_FRONT",
    r"C:\Users\Rajat\Downloads\archive\samples\CAM_FRONT_LEFT",
    r"C:\Users\Rajat\Downloads\archive\samples\CAM_FRONT_RIGHT",
    r"C:\Users\Rajat\Downloads\archive\samples\CAM_BACK",
    r"C:\Users\Rajat\Downloads\archive\samples\CAM_BACK_LEFT",
    r"C:\Users\Rajat\Downloads\archive\samples\CAM_BACK_RIGHT"
]

WEIGHTS_PATH = r"C:\Users\Rajat\runs\vehicle_detector_multi\weights\best.pt"
OUT_VIDEO = r"C:\Users\Rajat\Downloads\output_multi_camera_yolo.mp4"
RESULTS_DIR = r"C:\Users\Rajat\Downloads\results_multi_1"

FPS = 20
CONF_THRES = 0.5
MAX_SAMPLE_FRAMES = 10

def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def apply_roi(gray_img):
    h, w = gray_img.shape[:2]
    mask = np.zeros_like(gray_img)
    pts = np.array([[(0, h), (0, int(h*0.6)), (w, int(h*0.6)), (w, h)]], np.int32)
    cv2.fillPoly(mask, pts, 255)
    return cv2.bitwise_and(gray_img, mask)

def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = apply_roi(gray)
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 2, np.pi/180, 100, minLineLength=40, maxLineGap=25)

    overlay = np.zeros_like(frame)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 4)

    return overlay

def yolo_detect(model, frame, conf):
    results = model(frame, verbose=False)[0]
    for box in results.boxes:
        if float(box.conf[0]) < conf:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names.get(cls, str(cls))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{label} {box.conf[0]:.2f}",
                    (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return frame

def main():
    all_imgs = []
    for folder in CAM_FOLDERS:
        imgs = sorted(Path(folder).glob("*.jpg"))
        all_imgs.extend(imgs)

    if not all_imgs:
        raise RuntimeError("No images found!")

    Path(RESULTS_DIR).mkdir(exist_ok=True)
    device = select_device()
    model = YOLO(WEIGHTS_PATH).to(device)

    first = cv2.imread(str(all_imgs[0]))
    h, w = first.shape[:2]
    out = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))

    sample_count = 0
    start = time.time()

    for idx, path in enumerate(all_imgs):
        frame = cv2.imread(str(path))
        t0 = time.time()

        lanes = detect_lanes(frame)
        vis = cv2.addWeighted(frame, 0.8, lanes, 1.0, 1.0)
        vis = yolo_detect(model, vis, CONF_THRES)

        fps = 1.0 / (time.time() - t0 + 1e-8)
        cv2.putText(vis, f"FPS: {fps:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        out.write(vis)

        if sample_count < MAX_SAMPLE_FRAMES:
            cv2.imwrite(str(Path(RESULTS_DIR)/f"sample_{sample_count}.jpg"), vis)
            sample_count += 1

        if (idx+1) % 50 == 0:
            print(f"[INFO] Processed {idx+1}/{len(all_imgs)}")

    out.release()
    print(f"\n[INFO] Saved video at: {OUT_VIDEO}")
    print(f"[INFO] Total frames: {len(all_imgs)}")
    print(f"[INFO] Time: {round(time.time()-start,2)} sec")

if __name__ == "__main__":
    main()
