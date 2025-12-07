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

What this script does in simple words:
1) Reads all images from the 6 camera folders.
2) Runs basic lane detection using OpenCV.
3) Runs YOLO object detection using your trained model.
4) Overlays lanes + bounding boxes on each image.
5) Saves one combined output video.
6) Saves a few sample annotated frames for reporting.

Note:
- This is a simple classical lane method (Canny + Hough).
- It is good for demo/learning but not perfect in real-world cases.
----------------------------------------------------------
"""

import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ----------------------------- INPUT FOLDERS -----------------------------
# List of all nuScenes camera folders you want to use.
# The script will read ALL .jpg files from these folders.
CAM_FOLDERS = [
    r"C:\Users\Rajat\Downloads\archive\samples\CAM_FRONT",
    r"C:\Users\Rajat\Downloads\archive\samples\CAM_FRONT_LEFT",
    r"C:\Users\Rajat\Downloads\archive\samples\CAM_FRONT_RIGHT",
    r"C:\Users\Rajat\Downloads\archive\samples\CAM_BACK",
    r"C:\Users\Rajat\Downloads\archive\samples\CAM_BACK_LEFT",
    r"C:\Users\Rajat\Downloads\archive\samples\CAM_BACK_RIGHT"
]

# Path to your trained YOLO weights.
WEIGHTS_PATH = r"C:\Users\Rajat\runs\vehicle_detector_multi\weights\best.pt"

# Output video path.
OUT_VIDEO = r"C:\Users\Rajat\Downloads\output_multi_camera_yolo.mp4"

# Folder for saving a few annotated sample images.
RESULTS_DIR = r"C:\Users\Rajat\Downloads\results_multi_1"


# ----------------------------- SETTINGS -----------------------------
# Output video FPS.
FPS = 20

# Confidence threshold for YOLO detections.
CONF_THRES = 0.5

# Save only first N processed frames as images.
MAX_SAMPLE_FRAMES = 10


def select_device():
    """
    Select the best available device for running inference.
    Priority order:
    1) CUDA GPU (NVIDIA)
    2) MPS (Apple Silicon)
    3) CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def apply_roi(gray_img):
    """
    Apply a simple Region of Interest (ROI) mask.

    We assume lane lines mostly appear in the lower part of the image,
    so we keep only the bottom area to reduce noise.

    Input:
        gray_img: grayscale image

    Output:
        masked grayscale image limited to the ROI
    """
    h, w = gray_img.shape[:2]

    # Create a black mask same size as input image.
    mask = np.zeros_like(gray_img)

    # Polygon points defining the ROI (lower 40% of the image).
    pts = np.array([[(0, h), (0, int(h * 0.6)), (w, int(h * 0.6)), (w, h)]], np.int32)

    # Fill ROI area with white (255) to keep it.
    cv2.fillPoly(mask, pts, 255)

    # Apply mask to the grayscale image.
    return cv2.bitwise_and(gray_img, mask)


def detect_lanes(frame):
    """
    Basic lane detection using classical OpenCV steps:
    1) Convert to grayscale
    2) Apply ROI mask
    3) Canny edge detection
    4) HoughLinesP to get line segments
    5) Draw lane-like segments on an overlay

    Input:
        frame: original BGR image

    Output:
        overlay: image containing only lane lines (for blending)
    """
    # Convert frame to grayscale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Keep only the road-likely region.
    roi = apply_roi(gray)

    # Detect edges.
    edges = cv2.Canny(roi, 50, 150)

    # Detect line segments using probabilistic Hough transform.
    lines = cv2.HoughLinesP(edges, 2, np.pi / 180, 100, minLineLength=40, maxLineGap=25)

    # Create an empty overlay to draw lines.
    overlay = np.zeros_like(frame)

    # Draw detected line segments if any exist.
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 4)

    return overlay


def yolo_detect(model, frame, conf):
    """
    Run YOLO inference and draw bounding boxes + labels.

    Input:
        model: loaded YOLO model
        frame: image (BGR)
        conf: confidence threshold

    Output:
        frame with bounding boxes and labels drawn
    """
    # Run YOLO on the current frame.
    results = model(frame, verbose=False)[0]

    # Loop over each predicted bounding box.
    for box in results.boxes:
        # Skip detections below confidence threshold.
        if float(box.conf[0]) < conf:
            continue

        # Extract bounding box coordinates.
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Extract class id and convert to readable label.
        cls = int(box.cls[0])
        label = model.names.get(cls, str(cls))

        # Draw rectangle around object.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put label and confidence above the box.
        cv2.putText(frame, f"{label} {box.conf[0]:.2f}",
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame


def main():
    """
    Main pipeline:
    1) Collect all images from all 6 cameras
    2) Load YOLO model
    3) Initialize video writer using first image size
    4) For each image:
        - Detect lanes
        - Overlay lanes on frame
        - Run YOLO detection
        - Write to video
        - Save a few sample frames
    5) Print summary info
    """
    all_imgs = []

    # Collect all jpg images from each camera folder.
    for folder in CAM_FOLDERS:
        imgs = sorted(Path(folder).glob("*.jpg"))
        all_imgs.extend(imgs)

    # Stop early if no images are found.
    if not all_imgs:
        raise RuntimeError("No images found!")

    # Create results folder if it doesn't exist.
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    # Select device (GPU/MPS/CPU).
    device = select_device()

    # Load YOLO model weights and move to selected device.
    model = YOLO(WEIGHTS_PATH).to(device)

    # Read the first image to get output frame size.
    first = cv2.imread(str(all_imgs[0]))
    h, w = first.shape[:2]

    # Initialize video writer with chosen FPS and frame size.
    out = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))

    # Count how many sample images we saved.
    sample_count = 0

    # Track total runtime.
    start = time.time()

    # Process all images one by one.
    for idx, path in enumerate(all_imgs):
        frame = cv2.imread(str(path))
        t0 = time.time()

        # 1) Detect lanes.
        lanes = detect_lanes(frame)

        # 2) Blend lane overlay with the original frame.
        vis = cv2.addWeighted(frame, 0.8, lanes, 1.0, 1.0)

        # 3) Run YOLO detection and draw boxes.
        vis = yolo_detect(model, vis, CONF_THRES)

        # 4) Calculate instantaneous FPS for display.
        fps = 1.0 / (time.time() - t0 + 1e-8)

        # 5) Write FPS text on the frame.
        cv2.putText(vis, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 6) Add the processed frame to the output video.
        out.write(vis)

        # 7) Save a few sample annotated images for results.
        if sample_count < MAX_SAMPLE_FRAMES:
            cv2.imwrite(str(Path(RESULTS_DIR) / f"sample_{sample_count}.jpg"), vis)
            sample_count += 1

        # Print progress every 50 images to track execution.
        if (idx + 1) % 50 == 0:
            print(f"[INFO] Processed {idx + 1}/{len(all_imgs)}")

    # Release video writer properly.
    out.release()

    # Print final summary.
    print(f"\n[INFO] Saved video at: {OUT_VIDEO}")
    print(f"[INFO] Total frames: {len(all_imgs)}")
    print(f"[INFO] Time: {round(time.time() - start, 2)} sec")


# Standard Python entry point.
if __name__ == "__main__":
    main()
