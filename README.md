**Lane & Object Detection Using Computer Vision and YOLOv8**

This project integrates lane detection and object detection into a single pipeline using road camera images. Lane detection is performed with classical computer vision techniques, while object detection is handled by a custom-trained YOLOv8n model. The final output is a fully annotated driving video displaying detected lanes, objects, and live FPS.


**Project Goal**

The goal of this project is to detect:

- Lane boundaries  
- Vehicles such as cars, buses, trucks, and motorcycles  
- Pedestrians  
- Traffic lights and other road objects  

using real-world nuScenes v1.0-mini camera footage.


**Method Overview**

Lane Detection (Classical Computer Vision):

- Convert each frame to grayscale  
- Apply a Region of Interest mask  
- Use Canny edge detection  
- Extract lane lines using the Hough Transform  
- Overlay detected lanes on the original frame  


Object Detection (Deep Learning):

A YOLOv8n model was trained using auto-labeled nuScenes images that include cars, trucks, buses, motorcycles, pedestrians, and traffic lights.

Training pipeline:

1. Use YOLOv8x to auto-label images  
2. Split dataset into training and validation sets  
3. Generate dataset.yaml  
4. Train YOLOv8n for 40 epochs  
5. Save the best-performing model as best.pt  


**Dataset**

Dataset used: nuScenes v1.0-mini

Camera folders included:

- CAM_FRONT  
- CAM_FRONT_LEFT  
- CAM_FRONT_RIGHT  
- CAM_BACK  
- CAM_BACK_LEFT  
- CAM_BACK_RIGHT  

Images from all cameras were combined to create a multi-camera driving dataset.


**Final Output**

The final inference pipeline:

- Processes frames from all camera folders  
- Performs lane detection and YOLO object detection  
- Draws bounding boxes, labels, and lane overlays  
- Writes an annotated MP4 video  
- Saves sample frames into a results folder  


**Tools and Frameworks Used**

- Ultralytics YOLOv8  
- PyTorch  
- OpenCV  
- NumPy  
- tqdm  
- Python 3.10+  


**Citation**

nuScenes Dataset © Motional / nuScenes.org  
YOLOv8 Framework © Ultralytics  


**Repository Contents**

README.md  
week1log.txt  
week2log.txt  
week3log.txt  
week4log.txt  
week5log.txt  

documentation.txt  
reproducibility.txt  

results/  
code/train/generate_best_from_nuscenes_multi.py  
code/inference/part3_infer_and_video_multi.py  
requirements.txt  
