import cv2
import os
import numpy as np
import csv
import time
from datetime import timedelta

def create_summary_video(
    input_video_path, 
    target_classes, 
    confidence_threshold=0.5, 
    output_dir="output",
    progress_callback=None
):
    """
    Process a video to detect and track objects, then create a summary.
    
    Args:
        input_video_path: Path to the input video file
        target_classes: List of object classes to detect
        confidence_threshold: Minimum confidence threshold for detections
        output_dir: Directory to save output files
        progress_callback: Function to report progress (0-100)
    
    Returns:
        Dictionary with processing results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize YOLO model
    if progress_callback:
        progress_callback(1)
    
    # Load YOLOv8 model
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv8 model
    except ImportError:
        raise ImportError("Please install ultralytics: pip install ultralytics")
    
    if progress_callback:
        progress_callback(5)
    
    # Open the video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer for summary video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(output_dir, "summary.mp4")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Dictionary to store tracked objects
    tracked_objects = {}
    
    # Class name mapping (COCO dataset)
    class_names = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
        6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
        11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
        16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
        22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
        27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
        32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
        36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
        51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
        57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
        62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
        68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
        73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
        78: 'hair drier', 79: 'toothbrush'
    }
    
    # Map target classes to class IDs
    target_class_ids = []
    for class_id, class_name in class_names.items():
        if class_name in target_classes:
            target_class_ids.append(class_id)
    
    # Process video frames
    frame_count = 0
    last_progress = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate progress
        progress = int(5 + (frame_count / total_frames) * 85)  # 5-90% for frame processing
        if progress > last_progress and progress_callback:
            progress_callback(progress)
            last_progress = progress
        
        # Run YOLOv8 inference on the frame
        results = model.track(frame, persist=True, conf=confidence_threshold, classes=target_class_ids)
        
        # Get the first result (only one image was processed)
        if results and len(results) > 0:
            result = results[0]
            
            # Draw bounding boxes and labels on the frame
            annotated_frame = result.plot()
            
            # Process tracking results if available
            if hasattr(result, 'boxes') and hasattr(result.boxes, 'id') and result.boxes.id is not None:
                boxes = result.boxes.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    # Get box coordinates, confidence, class and tracking ID
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    track_id = int(box.id[0])
                    
                    # Get class name
                    class_name = class_names.get(class_id, f"class_{class_id}")
                    
                    # Skip if not in target classes
                    if class_name not in target_classes:
                        continue
                    
                    # Create a unique identifier for this object
                    object_id = f"{class_name}_{track_id}"
                    
                    # Calculate frame timestamp
                    timestamp = frame_count / fps
                    timestamp_str = str(timedelta(seconds=int(timestamp)))
                    
                    # Store object information if it's new
                    if object_id not in tracked_objects:
                        # Extract ROI (Region of Interest)
                        roi = frame[y1:y2, x1:x2].copy()
                        
                        # Save ROI as image
                        roi_filename = f"{object_id}.jpg"
                        roi_path = os.path.join(output_dir, roi_filename)
                        cv2.imwrite(roi_path, roi)
                        
                        tracked_objects[object_id] = {
                            'id': track_id,
                            'class': class_name,
                            'confidence': conf,
                            'first_appearance': timestamp_str,
                            'first_frame': frame_count,
                            'roi_path': roi_filename
                        }
            
            # Write the annotated frame to the output video
            out.write(annotated_frame)
        else:
            # If no detections, write the original frame
            out.write(frame)
        
        frame_count += 1
    
    # Release resources
    cap.release()
    out.release()
    
    if progress_callback:
        progress_callback(90)
    
    # Prepare results
    individual_objects = []
    for i, (object_id, obj_data) in enumerate(tracked_objects.items()):
        individual_objects.append({
            'serial': i + 1,
            'id': obj_data['id'],
            'class': obj_data['class'],
            'confidence': obj_data['confidence'],
            'first_appearance': obj_data['first_appearance'],
            'roi_path': obj_data['roi_path']
        })
    
    # Create summary statistics
    class_counts = {}
    for obj in individual_objects:
        class_name = obj['class']
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
    
    # Write summary statistics to CSV
    stats_path = os.path.join(output_dir, "summary_statistics.csv")
    with open(stats_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Statistic', 'Value'])
        writer.writerow(['Total Objects', len(individual_objects)])
        for class_name, count in class_counts.items():
            writer.writerow([f'Total {class_name}', count])
    
    if progress_callback:
        progress_callback(100)
    
    # Return results
    return {
        'summary_video': output_video_path,
        'total_objects': len(individual_objects),
        'class_counts': class_counts,
        'individual_objects': individual_objects
    }