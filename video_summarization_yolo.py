import os
from tabnanny import verbose
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import csv
import argparse
import numpy as np
import json
from datetime import timedelta
from ultralytics import YOLO
from tqdm import tqdm
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("video_summarization")

def create_summary_video(input_video_path, target_classes, confidence_threshold, output_dir, 
                        model_path=None, save_csv=True, save_json=True):
    """
    Create a summary video containing only frames with specified objects.
    
    Args:
        input_video_path (str): Path to the input video file
        target_classes (list): List of object classes to detect
        confidence_threshold (float): Confidence threshold for detections
        output_dir (str): Directory to save output files
        model_path (str, optional): Path to YOLO model. Defaults to yolo11x.pt
        
    Returns:
        dict: Dictionary containing summary information and paths to output files
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create directory for ROI images
        roi_dir = os.path.join(output_dir, "roi_images")
        os.makedirs(roi_dir, exist_ok=True)
        
        # Time model loading
        model_load_start = time.time()
        # Load the YOLOv11 model
        model_path = model_path or r"D:\code\yolov8-custom-training\weights\yolo11x.pt"
        model = YOLO(model_path)
        model_load_time = time.time() - model_load_start
        logger.info(f"Model loaded in {model_load_time:.2f} seconds")
        
        # Start timing video processing
        video_process_start = time.time()
        
        # Open the input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            error_msg = f"Could not open video file: {input_video_path}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Check if total_frames is valid
        if total_frames <= 0:
            logger.warning("Could not determine total frames. Progress tracking will be disabled.")
            use_progress_bar = False
            progress_bar = tqdm(desc="Processing video")
        else:
            use_progress_bar = True
            progress_bar = tqdm(total=total_frames, desc="Processing video")
        
        # Create output video writer
        output_video_path = os.path.join(output_dir, "summary.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Create CSV file for logging detections only if requested
        csv_file = None
        csv_writer = None
        if save_csv:
            csv_path = os.path.join(output_dir, "detections.csv")
            # Use UTF-8 encoding to handle special characters and emojis
            csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
            csv_writer = csv.writer(csv_file)
            # Updated CSV header to include ROI path
            csv_writer.writerow(["Object ID", "Class", "First Seen Frame", "First Seen Time", "ROI Path"])
        
        # Initialize JSON data only if requested
        json_path = None
        if save_json:
            json_path = os.path.join(output_dir, "detections.json")
            json_data = {
                "video_info": {
                    "path": input_video_path,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "total_frames": total_frames
                },
                "unique_objects": {}
            }
        
        # Dictionary to track unique objects by tracking ID
        unique_objects = {}
        # Dictionary to count unique objects by class
        class_counts = {cls: 0 for cls in target_classes}
        # Total unique objects counter
        total_unique_objects = 0
        
        # Dictionary to track last seen frame for each object
        last_seen_frames = {}
        
        # Process the video frame by frame
        frame_idx = 0
        saved_frames = 0
        
        # List to store unique objects for CSV writing at the end
        unique_objects_for_csv = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate timestamp
            timestamp = timedelta(seconds=frame_idx/fps)
            timestamp_str = str(timestamp).split('.')[0]  # Format as HH:MM:SS
            
            # Process frame with YOLO tracking
            results = model.track(frame, persist=True, conf=confidence_threshold, verbose=False)
            
            # Check if any target classes are detected
            detected_classes = []
            tracking_ids = []
            detections = []
            
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                
                for box in boxes:
                    cls_id = int(box.cls.item())
                    cls_name = model.names[cls_id]
                    confidence = float(box.conf.item())
                    
                    if cls_name in target_classes:
                        detected_classes.append(cls_name)
                        
                        # Get tracking ID if available
                        track_id = None
                        if hasattr(box, 'id') and box.id is not None:
                            track_id = int(box.id.item())
                            tracking_ids.append(track_id)
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
                            
                            # Create object key for tracking
                            object_key = f"{cls_name}_{track_id}"
                            
                            # Update last seen frame for duration tracking
                            last_seen_frames[object_key] = frame_idx
                            
                            # Check if this is a new unique object
                            if object_key not in unique_objects:
                                # Increment class counter
                                class_counts[cls_name] += 1
                                total_unique_objects += 1
                                
                                # Extract ROI
                                roi = frame[y1:y2, x1:x2]
                                if roi.size > 0:  # Ensure ROI is not empty
                                    # Save ROI image
                                    roi_filename = f"{cls_name}_{track_id}_{timestamp_str.replace(':', '_')}.jpg"
                                    roi_path = os.path.join(roi_dir, roi_filename)
                                    cv2.imwrite(roi_path, roi)
                                    
                                    # Store unique object info
                                    unique_objects[object_key] = {
                                        "class": cls_name,
                                        "tracking_id": track_id,
                                        "first_seen_frame": frame_idx,
                                        "first_seen_time": timestamp_str,
                                        "last_seen_frame": frame_idx,  # Initialize last seen frame
                                        "last_seen_time": timestamp_str,  # Initialize last seen time
                                        "duration_frames": 0,  # Initialize duration in frames
                                        "duration_time": "00:00:00",  # Initialize duration as time
                                        "confidence": confidence,
                                        "roi_path": roi_path
                                    }
                                    
                                    # Add to list for CSV writing
                                    if save_csv:
                                        unique_objects_for_csv.append([
                                            object_key,
                                            cls_name,
                                            frame_idx,
                                            timestamp_str,
                                            roi_path
                                        ])
                                    
                                    # Display on console
                                    console_msg = f"{cls_name}, {confidence*100:.1f}%, {timestamp_str}, {roi_path}"
                                    print(console_msg)
                                    logger.info(f"New unique object detected: {console_msg}")

            # If target classes detected, add to summary video
            if detected_classes:
                # Add frame to summary video
                out.write(frame)
                saved_frames += 1
                
                # Display real-time progress
                if use_progress_bar and total_frames > 0:
                    logger.debug(f"Frame: {frame_idx}/{total_frames} | "
                          f"Detected: {', '.join(set(detected_classes))} | "
                          f"Progress: {frame_idx/total_frames*100:.1f}%")
                else:
                    logger.debug(f"Frame: {frame_idx} | "
                          f"Detected: {', '.join(set(detected_classes))}")
            
            frame_idx += 1
            if use_progress_bar:
                progress_bar.update(1)
        
        # After video processing, do a final update of durations for all objects
        for object_key, obj_info in unique_objects.items():
            # Calculate final duration in frames
            duration_frames = last_seen_frames.get(object_key, obj_info["first_seen_frame"]) - obj_info["first_seen_frame"]
            obj_info["duration_frames"] = duration_frames
            
            # Calculate final duration as time
            duration_seconds = duration_frames / fps
            duration_time = str(timedelta(seconds=duration_seconds)).split('.')[0]
            obj_info["duration_time"] = duration_time
            
            # Calculate last seen time if not already updated
            if obj_info["last_seen_frame"] < last_seen_frames.get(object_key, obj_info["first_seen_frame"]):
                last_frame = last_seen_frames.get(object_key, obj_info["first_seen_frame"])
                last_time = str(timedelta(seconds=last_frame/fps)).split('.')[0]
                obj_info["last_seen_frame"] = last_frame
                obj_info["last_seen_time"] = last_time
        
        # Calculate video processing time
        video_process_time = time.time() - video_process_start
        
        # Clean up
        cap.release()
        out.release()
        
        # Calculate input video duration
        input_duration_seconds = total_frames / fps if fps > 0 else 0
        input_duration = str(timedelta(seconds=input_duration_seconds)).split('.')[0]
        
        # Calculate output video duration
        output_duration_seconds = saved_frames / fps if fps > 0 else 0
        output_duration = str(timedelta(seconds=output_duration_seconds)).split('.')[0]
        
        # Check if any objects were detected
        if total_unique_objects == 0:
            # If no objects detected, delete the empty output video
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
                logger.warning("No objects found in the video. Empty output video deleted.")
            
            # Prepare summary data with no objects found message
            summary_data = {
                "success": True,
                "input_video": input_video_path,
                "input_duration": input_duration,
                "output_duration": "00:00:00",  # No output video
                "total_frames": total_frames,
                "target_classes": target_classes,
                "confidence_threshold": confidence_threshold,
                "model_load_time": model_load_time,
                "video_process_time": video_process_time,
                "total_process_time": model_load_time + video_process_time,
                "unique_objects": {},
                "class_counts": class_counts,
                "total_unique_objects": 0,
                "roi_directory": roi_dir,
                "message": "No objects found in the video matching the specified classes and confidence threshold."
            }
            
            logger.warning("No objects found in the video matching the specified classes and confidence threshold.")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Model loading time: {model_load_time:.2f} seconds")
            logger.info(f"Video processing time: {video_process_time:.2f} seconds")
            logger.info(f"Total processing time: {model_load_time + video_process_time:.2f} seconds")
            
            return summary_data
        
        # If objects were detected, continue with normal processing
        # Write unique objects to CSV file
        if save_csv and csv_file:
            for obj_data in unique_objects_for_csv:
                csv_writer.writerow(obj_data)
            csv_file.close()
        
        # Add unique objects data to JSON and write to file if enabled
        if save_json:
            json_data["unique_objects"] = unique_objects
            json_data["class_counts"] = class_counts
            json_data["total_unique_objects"] = total_unique_objects
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
        
        # Prepare summary data
        summary_data = {
            "success": True,
            "input_video": input_video_path,
            "output_video": output_video_path,
            "input_duration": input_duration,
            "output_duration": output_duration,
            "total_frames": total_frames,
            "saved_frames": saved_frames,
            "target_classes": target_classes,
            "confidence_threshold": confidence_threshold,
            "model_load_time": model_load_time,
            "video_process_time": video_process_time,
            "total_process_time": model_load_time + video_process_time,
            "unique_objects": unique_objects,
            "class_counts": class_counts,
            "total_unique_objects": total_unique_objects,
            "roi_directory": roi_dir
        }
        
        # Only add paths to files that were actually created
        if save_csv and csv_file:
            summary_data["csv_path"] = csv_path
        
        if save_json and json_path:
            summary_data["json_path"] = json_path
            
        logger.info(f"Summary video created with {saved_frames} frames out of {total_frames} total frames.")
        logger.info(f"Output saved to: {output_dir}")
        
        # Check if any objects were detected
        if total_unique_objects == 0:
            logger.warning("No objects found in the video matching the specified classes and confidence threshold.")
            summary_data["message"] = "No objects found in the video matching the specified classes and confidence threshold."
        else:
            logger.info(f"Detected {total_unique_objects} unique objects across all classes.")
            for cls, count in class_counts.items():
                if count > 0:
                    logger.info(f"  - {cls}: {count} unique objects")
                    
                    # Log duration information for each object
                    for object_key, obj_info in unique_objects.items():
                        if obj_info["class"] == cls:
                            logger.info(f"    * {object_key}: visible for {obj_info['duration_time']} "
                                      f"(from {obj_info['first_seen_time']} to {obj_info['last_seen_time']})")
                        
        logger.info(f"ROI images saved to: {roi_dir}")
        logger.info(f"Model loading time: {model_load_time:.2f} seconds")
        logger.info(f"Video processing time: {video_process_time:.2f} seconds")
        logger.info(f"Total processing time: {model_load_time + video_process_time:.2f} seconds")
        
        return summary_data
        
    except Exception as e:
        logger.error(f"Error in video summarization: {str(e)}", exc_info=True)
        # Return error dictionary instead of None
        return {
            "success": False,
            "error": str(e)
        }


# API function for backend integration
def summarize_video(input_video, classes=None, confidence=0.8, output_dir=None, model_path=None, save_csv=True, save_json=True):
    """
    Process a video to detect and track objects, creating a summary video and detailed analytics.
    
    Args:
        input_video (str): Path to input video file
        classes (list or str, optional): Classes to detect. Can be:
            - None: Uses default classes ["person", "car", "truck", "baggage", "bus", "mobile"]
            - list: List of class names to detect (e.g., ["person", "car"])
            - str: Comma-separated class names (e.g., "person,car,truck")
        confidence (float, optional): Detection confidence threshold (0-1). Default is 0.8.
        output_dir (str, optional): Directory to save output files. If None, creates a timestamped directory.
        model_path (str, optional): Path to YOLO model file. If None, uses default model.
        save_csv (bool, optional): Whether to save detection data as CSV. Default is True.
        save_json (bool, optional): Whether to save detection data as JSON. Default is True.
        
    Returns:
        dict: Results dictionary with detection data and output file paths
    """
    # Handle string input for target_classes
    if classes is None:
        target_classes = ["person", "car", "truck", "baggage", "bus", "mobile"]
    elif isinstance(classes, str):
        target_classes = [cls.strip() for cls in classes.split(',')]
    else:
        target_classes = classes
    
    # Create a unique output directory based on input video name and timestamp if not provided
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_name = os.path.splitext(os.path.basename(input_video))[0]
        output_dir = os.path.join(os.getcwd(), f"{video_name}_{timestamp}")
    
    # Validate inputs
    if not os.path.exists(input_video):
        return {"success": False, "error": f"Input video not found: {input_video}"}
    
    # Validate that the file is a valid video
    try:
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            return {"success": False, "error": f"Invalid video file or unsupported format: {input_video}"}
        
        # Check if we can read at least one frame
        ret, _ = cap.read()
        if not ret:
            cap.release()
            return {"success": False, "error": f"Cannot read frames from video file: {input_video}"}
        
        # Reset the video capture
        cap.release()
    except Exception as e:
        return {"success": False, "error": f"Error validating video file: {input_video}. Error: {str(e)}"}
    
    # Validate model file exists if provided
    if model_path and not os.path.exists(model_path):
        return {"success": False, "error": f"Model file not found: {model_path}"}
    
    if not isinstance(confidence, (int, float)) or confidence <= 0 or confidence > 1:
        return {"success": False, "error": f"Invalid confidence value: {confidence}. Must be between 0 and 1."}
    
    # Process the video directly
    result = create_summary_video(
        input_video,
        target_classes,
        confidence,
        output_dir,
        model_path,
        save_csv,
        save_json
    )
    
    # If successful, read the JSON data to include in the response
    if result["success"]:
        try:
            json_data = {}
            # Only try to read JSON file if it exists and save_json was enabled
            if save_json and result["json_path"] and os.path.exists(result["json_path"]):
                with open(result["json_path"], 'r') as f:
                    json_data = json.load(f)
            
            # Create a summary of unique objects by class
            class_summary = {}
            total_unique = result["total_unique_objects"]
            
            # Format the class counts in a more readable way
            for cls, count in result["class_counts"].items():
                if count > 0:
                    class_summary[cls] = count
                
            # Create a simplified response with the most important information
            return {
                "success": True,
                "output_video": result.get("output_video", ""),
                "json_path": result.get("json_path", ""),
                "csv_path": result.get("csv_path", ""),
                "input_duration": result.get("input_duration", "00:00:00"),
                "output_duration": result.get("output_duration", "00:00:00"),
                "unique_objects": result["unique_objects"],
                "class_counts": result["class_counts"],
                "total_unique_objects": total_unique,
                "class_summary": class_summary,  # Add the formatted class summary
                "roi_directory": result["roi_directory"],
                "json_data": json_data,
                "stats": {
                    "total_frames": result["total_frames"],
                    "saved_frames": result.get("saved_frames", 0),
                    "processing_time": result["total_process_time"],
                    "input_duration": result.get("input_duration", "00:00:00"),
                    "output_duration": result.get("output_duration", "00:00:00")
                },
                "message": result.get("message", "")
            }
        except Exception as e:
            return {
                "success": False, 
                "error": f"Video processing succeeded but failed to read results: {str(e)}"
            }
    else:
        return result


def main():
    """Main function to run the video summarization."""
    result = summarize_video(
        # input_video=r"C:\Users\ASDF\Documents\20250415172711555.mp4",
        # input_video=r"C:\Users\ASDF\Documents\20250415154445022.mp4",
        # input_video=r"C:\Users\ASDF\Documents\20250415172502980.mp4",
        input_video=r"C:\Users\ASDF\Downloads\World's SLIMMEST Phone EVER ! 😳 #shorts.mp4",
        # input_video=r"C:\Users\ASDF\Downloads\videoplayback.mp4",
        classes=["person", "car", "truck", "bus","cell phone"],
        # classes=["person"],
        confidence=0.5,
        save_csv=True,
        save_json=False,
        model_path=r"D:\code\yolov8-custom-training\weights\yolo11x.pt"
    )
    
    # Output results
    if result["success"]:
        # Print input video duration in all cases
        logger.info(f"Input video duration: {result['input_duration']}")
        
        if "message" in result and result["message"]:
            # No objects found case
            logger.warning(result["message"])
            logger.info("No output video created as no objects were detected.")
        else:
            # Objects found case
            logger.info(f"Total Unique Objects: {result['total_unique_objects']}")
            
            # Print counts for each class
            for cls, count in result["class_counts"].items():
                if count > 0:
                    logger.info(f"Total Unique {cls}: {count}")
            
            # Print output video duration only if there is an output video
            logger.info(f"Output video duration: {result['output_duration']}")
            logger.info(f"Summary video saved to: {result['output_video']}")
            
            if "json_path" in result and result["json_path"]:
                logger.info(f"Detection data saved to: {result['json_path']}")
            elif "csv_path" in result and result["csv_path"]:
                logger.info(f"Detection data saved to: {result['csv_path']}")
    else:
        logger.error(f"Failed to create summary: {result['error']}")
    
    return result

if __name__ == "__main__":
    main()