import os
import shutil
import uuid
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
from pydantic import BaseModel
import time
from pathlib import Path
import json
from datetime import datetime

# Import your video summarization function
from video_summarization_yolo import summarize_video

app = FastAPI(title="Video Surveillance System")

# Create necessary directories
os.makedirs("static", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory="results"), name="results")

# Templates
templates = Jinja2Templates(directory="templates")

# Store processing status
processing_tasks = {}

class ProcessingStatus(BaseModel):
    task_id: str
    status: str
    progress: float = 0
    result: dict = None
    error: str = None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Get list of available classes from YOLO model
    available_classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", 
                        "truck", "boat", "traffic light", "fire hydrant", "stop sign", 
                        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", 
                        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", 
                        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
                        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
                        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", 
                        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", 
                        "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", 
                        "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", 
                        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
                        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", 
                        "hair drier", "toothbrush"]
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "available_classes": available_classes
    })

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    # Generate a unique ID for this upload
    upload_id = str(uuid.uuid4())
    
    # Create a directory for this upload
    upload_dir = os.path.join("uploads", upload_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save the uploaded file
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Get video metadata (you might want to use a library like ffmpeg-python for more accurate info)
    file_size = os.path.getsize(file_path)
    file_extension = os.path.splitext(file.filename)[1]
    
    # Return the upload info
    return {
        "upload_id": upload_id,
        "filename": file.filename,
        "file_path": file_path,
        "file_size": file_size,
        "file_extension": file_extension
    }

def process_video_task(task_id: str, input_video: str, classes: List[str], confidence: float):
    try:
        # Update status to processing
        processing_tasks[task_id].status = "processing"
        
        # Create a timestamped results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join("results", f"{task_id}_{timestamp}")
        
        # Process the video
        result = summarize_video(
            input_video=input_video,
            classes=classes,
            confidence=confidence,
            output_dir=result_dir,
            save_csv=True,
            save_json=True
        )
        
        # Update the result paths to be relative to the web server
        if result["success"]:
            if "output_video" in result and result["output_video"]:
                result["output_video"] = result["output_video"].replace("\\", "/")
                # Make path relative to the web server
                if os.path.exists(result["output_video"]):
                    rel_path = os.path.relpath(result["output_video"], start="results")
                    result["output_video"] = f"/results/{rel_path}"
            
            # Make ROI paths relative
            if "roi_directory" in result:
                result["roi_directory"] = result["roi_directory"].replace("\\", "/")
                if os.path.exists(result["roi_directory"]):
                    rel_path = os.path.relpath(result["roi_directory"], start="results")
                    result["roi_directory"] = f"/results/{rel_path}"
            
            # Process unique objects to include web-accessible paths
            if "unique_objects" in result:
                for obj_key, obj_data in result["unique_objects"].items():
                    if "roi_path" in obj_data and obj_data["roi_path"]:
                        if os.path.exists(obj_data["roi_path"]):
                            rel_path = os.path.relpath(obj_data["roi_path"], start="results")
                            obj_data["roi_path"] = f"/results/{rel_path}"
        
        # Update task status
        processing_tasks[task_id].status = "completed"
        processing_tasks[task_id].progress = 100
        processing_tasks[task_id].result = result
        
    except Exception as e:
        # Update task status with error
        processing_tasks[task_id].status = "failed"
        processing_tasks[task_id].error = str(e)

@app.post("/process-video/")
async def process_video(
    background_tasks: BackgroundTasks,
    upload_id: str = Form(...),
    filename: str = Form(...),
    classes: List[str] = Form(...),
    confidence: float = Form(0.5)
):
    # Generate a task ID
    task_id = str(uuid.uuid4())
    
    # Get the full path to the uploaded video
    input_video = os.path.join("uploads", upload_id, filename)
    
    # Check if the file exists
    if not os.path.exists(input_video):
        return JSONResponse(
            status_code=404,
            content={"error": f"Uploaded file not found: {filename}"}
        )
    
    # Initialize task status
    processing_tasks[task_id] = ProcessingStatus(
        task_id=task_id,
        status="queued"
    )
    
    # Start processing in the background
    background_tasks.add_task(
        process_video_task,
        task_id,
        input_video,
        classes,
        confidence
    )
    
    return {"task_id": task_id}

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in processing_tasks:
        return JSONResponse(
            status_code=404,
            content={"error": f"Task not found: {task_id}"}
        )
    
    task = processing_tasks[task_id]
    return {
        "task_id": task.task_id,
        "status": task.status,
        "progress": task.progress,
        "result": task.result,
        "error": task.error
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)