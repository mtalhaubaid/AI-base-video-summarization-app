document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const videoUploadInput = document.getElementById('video-upload');
    const videoPreviewContainer = document.getElementById('video-preview-container');
    const videoPreview = document.getElementById('video-preview');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const fileType = document.getElementById('file-type');
    const fileDuration = document.getElementById('file-duration');
    const confidenceThreshold = document.getElementById('confidence-threshold');
    const confidenceValue = document.getElementById('confidence-value');
    const processBtn = document.getElementById('process-btn');
    const objectClasses = document.getElementById('object-classes');
    const processingStatus = document.getElementById('processing-status');
    const progressBar = document.getElementById('progress-bar');
    const statusMessage = document.getElementById('status-message');
    const resultsSection = document.getElementById('results-section');
    
    // Variables to store upload information
    let uploadInfo = null;
    let taskId = null;
    let statusCheckInterval = null;
    
    // Update confidence value display
    confidenceThreshold.addEventListener('input', function() {
        confidenceValue.textContent = this.value;
    });
    
    // Handle video upload
    videoUploadInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        // Display file information
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        fileType.textContent = file.type;
        
        // Create object URL for video preview
        const videoURL = URL.createObjectURL(file);
        videoPreview.src = videoURL;
        
        // Show video preview
        videoPreviewContainer.classList.remove('d-none');
        
        // Get video duration when metadata is loaded
        videoPreview.onloadedmetadata = function() {
            fileDuration.textContent = formatDuration(videoPreview.duration);
        };
        
        // Enable process button
        processBtn.disabled = false;
        
        // Upload the video to the server
        uploadVideo(file);
    });
    
    // Handle process button click
    processBtn.addEventListener('click', function() {
        if (!uploadInfo) {
            alert('Please upload a video first.');
            return;
        }
        
        // Get selected classes
        const selectedClasses = Array.from(objectClasses.selectedOptions).map(option => option.value);
        if (selectedClasses.length === 0) {
            alert('Please select at least one object class to detect.');
            return;
        }
        
        // Get confidence threshold
        const confidence = parseFloat(confidenceThreshold.value);
        
        // Disable process button
        processBtn.disabled = true;
        
        // Show processing status
        processingStatus.classList.remove('d-none');
        progressBar.style.width = '0%';
        statusMessage.textContent = 'Initializing processing...';
        
        // Start processing
        processVideo(uploadInfo.upload_id, uploadInfo.filename, selectedClasses, confidence);
    });
    
    // Function to upload video
    async function uploadVideo(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            statusMessage.textContent = 'Uploading video...';
            
            const response = await fetch('/upload-video/', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Failed to upload video');
            }
            
            uploadInfo = await response.json();
            console.log('Upload successful:', uploadInfo);
            
            statusMessage.textContent = 'Video uploaded successfully. Ready to process.';
        } catch (error) {
            console.error('Error uploading video:', error);
            alert('Error uploading video: ' + error.message);
        }
    }
    
    // Function to process video
    async function processVideo(uploadId, filename, classes, confidence) {
        try {
            const formData = new FormData();
            formData.append('upload_id', uploadId);
            formData.append('filename', filename);
            classes.forEach(cls => formData.append('classes', cls));
            formData.append('confidence', confidence);
            
            const response = await fetch('/process-video/', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Failed to start processing');
            }
            
            const data = await response.json();
            taskId = data.task_id;
            console.log('Processing started with task ID:', taskId);
            
            // Start checking status
            statusCheckInterval = setInterval(checkProcessingStatus, 2000);
        } catch (error) {
            console.error('Error processing video:', error);
            alert('Error processing video: ' + error.message);
            processBtn.disabled = false;
        }
    }
    
    // Function to check processing status
    async function checkProcessingStatus() {
        if (!taskId) return;
        
        try {
            const response = await fetch(`/task-status/${taskId}`);
            
            if (!response.ok) {
                throw new Error('Failed to get task status');
            }
            
            const data = await response.json();
            console.log('Task status:', data);
            
            // Update progress bar
            progressBar.style.width = `${data.progress}%`;
            
            // Update status message
            switch (data.status) {
                case 'queued':
                    statusMessage.textContent = 'Waiting in queue...';
                    break;
                case 'processing':
                    statusMessage.textContent = 'Processing video...';
                    break;
                case 'completed':
                    statusMessage.textContent = 'Processing completed successfully!';
                    clearInterval(statusCheckInterval);
                    displayResults(data.result);
                    break;
                case 'failed':
                    statusMessage.textContent = `Processing failed: ${data.error}`;
                    clearInterval(statusCheckInterval);
                    processBtn.disabled = false;
                    break;
            }
        } catch (error) {
            console.error('Error checking task status:', error);
            statusMessage.textContent = 'Error checking task status: ' + error.message;
            clearInterval(statusCheckInterval);
            processBtn.disabled = false;
        }
    }
    
    // Function to display results
    function displayResults(result) {
        if (!result || !result.success) {
            alert('Processing failed: ' + (result?.error || 'Unknown error'));
            processBtn.disabled = false;
            return;
        }
        
        // Show results section
        resultsSection.classList.remove('d-none');
        
        // Set input video
        document.getElementById('input-video-player').src = videoPreview.src;
        
        // Set output video if available
        if (result.output_video) {
            document.getElementById('output-video-player').src = result.output_video;
        } else {
            document.getElementById('output-video-player').parentElement.innerHTML = 
                '<div class="alert alert-warning">No output video generated. No objects detected.</div>';
        }
        
        // Set summary information
        document.getElementById('input-duration').textContent = result.input_duration || '00:00:00';
        document.getElementById('output-duration').textContent = result.output_duration || '00:00:00';
        document.getElementById('total-objects').textContent = result.total_unique_objects || 0;
        
        // Set class counts
        const classCountsElement = document.getElementById('class-counts');
        classCountsElement.innerHTML = '';
        
        if (result.class_counts) {
            Object.entries(result.class_counts).forEach(([className, count]) => {
                if (count > 0) {
                    const item = document.createElement('li');
                    item.className = 'list-group-item d-flex justify-content-between';
                    item.innerHTML = `
                        <span>Total Unique ${className}:</span>
                        <span class="badge bg-info">${count}</span>
                    `;
                    classCountsElement.appendChild(item);
                }
            });
        }
        
        // Set detections table
        const detectionsTable = document.getElementById('detections-table');
        detectionsTable.innerHTML = '';
        
        if (result.unique_objects && Object.keys(result.unique_objects).length > 0) {
            let index = 1;
            
            // Convert object to array and sort by first_seen_frame
            const objectsArray = Object.entries(result.unique_objects).map(([key, value]) => ({
                key,
                ...value
            }));
            
            objectsArray.sort((a, b) => a.first_seen_frame - b.first_seen_frame);
            
            objectsArray.forEach(obj => {
                const row = document.createElement('tr');
                
                // Create ROI image element
                let roiImage = '';
                if (obj.roi_path) {
                    roiImage = `<img src="${obj.roi_path}" class="detection-image" alt="${obj.class}" 
                                data-bs-toggle="tooltip" title="${obj.class}">`;
                } else {
                    roiImage = '<span class="badge bg-secondary">No image</span>';
                }
                
                row.innerHTML = `
                    <td>${index}</td>
                    <td>${obj.class}</td>
                    <td>${obj.first_seen_time}</td>
                    <td>${obj.last_seen_time}</td>
                    <td>${obj.duration_time}</td>
                    <td>${roiImage}</td>
                `;
                
                detectionsTable.appendChild(row);
                index++;
            });
            
            // Initialize tooltips
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(toolttipTriggerEl);
            });
        }
        
        // Re-enable process button for another run
        processBtn.disabled = false;
    }
    
    // Utility function to format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Utility function to format duration
    function formatDuration(seconds) {
        if (!seconds || isNaN(seconds)) return '00:00:00';
        
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        return [
            hours.toString().padStart(2, '0'),
            minutes.toString().padStart(2, '0'),
            secs.toString().padStart(2, '0')
        ].join(':');
    }
    
    // Add event listener for image clicks to show larger preview
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('detection-image')) {
            const imgSrc = e.target.src;
            const imgAlt = e.target.alt;
            
            // Create modal for image preview
            const modal = document.createElement('div');
            modal.className = 'modal fade';
            modal.id = 'imagePreviewModal';
            modal.tabIndex = '-1';
            modal.setAttribute('aria-labelledby', 'imagePreviewModalLabel');
            modal.setAttribute('aria-hidden', 'true');
            
            modal.innerHTML = `
                <div class="modal-dialog modal-dialog-centered modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="imagePreviewModalLabel">${imgAlt}</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body text-center">
                            <img src="${imgSrc}" class="img-fluid" alt="${imgAlt}">
                        </div>
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
            
            // Show the modal
            const modalInstance = new bootstrap.Modal(modal);
            modalInstance.show();
            
            // Remove modal from DOM after it's hidden
            modal.addEventListener('hidden.bs.modal', function() {
                document.body.removeChild(modal);
            });
        }
    });
    
    // Add a reset button functionality
    const resetBtn = document.createElement('button');
    resetBtn.className = 'btn btn-secondary mt-3';
    resetBtn.innerHTML = '<i class="bi bi-arrow-repeat"></i> Process New Video';
    resetBtn.style.display = 'none';
    
    // Add reset button after results section
    resultsSection.appendChild(resetBtn);
    
    resetBtn.addEventListener('click', function() {
        // Reset form and UI
        videoUploadInput.value = '';
        videoPreviewContainer.classList.add('d-none');
        processingStatus.classList.add('d-none');
        resultsSection.classList.add('d-none');
        resetBtn.style.display = 'none';
        
        // Reset variables
        uploadInfo = null;
        taskId = null;
        
        // Clear intervals if any
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
            statusCheckInterval = null;
        }
    });
    
    // Show reset button after results are displayed
    const originalDisplayResults = displayResults;
    displayResults = function(result) {
        originalDisplayResults(result);
        resetBtn.style.display = 'block';
    };
});