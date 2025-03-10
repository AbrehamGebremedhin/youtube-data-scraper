from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, Path
from pydantic import BaseModel, Field
import datetime
import sys
import os
import json
import asyncio
import threading
from contextlib import asynccontextmanager

# Add the parent directory to the Python path so we can import modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from youtube_scraper import (
    download_videos, 
    find_candidate_videos, 
    download_single_video, 
    get_candidate_video,
    cancel_download_by_candidate_id
)
from models import get_db_session, Video, CandidateVideo, Image
import uvicorn
from typing import Optional, List, Dict

# Update these imports to use the Downloader and ImageDownloader directly
from downloader.Downloader import Downloader
from downloader.image_downloader import ImageDownloader

# Store for active WebSocket connections
active_connections: List[WebSocket] = []

# Store for download progress by video URL
download_progress: Dict[str, Dict] = {}

# Store for image download progress
image_download_progress: Dict[str, Dict] = {}

# Store the main event loop for thread-safe operations
main_event_loop = None

# Create separate named Downloader instances
video_downloader_instance = None
image_downloader_instance = None

# Define startup and shutdown events using lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic (formerly in @app.on_event("startup"))
    global main_event_loop, image_downloader_instance, video_downloader_instance
    main_event_loop = asyncio.get_running_loop()
    
    # Import check_and_update_schema and run it to ensure database is up to date
    from models import check_and_update_schema
    check_and_update_schema()
    
    # Initialize the image downloader as a separate instance
    image_downloader_instance = Downloader('image_downloader')
    # Create and set the ImageDownloader implementation directly
    image_downloader_impl = ImageDownloader()
    image_downloader_instance.set_downloader_implementation(image_downloader_impl)
    
    # Initialize the video downloader in youtube_scraper and our own instance
    video_downloader_instance = Downloader('video_downloader')
    video_downloader_instance.reset_to_default_downloader()
    
    # Try to initialize video downloader through youtube_scraper if available
    try:
        # Check if the function exists or if we need to manually create it
        from youtube_scraper import reinitialize_video_downloader
        reinitialize_video_downloader()
        print("Video downloader reinitialized through youtube_scraper")
    except ImportError:
        print("Warning: The initialize_video_downloader function is missing from youtube_scraper")
    except Exception as e:
        print(f"Warning: Could not initialize video downloader through youtube_scraper: {e}")
    
    # Check for Chrome installation for image scraping
    try:
        from rest.startup import check_chrome_installation
        check_chrome_installation()
    except ImportError:
        print("Warning: Could not check Chrome installation for image scraping")
    
    yield  # This is where the app runs
    
    # Shutdown logic (if needed)
    # Clean up resources, close connections, etc.
    print("Shutting down app resources...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="YouTube Video Downloader API",
    description="Download YouTube videos for QA testing purposes",
    lifespan=lifespan
)

# Base model for video search/download parameters
class VideoBaseRequest(BaseModel):
    category: str = Field(..., description="Video category to search or download")
    num_videos: Optional[int] = Field(default=5, ge=1, description="Number of videos to search or download")
    min_duration: Optional[int] = Field(default=300, ge=0, description="Minimum video duration in seconds")
    max_duration: Optional[int] = Field(default=600, gt=0, description="Maximum video duration in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "category": "sports",
                "num_videos": 5,
                "min_duration": 300,
                "max_duration": 600
            }
        }

# Request model for downloads that includes approved_videos
class DownloadRequest(VideoBaseRequest):
    approved_videos: Optional[List[str]] = Field(default=None, description="List of pre-approved video URLs to download")

    class Config:
        json_schema_extra = {
            "example": {
                "category": "sports",
                "num_videos": 5,
                "min_duration": 300,
                "max_duration": 600,
                "approved_videos": None
            }
        }

# Request model for finding candidates (inherits all base parameters)
class CandidateRequest(VideoBaseRequest):
    pass

class DownloadByIdRequest(BaseModel):
    candidate_id: str = Field(..., description="ID of the candidate video to download")

    class Config:
        json_schema_extra = {
            "example": {
                "candidate_id": "01234567-89ab-cdef-0123-456789abcdef"
            }
        }

class ApproveVideoRequest(BaseModel):
    download_path: Optional[str] = Field(None, description="Optional custom download path")

    class Config:
        json_schema_extra = {
            "example": {
                "download_path": "D:\\Videos\\CustomFolder"
            }
        }

# New model for image download request
class ImageDownloadRequest(BaseModel):
    category: str = Field(..., description="Image category to search for")
    num_images: int = Field(10, ge=1, le=50, description="Number of images to download")
    quality: str = Field("1080p", description="Image quality (480p, 720p, 1080p, 4k)")
    custom_download_path: Optional[str] = Field(None, description="Optional custom download path")

    class Config:
        json_schema_extra = {
            "example": {
                "category": "nature",
                "num_images": 10,
                "quality": "1080p",
                "custom_download_path": None
            }
        }

# Function to broadcast messages to all connected clients
async def broadcast_message(message: dict):
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception:
            # If sending fails, we'll handle disconnection elsewhere
            pass

# Thread-safe function to queue a broadcast message
def queue_broadcast(message: dict):
    global main_event_loop
    if main_event_loop and main_event_loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast_message(message), main_event_loop)
    else:
        print(f"Warning: Cannot broadcast message, no running event loop: {message}")

# Register progress callback for the downloader - thread-safe version
def register_progress_callback(url, percent, status="downloading"):
    global download_progress
    download_progress[url] = {"percent": percent, "status": status}
    # Use the thread-safe function to broadcast
    queue_broadcast({"type": "progress", "data": download_progress})

# Register progress callback for the image downloader - thread-safe version
def register_image_progress_callback(category, percent, status="downloading"):
    global image_download_progress
    image_download_progress[category] = {"percent": percent, "status": status}
    # Use the thread-safe function to broadcast
    queue_broadcast({"type": "image_progress", "data": image_download_progress})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        # Send initial progress data when client connects
        await websocket.send_json({
            "type": "progress", 
            "data": download_progress
        })
        
        # Also send image download progress if any
        if image_download_progress:
            await websocket.send_json({
                "type": "image_progress", 
                "data": image_download_progress
            })
        
        # Listen for messages from the client
        while True:
            data = await websocket.receive_text()
            # Handle any client messages if needed
            await websocket.send_json({"type": "message", "data": f"Received: {data}"})
    except WebSocketDisconnect:
        # Remove connection when client disconnects
        active_connections.remove(websocket)

@app.post("/api/approve/{candidate_id}")
async def api_approve_video(
    candidate_id: str = Path(..., description="ID of the candidate video to approve and download"),
    request: ApproveVideoRequest = None
):
    """Approve and download a specific candidate video by its ID."""
    try:
        # Get the download path from the request if provided
        download_path = request.download_path if request and request.download_path else None
        
        # Verify the candidate exists
        candidate_info = get_candidate_video(candidate_id)
        if not candidate_info:
            raise HTTPException(status_code=404, detail=f"Candidate video with ID {candidate_id} not found")
        
        # Verify the download path is valid if provided
        if download_path and not os.path.isdir(download_path):
            raise HTTPException(status_code=400, detail=f"Download path '{download_path}' does not exist or is not a directory")
        
        # Store the current event loop for thread-safe operations
        global main_event_loop
        main_event_loop = asyncio.get_event_loop()
        
        # Clear previous progress data for this video
        if candidate_info['url'] in download_progress:
            del download_progress[candidate_info['url']]
            
        # Broadcast that we're starting a new download
        await broadcast_message({
            "type": "start_single", 
            "data": {
                "candidate_id": candidate_id,
                "title": candidate_info['title'],
                "category": candidate_info['category'],
                "download_path": download_path
            }
        })
        
        # Make sure we're using the correct downloader for videos
        # First try to use youtube_scraper's reinitialize function
        try:
            from youtube_scraper import reinitialize_video_downloader
            reinitialize_video_downloader()
            print("Video downloader reinitialized through youtube_scraper")
        except Exception as e:
            print(f"Error reinitializing video downloader through youtube_scraper: {e}")
            # Fall back to our own instance if available
            global video_downloader_instance
            if video_downloader_instance:
                video_downloader_instance.reset_to_default_downloader()
                print("Video downloader reset to default implementation")
        
        # Run the download in a separate thread to avoid blocking the event loop
        result = await asyncio.to_thread(
            download_single_video,
            candidate_id,
            progress_callback=register_progress_callback,
            download_path=download_path
        )
        
        # Broadcast download completion
        await broadcast_message({"type": "complete_single", "data": result})
        
        if not result.get('success', False):
            raise HTTPException(status_code=500, detail=result.get('message', 'Download failed'))
            
        # Only remove the candidate from the database after successful download
        if result.get('success', False):
            try:
                db_session = get_db_session()
                candidate = db_session.query(CandidateVideo).filter(CandidateVideo.id == candidate_id).first()
                if candidate:
                    db_session.delete(candidate)
                    db_session.commit()
                db_session.close()
            except Exception as e:
                print(f"Error removing candidate from database: {e}")
                # We don't want to fail the whole operation if just the cleanup fails
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        # Broadcast error
        error_message = f"Server error: {str(e)}"
        await broadcast_message({"type": "error", "data": {"message": error_message}})
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/api/candidate/{candidate_id}")
async def api_get_candidate(candidate_id: str = Path(..., description="ID of the candidate video to retrieve")):
    """Get detailed information about a specific candidate video."""
    try:
        # Get candidate info
        candidate_info = get_candidate_video(candidate_id)
        if not candidate_info:
            raise HTTPException(status_code=404, detail=f"Candidate video with ID {candidate_id} not found")
            
        return {"success": True, "candidate": candidate_info}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve candidate: {str(e)}")

@app.get("/api/candidates")
async def api_get_candidates(
    category: Optional[str] = Query(None, description="Filter by video category"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of candidates to return")
):
    """Retrieve stored candidate videos with optional filtering."""
    try:
        session = get_db_session()
        query = session.query(CandidateVideo)
        
        # Apply filters if provided
        if category:
            query = query.filter(CandidateVideo.category == category)
        
        # Get most recent candidates first
        query = query.order_by(CandidateVideo.creation_date.desc()).limit(limit)
        
        candidates = []
        for c in query.all():
            candidates.append({
                "candidate_id": str(c.id),
                "video_id": c.video_id,
                "url": c.url,
                "title": c.title,
                "category": c.category,
                "description": c.description,
                "duration": c.duration,
                "height": c.height,
                "thumbnail": c.thumbnail,
                "vbr": c.vbr,
                "abr": c.abr,
                "view_count": c.view_count,
                "validation_message": c.validation_message,
                "status": c.status,
                "creation_date": c.creation_date.isoformat() if c.creation_date else None
            })
            
        return {"success": True, "candidates": candidates, "count": len(candidates)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve candidates: {str(e)}")
    finally:
        session.close()

@app.post("/api/candidates")
async def api_find_candidate_videos(request: CandidateRequest):
    """Find candidate videos for review without downloading them."""
    try:
        if request.max_duration < request.min_duration:
            raise HTTPException(status_code=400, detail="max_duration must be greater than min_duration")
        
        # Store the current event loop for thread-safe operations
        global main_event_loop
        main_event_loop = asyncio.get_event_loop()
        
        # Broadcast that we're starting to find candidates
        await broadcast_message({"type": "finding_candidates", "data": {"category": request.category}})
        
        # Run the search in a separate thread to avoid blocking the event loop
        result = await asyncio.to_thread(
            find_candidate_videos,
            request.category,
            request.num_videos,
            request.min_duration,
            request.max_duration
        )
        
        # Broadcast completion
        await broadcast_message({"type": "candidates_found", "data": result})
        
        if not result.get('success', False):
            raise HTTPException(status_code=500, detail=result.get('message', 'Failed to find videos'))
            
        return result
        
    except Exception as e:
        # Broadcast error
        error_message = f"Server error: {str(e)}"
        await broadcast_message({"type": "error", "data": {"message": error_message}})
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/api/download")
async def api_download_videos(request: DownloadRequest):
    """Download YouTube videos based on category and parameters."""
    try:
        if request.max_duration < request.min_duration:
            raise HTTPException(status_code=400, detail="max_duration must be greater than min_duration")
        
        # Store the current event loop for thread-safe operations
        global main_event_loop
        main_event_loop = asyncio.get_event_loop()
        
        # Clear previous progress data
        download_progress.clear()
        # Broadcast that we're starting a new download
        await broadcast_message({"type": "start", "data": {"category": request.category}})
        
        # Run the download in a separate thread to avoid blocking the event loop
        result = await asyncio.to_thread(
            download_videos,
            request.category,
            request.approved_videos,
            request.num_videos,
            request.min_duration,
            request.max_duration,
            progress_callback=register_progress_callback
        )
        
        # Broadcast download completion
        await broadcast_message({"type": "complete", "data": result})
        
        if not result.get('success', False):
            raise HTTPException(status_code=500, detail=result.get('message', 'Download failed'))
            
        return result
        
    except Exception as e:
        # Broadcast error
        error_message = f"Server error: {str(e)}"
        await broadcast_message({"type": "error", "data": {"message": error_message}})
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/api/cancel/{candidate_id}")
async def api_cancel_download(candidate_id: str = Path(..., description="ID of the candidate video download to cancel")):
    """Cancel an in-progress download for a specific candidate video."""
    try:
        # Store the current event loop for thread-safe operations
        global main_event_loop
        main_event_loop = asyncio.get_event_loop()
        
        # Run the cancellation in a separate thread to avoid blocking the event loop
        result = await asyncio.to_thread(cancel_download_by_candidate_id, candidate_id)
        
        # Broadcast the cancellation result
        await broadcast_message({
            "type": "download_cancelled", 
            "data": result
        })
        
        if not result.get("success", False):
            # Don't raise an exception for non-active downloads, just return the result
            if "No active download found" in result.get("message", ""):
                return result
            raise HTTPException(status_code=400, detail=result.get("message", "Failed to cancel download"))
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        error_message = f"Server error: {str(e)}"
        await broadcast_message({"type": "error", "data": {"message": error_message}})
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/api/history")
async def get_download_history(
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of records to return"),
    status: Optional[str] = Query(None, description="Filter by status (pending, success, failed)"),
    category: Optional[str] = Query(None, description="Filter by video category")
):
    """Retrieve download history with optional filters."""
    try:
        db = get_db_session()
        query = db.query(Video)
        
        # Apply filters if provided
        if status:
            query = query.filter(Video.status == status)
        if category:
            query = query.filter(Video.category == category)
        
        # Order by latest downloads first
        query = query.order_by(Video.download_date.desc()).limit(limit)
        
        # Execute query and convert to dictionary
        videos = query.all()
        result = []
        
        for video in videos:
            result.append({
                "id": video.id,
                "url": video.url,
                "title": video.title,
                "category": video.category,
                "download_date": video.download_date.isoformat() if video.download_date else None,
                "status": video.status,
                "duration": video.duration,
                "resolution": video.resolution,
                "file_path": video.file_path,
                "error_message": video.error_message
            })
            
        return {"count": len(result), "videos": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")
    finally:
        db.close()

@app.get("/api/candidates/history")
async def api_get_candidate_history(
    category: Optional[str] = Query(None, description="Filter by video category"),
    status: Optional[str] = Query(None, description="Filter by status (pending, approved, downloaded, rejected)"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of candidates to return"),
    days: int = Query(30, ge=1, le=365, description="How many days of history to include")
):
    """Retrieve historical candidate videos with optional filtering."""
    try:
        session = get_db_session()
        query = session.query(CandidateVideo)
        
        # Apply filters if provided
        if category:
            query = query.filter(CandidateVideo.category == category)
        
        if status:
            query = query.filter(CandidateVideo.status == status)
        
        # Apply date filter
        oldest_date = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        query = query.filter(CandidateVideo.creation_date >= oldest_date)
        
        # Get most recent candidates first
        query = query.order_by(CandidateVideo.creation_date.desc()).limit(limit)
        
        candidates = []
        for c in query.all():
            candidates.append({
                "candidate_id": str(c.id),
                "video_id": c.video_id,
                "url": c.url,
                "title": c.title,
                "category": c.category,
                "description": c.description,
                "duration": c.duration,
                "height": c.height,
                "thumbnail": c.thumbnail,
                "vbr": c.vbr,
                "abr": c.abr,
                "view_count": c.view_count,
                "validation_message": c.validation_message,
                "status": c.status,
                "creation_date": c.creation_date.isoformat() if c.creation_date else None,
                "last_download_date": c.last_download_date.isoformat() if c.last_download_date else None
            })
            
        return {"success": True, "candidates": candidates, "count": len(candidates)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve historical candidates: {str(e)}")
    finally:
        session.close()

@app.delete("/api/candidate/{candidate_id}")
async def api_delete_candidate(candidate_id: str = Path(..., description="ID of the candidate video to delete")):
    """Delete a specific candidate video from the database."""
    try:
        db_session = get_db_session()
        candidate = db_session.query(CandidateVideo).filter(CandidateVideo.id == candidate_id).first()
        
        if not candidate:
            raise HTTPException(status_code=404, detail=f"Candidate video with ID {candidate_id} not found")
            
        # Delete the candidate
        db_session.delete(candidate)
        db_session.commit()
        db_session.close()
        
        return {
            "success": True,
            "message": f"Candidate video with ID {candidate_id} has been deleted"
        }
            
    except HTTPException:
        raise
    except Exception as e:
        if 'db_session' in locals():
            db_session.close()
        raise HTTPException(status_code=500, detail=f"Failed to delete candidate: {str(e)}")

@app.post("/api/images/download")
async def api_download_images(request: ImageDownloadRequest):
    """Download high-quality images based on category."""
    try:
        # Verify the download path is valid if provided
        if request.custom_download_path and not os.path.isdir(request.custom_download_path):
            raise HTTPException(status_code=400, detail=f"Download path '{request.custom_download_path}' does not exist or is not a directory")
        
        # Store the current event loop for thread-safe operations
        global main_event_loop
        main_event_loop = asyncio.get_running_loop()
        
        # Clear previous progress data for this category
        if request.category in image_download_progress:
            del image_download_progress[request.category]
            
        # Broadcast that we're starting a new image download
        await broadcast_message({
            "type": "start_image_download", 
            "data": {
                "category": request.category,
                "num_images": request.num_images,
                "quality": request.quality
            }
        })
        
        # Get download directory
        download_dir = request.custom_download_path
        if not download_dir:
            download_dir = os.path.join(os.getcwd(), f"{request.category}_images")
            
        # Get access to the ImageDownloader implementation directly
        global image_downloader_instance
        if image_downloader_instance is None:
            image_downloader_instance = Downloader('image_downloader')
            image_downloader_impl = ImageDownloader()
            image_downloader_instance.set_downloader_implementation(image_downloader_impl)
        else:
            # Get the implementation if available
            try:
                image_downloader_impl = image_downloader_instance._downloader_implementation
                if not isinstance(image_downloader_impl, ImageDownloader):
                    image_downloader_impl = ImageDownloader()
                    image_downloader_instance.set_downloader_implementation(image_downloader_impl)
            except AttributeError:
                image_downloader_impl = ImageDownloader()
                image_downloader_instance.set_downloader_implementation(image_downloader_impl)

        # Pass the num_images in the quality string (format: "quality:num_images")
        quality_with_count = f"{request.quality}:{request.num_images}"

        # Start the download process (using category as the "link" parameter)
        result = await asyncio.to_thread(
            image_downloader_instance.download_video,  # Using the existing method
            request.category,  # Using category as the link
            request.category,
            'jpg',  # Image type
            quality_with_count,  # Quality setting with count embedded
            register_image_progress_callback,  # Progress callback
            None,  # No video_info
            download_dir  # Custom download dir
        )
        
        # Return immediate response - downloading continues in background
        result = {
            "success": True,
            "message": f"Started downloading {request.num_images} images for category '{request.category}' at {request.quality}",
            "category": request.category,
            "download_dir": download_dir
        }
        
        # Broadcast download initiation
        await broadcast_message({"type": "image_download_started", "data": result})
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        # Broadcast error
        error_message = f"Server error: {str(e)}"
        await broadcast_message({"type": "error", "data": {"message": error_message}})
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/api/images/status/{download_id}")
async def api_get_image_download_status(download_id: str = Path(..., description="ID of the image download job")):
    """Get the status of an image download job."""
    try:
        global image_downloader_instance
        
        if image_downloader_instance is None:
            return {"status": "not_found", "message": "No image downloader instance available"}
            
        # Try to access the implementation directly
        try:
            downloader_impl = image_downloader_instance._downloader_implementation
            if isinstance(downloader_impl, ImageDownloader):
                # Use the direct implementation's method
                status = downloader_impl.get_status(download_id)
                return status
        except AttributeError:
            pass
            
        # Fallback to the global download progress tracking
        global image_download_progress
        # For now, we'll just return the progress data we have
        categories = list(image_download_progress.keys())
        if categories:
            return {"status": "active", "progress": image_download_progress}
        return {"status": "not_found", "message": "No active image downloads found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving download status: {str(e)}")

@app.post("/api/images/cancel/{category}")
async def api_cancel_image_download(category: str = Path(..., description="Category of images to cancel download")):
    """Cancel an active image download job."""
    try:
        global image_downloader_instance
        
        if image_downloader_instance is None:
            raise HTTPException(status_code=404, detail="No image downloader instance available")
        
        # Use the cancel_download method, with category as the "link" parameter
        result = await asyncio.to_thread(
            image_downloader_instance.cancel_download,
            category
        )
        
        if not result:
            # Don't raise an exception if no active download - just return a more informative response
            return {"success": False, "message": f"No active download found for category '{category}'"}
            
        # Broadcast cancellation
        await broadcast_message({
            "type": "image_download_cancelled", 
            "data": {"category": category, "success": True}
        })
        
        return {"success": True, "message": f"Download for category '{category}' has been cancelled"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling download: {str(e)}")

@app.get("/api/images/history")
async def api_get_image_history(
    category: Optional[str] = Query(None, description="Filter by image category"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of records to return")
):
    """Retrieve image download history with optional filtering."""
    try:
        session = get_db_session()
        query = session.query(Image)
        
        # Apply filters if provided
        if category:
            query = query.filter(Image.category == category)
        
        # Order by latest downloads first
        query = query.order_by(Image.download_date.desc()).limit(limit)
        
        # Execute query and convert to dictionary
        images = query.all()
        result = []
        
        for img in images:
            result.append({
                "id": img.id,
                "url": img.url,
                "title": img.title,
                "category": img.category,
                "download_date": img.download_date.isoformat() if img.download_date else None,
                "file_path": img.file_path,
                "status": img.status,
                "width": img.width,
                "height": img.height,
                "file_size": img.file_size,
                "thumbnail": img.thumbnail
            })
            
        session.close()
        return {"success": True, "count": len(result), "images": result}
        
    except Exception as e:
        if 'session' in locals():
            session.close()
        raise HTTPException(status_code=500, detail=f"Error retrieving image history: {str(e)}")

@app.get("/")
async def root():
    """API root endpoint showing service information."""
    return {
        "service": "YouTube Video Downloader API",
        "endpoints": [
            {
                "path": "/api/candidates",
                "method": "POST",
                "description": "Find candidate videos for user review"
            },
            {
                "path": "/api/candidates",
                "method": "GET",
                "description": "Get stored candidate videos"
            },
            {
                "path": "/api/candidates/history",
                "method": "GET",
                "description": "Get historical candidate videos"
            },
            {
                "path": "/api/candidate/{candidate_id}",
                "method": "GET",
                "description": "Get details for a specific candidate video"
            },
            {
                "path": "/api/approve/{candidate_id}",
                "method": "POST",
                "description": "Approve and download a specific candidate video"
            },
            {
                "path": "/api/cancel/{candidate_id}",
                "method": "POST", 
                "description": "Cancel an in-progress download for a specific candidate video"
            },
            {
                "path": "/api/download",
                "method": "POST",
                "description": "Download approved YouTube videos by category (legacy)"
            },
            {
                "path": "/ws",
                "method": "WebSocket",
                "description": "WebSocket connection for real-time download progress"
            },
            {
                "path": "/api/history",
                "method": "GET",
                "description": "Retrieve download history with optional filters"
            },
            {
                "path": "/api/candidate/{candidate_id}",
                "method": "DELETE",
                "description": "Delete a specific candidate video from the database"
            },
            {
                "path": "/api/images/download",
                "method": "POST",
                "description": "Download images by category"
            },
            {
                "path": "/api/images/status/{download_id}",
                "method": "GET", 
                "description": "Get status of an image download job"
            },
            {
                "path": "/api/images/cancel/{download_id}",
                "method": "POST", 
                "description": "Cancel an image download job"
            },
            {
                "path": "/api/images/history",
                "method": "GET",
                "description": "Retrieve image download history"
            }
        ]
    }

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
