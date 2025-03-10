import os
import concurrent.futures
import time
from threading import Lock
import datetime
from typing import Dict, Optional, List

# Import the database models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rest.models import Video, get_db_session

# Import the new downloader implementations
from .yt_dlp_downloader import YtDlpDownloader
from .base_downloader import BaseDownloader
from .image_downloader import ImageDownloader

class Downloader:
    _instances = {}
    _lock = Lock()
    
    def __new__(cls, instance_name='default'):
        with cls._lock:
            if instance_name not in cls._instances:
                instance = super(Downloader, cls).__new__(cls)
                instance._initialized = False
                cls._instances[instance_name] = instance
            return cls._instances[instance_name]
    
    def __init__(self, instance_name='default'):
        if self._initialized:
            return
            
        # Initialize the downloader implementation
        self._downloader: BaseDownloader = YtDlpDownloader()
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self._active_downloads = {}
        self._progress_callbacks = {}
        self._downloads_lock = Lock()
        self._initialized = True
        self._instance_name = instance_name
        
        # Allow changing the downloader implementation
        self._is_image_downloader = False
    
    def reset_to_default_downloader(self):
        """Reset to the default video downloader implementation"""
        self._downloader = YtDlpDownloader()
        self._is_image_downloader = False
        print(f"Downloader instance '{self._instance_name}' reset to default YtDlpDownloader")
    
    def set_downloader_implementation(self, downloader_impl: BaseDownloader):
        """Set a different downloader implementation"""
        if not isinstance(downloader_impl, BaseDownloader):
            raise TypeError("Downloader implementation must implement BaseDownloader interface")
        self._downloader = downloader_impl
        self._is_image_downloader = isinstance(downloader_impl, ImageDownloader)
        print(f"Downloader instance '{self._instance_name}' set to {downloader_impl.__class__.__name__}")
    
    def download_video(self, link, category, video_type='mp4', quality='1080', progress_callback=None, video_info=None, custom_download_dir=None):
        """
        Downloads a YouTube video to a category-specific directory using thread pool.
        
        Args:
            link (str): YouTube video URL
            category (str): Category name for directory organization
            video_type (str): Video format (e.g., mp4, webm)
            quality (str): Video quality (e.g., 1080, 720, best)
            progress_callback (callable): Function to call with progress updates
            video_info (dict): Optional video metadata
            custom_download_dir (str): Optional custom download directory
            
        Returns:
            concurrent.futures.Future: A Future representing the download task
        """
        # Record the download in the database
        self._record_download_start(link, category, video_info)
        
        if progress_callback:
            with self._downloads_lock:
                self._progress_callbacks[link] = progress_callback
                # Initialize progress at 0%
                progress_callback(link, 0, "initializing")
                
        future = self._thread_pool.submit(
            self._download_video_task, link, category, video_type, quality, custom_download_dir
        )
        
        with self._downloads_lock:
            self._active_downloads[link] = future
            
        future.add_done_callback(lambda f: self._handle_download_completion(link, f, custom_download_dir))
        return future
    
    def _record_download_start(self, link, category, video_info=None):
        """Records the start of a download in the database"""
        try:
            session = get_db_session()
            # Check if this URL is already in the database
            existing_video = session.query(Video).filter(Video.url == link).first()
            
            if existing_video:
                # Update existing record
                existing_video.status = "pending"
                existing_video.download_date = datetime.datetime.now()  # Use datetime object instead of string
                existing_video.category = category
                if video_info:
                    if 'title' in video_info:
                        existing_video.title = video_info['title']
                    if 'duration' in video_info:
                        existing_video.duration = video_info['duration']
                    if 'height' in video_info:
                        existing_video.resolution = f"{video_info['height']}p"
                    if 'vbr' in video_info:
                        existing_video.video_bitrate = video_info['vbr']
                    if 'abr' in video_info:
                        existing_video.audio_bitrate = video_info['abr']
            else:
                # Create new record
                new_video = Video(url=link, category=category, status="pending")
                if video_info:
                    if 'title' in video_info:
                        new_video.title = video_info['title']
                    if 'duration' in video_info:
                        new_video.duration = video_info['duration']
                    if 'height' in video_info:
                        new_video.resolution = f"{video_info['height']}p"
                    if 'vbr' in video_info:
                        new_video.video_bitrate = video_info['vbr']
                    if 'abr' in video_info:
                        new_video.audio_bitrate = video_info['abr']
                session.add(new_video)
                
            session.commit()
            session.close()
        except Exception as e:
            print(f"Error recording download start: {e}")
            # Don't let database errors stop the download
            if 'session' in locals():
                session.close()
    
    def _handle_download_completion(self, link, future, custom_download_dir=None):
        """Handle download completion and update database"""
        try:
            result = future.result()
            success = isinstance(result, dict) and result.get("status") == "success"
            
            # Update database with completion status
            session = get_db_session()
            video = session.query(Video).filter(Video.url == link).first()
            
            if video:
                if custom_download_dir:
                    download_dir = custom_download_dir
                else:
                    download_dir = os.path.join(os.getcwd(), f"{video.category}_videos")
                    
                if success:
                    video.status = "success"
                    # Try to find the file path by listing files in the directory
                    # and finding the most recently modified file
                    if os.path.exists(download_dir):
                        files = os.listdir(download_dir)
                        if files:
                            # Find the most recently modified file
                            latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(download_dir, f)))
                            video.file_path = os.path.join(download_dir, latest_file)
                else:
                    video.status = "failed"
                    if isinstance(result, str):
                        video.error_message = result
                    elif isinstance(result, dict) and "error" in result:
                        video.error_message = result["error"]
                
                session.commit()
            session.close()
        except Exception as e:
            print(f"Error updating download status: {e}")
            if 'session' in locals():
                session.close()
        finally:
            # Remove from active downloads
            self._remove_download(link)
    
    def _download_video_task(self, link, category, video_type='mp4', quality='1080', custom_download_dir=None):
        """Internal method that performs the actual download in a thread."""
        try:
            # Use custom download path if provided, otherwise default to category folder
            if custom_download_dir:
                download_dir = custom_download_dir
            else:
                download_dir = os.path.join(os.getcwd(), f"{category}_videos")
                
            # Define a progress callback that will use our internal callback system
            def progress_update_callback(link, percent, status):
                self._update_progress(link, percent, status)
                
            # Delegate to the downloader implementation
            result = self._downloader.download(
                link=link,
                download_dir=download_dir,
                quality=quality,
                video_type=video_type,
                progress_callback=progress_update_callback
            )
            
            return result
            
        except Exception as e:
            self._update_progress(link, 0, "error")
            return f"Error downloading {link}: {str(e)}"
    
    def _update_progress(self, link, percent, status="downloading"):
        """Updates download progress via callback if registered."""
        with self._downloads_lock:
            if link in self._progress_callbacks:
                try:
                    # Call the progress callback with proper error handling
                    callback = self._progress_callbacks[link]
                    try:
                        callback(link, percent, status)
                    except Exception as e:
                        # Just log the error but don't raise it to avoid stopping the download
                        print(f"Progress callback error (continuing download): {e}")
                except Exception as e:
                    print(f"Error accessing progress callback: {e}")
    
    def _remove_download(self, link):
        """Remove a download from the active downloads dictionary once complete."""
        with self._downloads_lock:
            if link in self._active_downloads:
                del self._active_downloads[link]
            # The _processes attribute no longer exists in this class
            # It was moved to the YtDlpDownloader class
            # Keep the callback for final progress updates
    
    def get_active_downloads(self):
        """Returns a list of URLs currently being downloaded."""
        with self._downloads_lock:
            return list(self._active_downloads.keys())
    
    def shutdown(self):
        """Shutdown the thread pool and wait for all downloads to complete."""
        self._thread_pool.shutdown(wait=True)
        
    def cancel_download(self, link):
        """Cancel a specific download if it's still running."""
        was_cancelled = False
        future_to_cancel = None
        
        # Minimize lock scope - only use it to get references to objects we need to operate on
        with self._downloads_lock:
            if link in self._active_downloads:
                future_to_cancel = self._active_downloads[link]
                del self._active_downloads[link]
                was_cancelled = True
        
        # Cancel the download with our implementation
        self._downloader.cancel(link)
        
        # Cancel the future if it exists
        if future_to_cancel:
            try:
                future_to_cancel.cancel()
            except Exception as e:
                print(f"Error cancelling future: {e}")
        
        # Update progress to show cancellation
        self._update_progress(link, 0, "cancelled")
        
        # Clean up any partial download files
        if was_cancelled:
            try:
                self._clean_partial_download_files(link)
            except Exception as e:
                print(f"Error cleaning up partial download files: {e}")
        
        # Update database to reflect cancelled status
        if was_cancelled:
            try:
                session = get_db_session()
                video = session.query(Video).filter(Video.url == link).first()
                if video:
                    video.status = "cancelled"
                    video.download_date = datetime.datetime.now()
                    video.error_message = "Download cancelled by user"
                    session.commit()
                session.close()
            except Exception as e:
                print(f"Error updating database for cancelled download: {e}")
                if 'session' in locals():
                    session.close()
            
        return was_cancelled

    def _clean_partial_download_files(self, link):
        """Clean up any partial download files for a cancelled download."""
        try:
            # Extract video ID to help identify relevant files
            video_id = None
            if "youtube.com" in link or "youtu.be" in link:
                # Extract YouTube video ID
                if "v=" in link:
                    video_id = link.split("v=")[1].split("&")[0]
                elif "youtu.be/" in link:
                    video_id = link.split("youtu.be/")[1].split("?")[0]
            
            if not video_id:
                print("Could not extract video ID for cleanup, skipping file cleanup")
                return
                
            print(f"Looking for partial files for video ID: {video_id}")
            
            # First try to get information about the video from the database
            session = None
            try:
                session = get_db_session()
                video = session.query(Video).filter(Video.url == link).first()
                
                if video and video.category:
                    # First check if there's a file path specified in the database
                    if video.file_path and os.path.exists(os.path.dirname(video.file_path)):
                        self._downloader.clean_partial_files(link, video_id, os.path.dirname(video.file_path))
                    
                    # Also check the default category directory
                    category_dir = os.path.join(os.getcwd(), f"{video.category}_videos")
                    if os.path.exists(category_dir):
                        self._downloader.clean_partial_files(link, video_id, category_dir)
                
                # Close the session when done
                if session:
                    session.close()
            except Exception as e:
                print(f"Error accessing database for cleanup: {e}")
                if session:
                    session.close()
            
            # If we couldn't get info from the database, try common download directories
            common_dirs = [
                os.path.join(os.getcwd(), "downloads"),
                os.path.join(os.getcwd(), "videos")
            ]
            
            for directory in common_dirs:
                if os.path.exists(directory):
                    self._downloader.clean_partial_files(link, video_id, directory)
                    
        except Exception as e:
            print(f"Error during file cleanup: {e}")
    
    def _clean_directory_for_video(self, directory, video_id, link):
        """This method is now delegated to the downloader implementation."""
        return self._downloader.clean_partial_files(link, video_id, directory)

    def get_download_status(self, link):
        """Get the current status of a download."""
        # First check if it's an active download
        with self._downloads_lock:
            if link in self._active_downloads:
                return "active"
                
        # If not active, check the database
        try:
            session = get_db_session()
            video = session.query(Video).filter(Video.url == link).first()
            if video:
                return video.status
            session.close()
        except Exception as e:
            print(f"Error getting download status: {e}")
            if 'session' in locals():
                session.close()
                
        return "unknown"