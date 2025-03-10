import os
import subprocess
import re
import psutil
from typing import Callable, Dict, Any, Optional
import time

from .base_downloader import BaseDownloader

class YtDlpDownloader(BaseDownloader):
    """YouTube video downloader using yt-dlp"""
    
    def __init__(self, auto_update=False):
        self._processes = {}  # Track subprocess objects by link
        if auto_update:
            self.update_yt_dlp()
        
    def update_yt_dlp(self) -> Dict[str, Any]:
        """
        Check for updates to yt-dlp and update to the latest version.
        
        Returns:
            dict: Status information about the update
        """
        try:
            print("Checking for yt-dlp updates...")
            
            # Run pip install --upgrade yt-dlp
            process = subprocess.run(
                ["pip", "install", "--upgrade", "yt-dlp"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if process.returncode == 0:
                if "Requirement already satisfied" in process.stdout and "is up to date" in process.stdout:
                    print("yt-dlp is already up to date")
                    return {"status": "up-to-date", "message": "yt-dlp is already up to date"}
                else:
                    print("yt-dlp has been updated to the latest version")
                    return {"status": "updated", "message": "yt-dlp has been updated to the latest version"}
            else:
                print(f"Error updating yt-dlp: {process.stderr}")
                return {
                    "status": "failed",
                    "error": f"Error updating yt-dlp: {process.stderr}"
                }
        except Exception as e:
            print(f"Exception while updating yt-dlp: {str(e)}")
            return {
                "status": "failed",
                "error": f"Exception while updating yt-dlp: {str(e)}"
            }
    
    def download(self, 
                link: str, 
                download_dir: str, 
                quality: str = '1080',
                video_type: str = 'mp4',
                progress_callback: Optional[Callable] = None) -> dict:
        """
        Download YouTube video using yt-dlp
        
        Args:
            link: YouTube URL
            download_dir: Directory to save the video
            quality: Video quality (height)
            video_type: Video format (e.g., mp4)
            progress_callback: Function to receive progress updates
            
        Returns:
            dict: Status information with success/failure details
        """
        try:
            os.makedirs(download_dir, exist_ok=True)
            
            # Enforce specified format
            format_spec = f'bestvideo[height={quality}][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]'
            
            retries = 3
            while retries > 0:
                try:
                    # Configure yt-dlp command with all parameters
                    command = (
                        f'yt-dlp --no-check-certificate '
                        f'--cookies cookies.txt '
                        f'--format "{format_spec}" '
                        f'--merge-output-format mp4 '
                        f'--output "{download_dir}/%(title)s.%(ext)s" '
                        f'"{link}"'
                    )
                    
                    # Execute the download command
                    print(f"Starting download of {link}")
                    
                    # Use Popen to capture output in real-time
                    process = subprocess.Popen(
                        command, 
                        shell=True, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        bufsize=1
                    )
                    
                    # Store process reference for cancellation
                    self._processes[link] = process
                    
                    # Pattern to match progress percentage
                    progress_pattern = re.compile(r'\[download\]\s+(\d+\.\d+)%')
                    merging_pattern = re.compile(r'Merging formats')
                    
                    # Read output line by line to track progress
                    for line in iter(process.stdout.readline, ''):
                        print(line, end='')
                        
                        # Look for progress percentage
                        progress_match = progress_pattern.search(line)
                        if progress_match and progress_callback:
                            percent = float(progress_match.group(1))
                            progress_callback(link, percent, "downloading")
                        
                        # Check if merging started
                        if merging_pattern.search(line) and progress_callback:
                            progress_callback(link, 95, "merging")
                    
                    # Wait for process to complete
                    process.wait()
                    
                    # Remove process reference when complete
                    if link in self._processes:
                        del self._processes[link]
                    
                    # Final progress update
                    if progress_callback:
                        progress_callback(link, 100, "complete")
                    
                    print(f"Download complete for {link}")
                    return {"status": "success", "link": link}
                    
                except Exception as e:
                    # Clean up process reference if exception occurs
                    if link in self._processes:
                        del self._processes[link]
                            
                    if "WinError 32" in str(e) and retries > 1:
                        print(f"Temporary file access error for {link}. Retrying... ({retries-1} attempts left)")
                        time.sleep(5)
                        retries -= 1
                    else:
                        if progress_callback:
                            progress_callback(link, 0, "error")
                        raise
                        
            if progress_callback:
                progress_callback(link, 0, "failed")
            return {"status": "failed", "error": f"Failed after multiple retries for {link}"}
            
        except Exception as e:
            # Clean up process reference if exception occurs
            if link in self._processes:
                del self._processes[link]
            
            if progress_callback:
                progress_callback(link, 0, "error")
            
            return {
                "status": "failed", 
                "error": f"Error downloading {link}: {str(e)}"
            }
    
    def cancel(self, link: str) -> bool:
        """
        Cancel a specific download
        
        Args:
            link: YouTube URL of the download to cancel
            
        Returns:
            bool: True if download was found and cancelled, False otherwise
        """
        if link not in self._processes:
            return False
            
        process_to_terminate = self._processes[link]
        del self._processes[link]
        
        try:
            # On Windows, we need to kill the entire process tree because
            # of how shell=True creates processes
            if os.name == 'nt':
                # Kill process tree
                try:
                    parent = psutil.Process(process_to_terminate.pid)
                    for child in parent.children(recursive=True):
                        try:
                            child.terminate()
                        except (psutil.NoSuchProcess, Exception) as e:
                            print(f"Error terminating child process: {e}")
                    # Kill the parent
                    try:
                        parent.terminate()
                    except (psutil.NoSuchProcess, Exception) as e:
                        print(f"Error terminating parent process: {e}")
                except Exception as e:
                    print(f"Error accessing process: {e}")
            else:
                # On Unix-like systems
                process_to_terminate.terminate()
                
            return True
                
        except Exception as e:
            print(f"Error killing process: {e}")
            return False
            
    def clean_partial_files(self, link: str, video_id: str, directory: str) -> int:
        """
        Clean up partial download files for a specific video
        
        Args:
            link: YouTube URL
            video_id: YouTube video ID
            directory: Directory to check for partial files
            
        Returns:
            int: Number of files cleaned up
        """
        if not os.path.exists(directory) or not os.path.isdir(directory):
            return 0
            
        if not video_id:
            print("Cannot clean directory without video ID")
            return 0
            
        print(f"Checking for files in {directory} related to video ID: {video_id}")
        
        # Common patterns that identify yt-dlp partial files
        partial_patterns = [".part", ".ytdl"]
        
        cleaned_files = 0
        # Check all files in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
                
            # Skip files that don't contain the video ID
            if video_id not in filename:
                continue
                
            # Check if the file is a partial download
            is_partial = any(pattern in filename for pattern in partial_patterns)
            
            # Only remove partial download files
            if is_partial:
                try:
                    # Check if file is currently being accessed
                    try_count = 3
                    while try_count > 0:
                        try:
                            os.remove(file_path)
                            cleaned_files += 1
                            print(f"Removed partial file: {file_path}")
                            break
                        except PermissionError:
                            try_count -= 1
                            if try_count > 0:
                                time.sleep(1)  # Wait a bit before retrying
                            else:
                                print(f"Could not remove file (in use): {file_path}")
                        except FileNotFoundError:
                            # File might have already been deleted
                            break
                        except Exception as e:
                            print(f"Error removing file {file_path}: {e}")
                            break
                except Exception as e:
                    print(f"Error handling file {file_path}: {e}")
        
        if cleaned_files > 0:
            print(f"Cleaned up {cleaned_files} partial download files for video ID {video_id}")
            
        return cleaned_files
