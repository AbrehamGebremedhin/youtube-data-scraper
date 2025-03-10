from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Optional

class BaseDownloader(ABC):
    """Abstract base class for video downloaders"""
    
    @abstractmethod
    def download(self, 
                link: str, 
                download_dir: str, 
                quality: str = '1080',
                video_type: str = 'mp4',
                progress_callback: Optional[Callable] = None) -> dict:
        """
        Download video from the provided link
        
        Args:
            link: URL of the video to download
            download_dir: Directory to save the downloaded video
            quality: Video quality (e.g. '1080', '720')
            video_type: Format of the video (e.g. 'mp4')
            progress_callback: Function to call with progress updates
            
        Returns:
            dict: Status information about the download
        """
        pass
    
    @abstractmethod
    def cancel(self, link: str) -> bool:
        """
        Cancel an ongoing download
        
        Args:
            link: URL of the video being downloaded
            
        Returns:
            bool: True if download was cancelled, False otherwise
        """
        pass
