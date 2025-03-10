import os
import time
import uuid
import requests
from PIL import Image as PILImage, UnidentifiedImageError
import io
import random
import json
import google.generativeai as genai
from urllib.parse import quote_plus
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service  # Add Service import
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv
import threading
from typing import Dict, List, Optional, Callable

# Import database models
from rest.models import Image, get_db_session

# Load environment variables
load_dotenv()

# Configure Gemini API key from environment variable
api_key = os.getenv('GEMINI_API_KEY')
if api_key:
    genai.configure(api_key=api_key)

# Store for active downloads and progress tracking
active_downloads = {}
download_progress = {}
progress_callbacks = {}
download_lock = threading.Lock()

def generate_search_terms(category: str, num_terms: int = 5) -> List[str]:
    """Generate diverse search terms for image search using Gemini if available"""
    if not api_key:
        # Fallback if no API key is available
        return [f"high quality {category} wallpaper", 
                f"professional {category} photography", 
                f"beautiful {category} 1080p", 
                f"stunning {category} images",
                f"{category} high resolution"]
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""Generate {num_terms} diverse, specific search terms for high-quality 1080p images related to {category}.
        Focus on professional photography terms that would yield beautiful results.
        The terms should be varied and specific enough to return diverse results.
        Return ONLY the search terms, one per line, no numbering or extra text.
        Example output format:
        stunning sunset over mountains professional photography
        close-up macro photography of flowers
        aerial view cityscape high resolution
        """

        response = model.generate_content(prompt)
        search_terms = [term.strip() for term in response.text.strip().split("\n") if term.strip()]
        return search_terms[:num_terms] if search_terms else [f"high quality {category} images"]
    except Exception as e:
        print(f"Error generating search terms: {e}")
        return [f"high quality {category} images"]

def search_and_download_images(category: str, 
                              num_images: int = 10, 
                              min_width: int = 1920,
                              min_height: int = 1080,
                              progress_callback: Optional[Callable] = None,
                              custom_download_dir: Optional[str] = None) -> Dict:
    """
    Search for images using Google Images and download them
    
    Args:
        category: Search category
        num_images: Number of images to download
        min_width: Minimum image width
        min_height: Minimum image height
        progress_callback: Function to call with progress updates
        custom_download_dir: Optional custom download directory
        
    Returns:
        Dict: Result information with success/failure details
    """
    # Register progress callback if provided
    if progress_callback:
        with download_lock:
            progress_callbacks[category] = progress_callback
            progress_callback(category, 0, "initializing")
    
    # Generate search terms
    search_terms = generate_search_terms(category, num_terms=5)
    print(f"Generated search terms: {search_terms}")
    
    # Create download directory
    if custom_download_dir and os.path.isdir(custom_download_dir):
        download_dir = custom_download_dir
    else:
        download_dir = os.path.join(os.getcwd(), f"{category}_images")
    
    os.makedirs(download_dir, exist_ok=True)
    
    # Start tracking this download
    download_id = str(uuid.uuid4())
    with download_lock:
        active_downloads[download_id] = {
            "category": category,
            "status": "searching",
            "progress": 0,
            "download_dir": download_dir
        }
    
    # Create a new thread for the download process
    download_thread = threading.Thread(
        target=_download_images_thread,
        args=(download_id, category, search_terms, num_images, min_width, min_height, download_dir)
    )
    download_thread.daemon = True
    download_thread.start()
    
    return {
        "success": True,
        "message": f"Started searching for {num_images} images related to '{category}'",
        "download_id": download_id,
        "download_dir": download_dir
    }

def _download_images_thread(download_id: str, 
                           category: str,
                           search_terms: List[str],
                           num_images: int,
                           min_width: int,
                           min_height: int,
                           download_dir: str) -> None:
    """Background thread for searching and downloading images"""
    try:
        # Setup Chrome in headless mode
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Initialize the driver - FIX: Use Service class for ChromeDriverManager
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        downloaded_count = 0
        downloaded_urls = set()
        
        # Try each search term until we get enough images
        for search_term in search_terms:
            if downloaded_count >= num_images:
                break
                
            _update_progress(category, 10, f"searching for: {search_term}")
            
            # Encode search term for URL
            encoded_search = quote_plus(search_term)
            # Use Google Images search with size filter for large images
            search_url = f"https://www.google.com/search?q={encoded_search}&tbm=isch&tbs=isz:l"
            
            driver.get(search_url)
            
            # Wait for images to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "img.rg_i"))
            )
            
            # Scroll to load more images
            for _ in range(5):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
            
            # Find all image elements
            thumbnail_elements = driver.find_elements(By.CSS_SELECTOR, "img.rg_i")
            print(f"Found {len(thumbnail_elements)} thumbnail elements")
            
            # Process each thumbnail
            for i, thumbnail in enumerate(thumbnail_elements):
                if downloaded_count >= num_images:
                    break
                    
                try:
                    # Click on thumbnail to get full image
                    driver.execute_script("arguments[0].click();", thumbnail)
                    
                    # Wait for the image details to appear
                    WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "img.r48jcc, img.sFlh5c"))
                    )
                    
                    # Get the large image element
                    large_image = driver.find_element(By.CSS_SELECTOR, "img.r48jcc, img.sFlh5c")
                    image_url = large_image.get_attribute("src")
                    
                    # Skip if already downloaded or if it's a data URL
                    if not image_url or image_url.startswith("data:") or image_url in downloaded_urls:
                        continue
                    
                    image_title = driver.find_element(By.CSS_SELECTOR, "h2.kCmff, h3.NB0Rle").text
                    if not image_title:
                        image_title = f"{category}_image_{downloaded_count + 1}"
                    
                    # Download and validate the image
                    download_success = _download_and_validate_image(
                        image_url, 
                        image_title,
                        category, 
                        download_dir,
                        min_width,
                        min_height,
                        downloaded_count,
                        num_images
                    )
                    
                    if download_success:
                        downloaded_urls.add(image_url)
                        downloaded_count += 1
                        print(f"Downloaded image {downloaded_count}/{num_images}: {image_title}")
                        _update_progress(category, (downloaded_count / num_images) * 100, "downloading")
                    
                    # Small delay between downloads
                    time.sleep(random.uniform(1.0, 2.0))
                    
                except Exception as e:
                    print(f"Error processing thumbnail {i}: {e}")
        
        driver.quit()
        
        # Final status update
        if downloaded_count > 0:
            _update_progress(category, 100, "complete")
            with download_lock:
                if download_id in active_downloads:
                    active_downloads[download_id]["status"] = "complete"
                    active_downloads[download_id]["downloaded_count"] = downloaded_count
            
            files = [f for f in os.listdir(download_dir) if os.path.isfile(os.path.join(download_dir, f))]
            print(f"Download complete: {downloaded_count} images downloaded to {download_dir}")
        else:
            _update_progress(category, 0, "failed")
            with download_lock:
                if download_id in active_downloads:
                    active_downloads[download_id]["status"] = "failed"
            print(f"Failed to download any images for category: {category}")
    
    except Exception as e:
        print(f"Error in download thread: {e}")
        _update_progress(category, 0, "error")
        with download_lock:
            if download_id in active_downloads:
                active_downloads[download_id]["status"] = "failed"
                active_downloads[download_id]["error"] = str(e)

def _download_and_validate_image(url: str, 
                                title: str, 
                                category: str, 
                                download_dir: str,
                                min_width: int,
                                min_height: int,
                                current_count: int,
                                total_count: int) -> bool:
    """
    Download and validate an image
    
    Args:
        url: Image URL
        title: Image title
        category: Image category
        download_dir: Directory to save the image
        min_width: Minimum width requirement
        min_height: Minimum height requirement
        current_count: Current download count
        total_count: Total target count
        
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        # Create a valid filename from title
        safe_title = "".join([c if c.isalnum() or c in " ._-" else "_" for c in title]).strip()
        if not safe_title:
            safe_title = f"{category}_image_{current_count + 1}"
            
        # Add unique identifier to avoid filename conflicts
        filename = f"{safe_title}_{str(uuid.uuid4())[:8]}.jpg"
        file_path = os.path.join(download_dir, filename)
        
        # Download image
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        response.raise_for_status()
        
        # Verify it's an image and check dimensions
        img = PILImage.open(io.BytesIO(response.content))
        width, height = img.size
        
        # Check if image meets minimum size requirements
        if width < min_width or height < min_height:
            print(f"Image too small: {width}x{height}, required: {min_width}x{min_height}")
            return False
        
        # Save the image
        with open(file_path, 'wb') as f:
            f.write(response.content)
            
        # Record in database
        try:
            session = get_db_session()
            
            # Check if this URL is already in the database
            existing_image = session.query(Image).filter(Image.url == url).first()
            
            if existing_image:
                # Update existing record
                existing_image.title = title
                existing_image.file_path = file_path
                existing_image.status = "success"
                existing_image.width = width
                existing_image.height = height
                existing_image.file_size = os.path.getsize(file_path)
            else:
                # Create new record
                new_image = Image(
                    url=url,
                    title=title,
                    category=category,
                    file_path=file_path,
                    status="success",
                    width=width,
                    height=height,
                    file_size=os.path.getsize(file_path),
                    thumbnail=url  # Using original URL as thumbnail
                )
                session.add(new_image)
                
            session.commit()
            session.close()
        except Exception as e:
            print(f"Error recording image in database: {e}")
            if 'session' in locals():
                session.close()
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Request error downloading image: {e}")
        return False
    except UnidentifiedImageError:
        print("Error: Cannot identify image file")
        return False
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False

def _update_progress(category: str, percent: float, status: str = "downloading") -> None:
    """Update progress via callback if registered"""
    with download_lock:
        if category in progress_callbacks:
            try:
                callback = progress_callbacks[category]
                callback(category, percent, status)
            except Exception as e:
                print(f"Error in progress callback: {e}")

def get_image_download_status(download_id: str) -> Dict:
    """Get the status of an image download"""
    with download_lock:
        if download_id in active_downloads:
            return active_downloads[download_id]
        return {"status": "not_found", "message": "Download not found"}

def cancel_image_download(download_id: str) -> Dict:
    """Cancel an active image download"""
    with download_lock:
        if download_id in active_downloads:
            # Mark as cancelled
            active_downloads[download_id]["status"] = "cancelled"
            category = active_downloads[download_id]["category"]
            
            # Update progress callback with cancelled status
            if category in progress_callbacks:
                try:
                    callback = progress_callbacks[category]
                    callback(category, 0, "cancelled")
                except Exception as e:
                    print(f"Error in progress callback: {e}")
            
            return {"success": True, "message": f"Download for {category} has been cancelled"}
        return {"success": False, "message": "Download not found or already completed"}

def get_image_history(category: Optional[str] = None, limit: int = 50) -> Dict:
    """Get image download history"""
    try:
        session = get_db_session()
        query = session.query(Image)
        
        # Apply filter if category provided
        if category:
            query = query.filter(Image.category == category)
        
        # Get most recent downloads first
        query = query.order_by(Image.download_date.desc()).limit(limit)
        
        images = []
        for img in query.all():
            images.append({
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
        return {"success": True, "count": len(images), "images": images}
    
    except Exception as e:
        if 'session' in locals():
            session.close()
        return {"success": False, "message": f"Error retrieving image history: {e}"}
