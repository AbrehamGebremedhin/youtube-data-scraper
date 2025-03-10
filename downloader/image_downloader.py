"""
Image downloader module that implements the BaseDownloader interface
"""
from .base_downloader import BaseDownloader
from typing import Optional, Callable, Dict, Any, List
import os
import time
import uuid
import requests
from PIL import Image as PILImage, UnidentifiedImageError
import io
import random
import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import quote_plus
import google.generativeai as genai
from dotenv import load_dotenv
import traceback
import json

# Import database models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rest.models import Image, get_db_session

# Load environment variables
load_dotenv()

# Configure Gemini API key from environment variable
api_key = os.getenv('GEMINI_API_KEY')
if api_key:
    genai.configure(api_key=api_key)

class ImageDownloader(BaseDownloader):
    """Implements BaseDownloader interface for image downloading"""
    
    def __init__(self):
        # Store for active downloads
        self._active_downloads = {}
        self._download_threads = {}
        self._lock = threading.Lock()
    
    def generate_search_terms(self, category: str, num_terms: int = 5, quality: str = '1080p') -> list:
        """Generate diverse search terms for image search using Gemini if available"""
        if not api_key:
            # Fallback if no API key is available
            base_terms = [
                f"high quality {category}", 
                f"professional {category} photography", 
                f"{category} high resolution", 
                f"beautiful {category} images",
                f"stunning {category} professional photo"
            ]
            return base_terms[:num_terms]
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Convert quality to more descriptive terms for the prompt
            quality_desc = quality.upper() if quality.lower() != '4k' else "4K"
            is_high_quality = any(q in quality.lower() for q in ['1080p', '1440p', '4k', '2160p'])
            quality_terms = "high-quality" if is_high_quality else quality_desc
            
            prompt = f"""Generate {num_terms} diverse search terms for {quality_terms} images related to: "{category}".

            IMPORTANT: This input may contain multiple tags or keywords that together specify exactly what the user wants.
            Your search terms must maintain ALL the important keywords/tags from the input, but add terms that will help find professional, {quality_desc}-quality images.
            
            Focus on creating phrases that would yield beautiful, professional photography results at {quality_desc} resolution.
            The terms should have some variation but should ALL be about the EXACT same subject - don't change the core subject.
            
            Return ONLY the search terms, one per line, no numbering or extra text.
            
            Examples for different domains:
            
            Input: "nature waterfall yosemite"
            Example output:
            professional photography yosemite waterfall {quality_desc}
            high resolution yosemite waterfall landscape
            stunning yosemite waterfall scenery
            yosemite waterfall long exposure photography
            dramatic yosemite waterfall vista
            
            Input: "architecture modern skyscraper"
            Example output:
            modern skyscraper architecture professional photography
            contemporary skyscraper architectural details {quality_desc}
            sleek modern skyscraper against sky
            modern skyscraper glass facade closeup
            dramatic perspective modern skyscraper
            
            Input: "food chocolate dessert"
            Example output:
            gourmet chocolate dessert professional photography
            elegant chocolate dessert plating {quality_desc}
            artistic chocolate dessert presentation
            chocolate dessert with garnish studio lighting
            luxurious chocolate dessert close-up
            """

            response = model.generate_content(prompt)
            search_terms = [term.strip() for term in response.text.strip().split("\n") if term.strip()]
            return search_terms[:num_terms] if search_terms else [f"{quality_desc} {category} images"]
        except Exception as e:
            print(f"Error generating search terms: {e}")
            return [f"{quality} {category} images"]
    
    def download(self, 
                link: str, 
                download_dir: str, 
                quality: str = '1080p',
                video_type: Optional[str] = None, 
                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Download images based on the provided category (link) and quality
        
        Args:
            link: The category to search for (reusing the link parameter)
            download_dir: Directory to save downloaded images
            quality: Quality level expressed as "480p", "720p", "1080p", "4k", etc.
            video_type: Unused but required by interface
            progress_callback: Function to call with progress updates
            
        Returns:
            dict: Status information about the download
        """
        category = link  # Use the link parameter as category
        # Extract num_images from the progress callback's context if available
        num_images = 10
        
        # Parse num_images from any extra info passed in quality string (format: "1080p:20")
        if ":" in quality:
            parts = quality.split(":", 1)
            quality = parts[0]
            try:
                num_images = int(parts[1])
            except (ValueError, IndexError):
                pass
        
        # Convert quality string to pixel dimensions
        if quality == '4k':
            min_width, min_height = 3840, 2160
        elif quality == '1440p':
            min_width, min_height = 2560, 1440
        elif quality == '1080p':
            min_width, min_height = 1920, 1080
        elif quality == '720p':
            min_width, min_height = 1280, 720
        elif quality == '480p':
            min_width, min_height = 854, 480
            # For 480p, be more lenient on width requirement to find more images
            min_width = 640  # Relaxed minimum width
        elif quality == '360p':
            min_width, min_height = 640, 360
        elif quality == '240p':  # Add support for 240p
            min_width, min_height = 426, 240
        else:
            # Use quality value to determine dimensions
            # If quality is like "720", convert to "720p"
            if quality.isdigit():
                quality = f"{quality}p"
                
            # Try to extract numeric value from quality string
            try:
                height = int(''.join(filter(str.isdigit, quality)))
                # Estimate width based on 16:9 aspect ratio
                min_width = int(height * 16 / 9)
                min_height = height
            except ValueError:
                # Default to 1080p if unknown quality string is provided
                min_width, min_height = 1920, 1080
        
        print(f"Quality '{quality}' => minimum dimensions: {min_width}x{min_height}")
        
        # Create download directory
        os.makedirs(download_dir, exist_ok=True)
        
        # Generate a unique ID for this download job
        download_id = str(uuid.uuid4())
        
        with self._lock:
            self._active_downloads[download_id] = {
                "category": category,
                "link": link,
                "status": "searching",
                "progress": 0,
                "quality": quality,
                "num_images": num_images,  # Store the num_images in the download status
                "min_width": min_width,    # Store the min dimensions for later use
                "min_height": min_height,
                "download_dir": download_dir
            }
        
        # Start download in a thread
        download_thread = threading.Thread(
            target=self._download_images_thread,
            args=(download_id, category, num_images, min_width, min_height, download_dir, progress_callback)
        )
        download_thread.daemon = True
        
        with self._lock:
            self._download_threads[download_id] = download_thread
            
        download_thread.start()
        
        return {
            "status": "success",
            "download_id": download_id,
            "message": f"Started downloading {num_images} images for category: {category} at {quality} ({min_width}x{min_height})",
            "download_dir": download_dir
        }

    def cancel(self, link: str) -> bool:
        """
        Cancel an ongoing download
        
        Args:
            link: Category name of the images being downloaded
            
        Returns:
            bool: True if download was cancelled, False otherwise
        """
        found_and_cancelled = False
        download_ids_to_cancel = []
        
        with self._lock:
            for download_id, data in self._active_downloads.items():
                if data.get("link") == link or data.get("category") == link:
                    data["status"] = "cancelled"
                    download_ids_to_cancel.append(download_id)
                    found_and_cancelled = True
        
        return found_and_cancelled
    
    def get_status(self, download_id: str) -> Dict:
        """Get status of a download by ID"""
        with self._lock:
            if download_id in self._active_downloads:
                return self._active_downloads[download_id].copy()  # Return a copy to avoid race conditions
        return {"status": "not_found"}
    
    def get_active_downloads(self) -> Dict:
        """Get all active downloads"""
        with self._lock:
            return {k: v.copy() for k, v in self._active_downloads.items()}
    
    def search_and_download_images(self, 
                                  category: str, 
                                  num_images: int = 10, 
                                  min_width: int = 1920,
                                  min_height: int = 1080,
                                  progress_callback: Optional[Callable] = None,
                                  custom_download_dir: Optional[str] = None) -> Dict:
        """
        Legacy method to maintain compatibility with existing code
        
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
        # Determine quality from dimensions
        quality = '1080p'  # default
        if min_width >= 3840 and min_height >= 2160:
            quality = '4k'
        elif min_width >= 2560 and min_height >= 1440:
            quality = '1440p'
        elif min_width >= 1920 and min_height >= 1080:
            quality = '1080p'
        elif min_width >= 1280 and min_height >= 720:
            quality = '720p'
        elif min_width >= 854 and min_height >= 480:
            quality = '480p'
        
        # Add num_images to quality string
        quality_with_count = f"{quality}:{num_images}"
        
        # Create download directory
        if custom_download_dir and os.path.isdir(custom_download_dir):
            download_dir = custom_download_dir
        else:
            download_dir = os.path.join(os.getcwd(), f"{category}_images")
            
        # Call the main download method
        result = self.download(
            link=category,
            download_dir=download_dir,
            quality=quality_with_count,
            progress_callback=progress_callback
        )
        
        return {
            "success": True,
            "message": result["message"],
            "download_id": result["download_id"],
            "download_dir": download_dir
        }

    def get_image_download_status(self, download_id: str) -> Dict:
        """Get the status of an image download"""
        return self.get_status(download_id)

    def cancel_image_download(self, category: str) -> Dict:
        """Cancel an active image download"""
        cancelled = self.cancel(category)
        if cancelled:
            return {"success": True, "message": f"Download for {category} has been cancelled"}
        return {"success": False, "message": "Download not found or already completed"}

    def get_image_history(self, category: Optional[str] = None, limit: int = 50) -> Dict:
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
            
    def _download_images_thread(self, 
                              download_id: str, 
                              category: str,
                              num_images: int,
                              min_width: int,
                              min_height: int,
                              download_dir: str,
                              progress_callback: Optional[Callable] = None) -> None:
        """Background thread for searching and downloading images"""
        driver = None
        try:
            # Update progress if callback provided
            if progress_callback:
                progress_callback(category, 0, "initializing")
            
            # Get the quality setting from the active_downloads
            quality = '1080p'  # Default
            with self._lock:
                if download_id in self._active_downloads:
                    quality = self._active_downloads[download_id].get("quality", "1080p")
            
            # Generate search terms with quality parameter
            search_terms = self.generate_search_terms(category, num_terms=5, quality=quality)
            print(f"Generated search terms: {search_terms}")
            
            # Update status
            with self._lock:
                if download_id in self._active_downloads:
                    self._active_downloads[download_id]["status"] = "searching"
            
            if progress_callback:
                progress_callback(category, 10, "searching")
            
            # Set environment variables to disable TensorFlow
            os.environ['DISABLE_TENSORRT_OPTIMIZATIONS'] = '1'
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            os.environ['TF_USE_CUDA_XLA'] = '0'
            os.environ['TF_DISABLE_XNNPACK'] = '1'  # Explicitly disable XNNPACK delegate
            
            # Setup Chrome in headless mode with additional options to prevent TensorFlow errors
            chrome_options = Options()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--window-size=1920,1080")
            
            # Disable all GPU and ML features that might trigger TensorFlow
            chrome_options.add_argument("--disable-gpu-compositing")
            chrome_options.add_argument("--disable-software-rasterizer")
            chrome_options.add_argument("--disable-webgl")
            chrome_options.add_argument("--disable-canvas-aa")
            chrome_options.add_argument("--disable-2d-canvas-clip-aa")
            chrome_options.add_argument("--disable-accelerated-2d-canvas")
            chrome_options.add_argument("--disable-webrtc-hw-encoding")
            chrome_options.add_argument("--disable-gl-drawing-for-tests")
            chrome_options.add_argument("--disable-features=UseOzonePlatform")
            chrome_options.add_argument("--disable-features=VizDisplayCompositor")
            chrome_options.add_argument("--disable-features=GlobalMediaControls")
            chrome_options.add_argument("--disable-features=AutofillServerCommunication")
            chrome_options.add_argument("--disable-features=OptimizationHints")
            chrome_options.add_argument("--disable-features=SharedArrayBuffer")
            
            # Disable Machine Learning & AI features that might trigger TensorFlow
            chrome_options.add_argument("--disable-features=AudioServiceOutOfProcess")
            chrome_options.add_argument("--disable-features=BlinkGenPropertyTrees")
            chrome_options.add_argument("--disable-features=TensorWhisperer")
            chrome_options.add_argument("--disable-features=TensorProcessorML")
            chrome_options.add_argument("--disable-features=WebAssemblyTrapHandler")
            chrome_options.add_argument("--disable-features=WebAssemblyStreaming")
            chrome_options.add_argument("--disable-features=WebXR")
            
            # Disable automation flags
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option("useAutomationExtension", False)
            
            # Initialize the driver with multi-level error handling
            success = False
            error_messages = []
            
            # Try first approach - with WebDriver Manager
            try:
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=chrome_options)
                print("Chrome WebDriver initialized successfully")
                success = True
            except Exception as e:
                error_message = f"First approach failed: {str(e)}"
                error_messages.append(error_message)
                print(error_message)
            
            # If first approach failed, try second approach - manually specified path
            if not success:
                try:
                    print("Trying second approach with system Chrome...")
                    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
                    
                    # Try to find Chrome in common locations
                    chrome_paths = [
                        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                        "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
                        "/usr/bin/google-chrome",
                        "/usr/bin/google-chrome-stable"
                    ]
                    
                    chrome_path = None
                    for path in chrome_paths:
                        if os.path.exists(path):
                            chrome_path = path
                            break
                    
                    if chrome_path:
                        chrome_options.binary_location = chrome_path
                    
                    service = Service()
                    driver = webdriver.Chrome(service=service, options=chrome_options)
                    print("Second approach WebDriver initialization successful")
                    success = True
                except Exception as e:
                    error_message = f"Second approach failed: {str(e)}"
                    error_messages.append(error_message)
                    print(error_message)
            
            # Last resort - try a very minimal configuration
            if not success:
                try:
                    print("Trying last resort approach with minimal configuration...")
                    minimal_options = Options()
                    minimal_options.add_argument("--headless")
                    minimal_options.add_argument("--disable-gpu")
                    minimal_options.add_argument("--no-sandbox")
                    minimal_options.add_argument("--disable-dev-shm-usage")
                    
                    service = Service()
                    driver = webdriver.Chrome(service=service, options=minimal_options)
                    print("Last resort WebDriver initialization successful")
                    success = True
                except Exception as e:
                    error_message = f"Last resort approach failed: {str(e)}"
                    error_messages.append(error_message)
                    print(error_message)
                    
            # If all attempts failed, use a fallback method to get images
            if not success:
                # Report error
                error_summary = "\n".join(error_messages)
                print(f"All WebDriver initialization approaches failed:\n{error_summary}")
                
                # Fall back to direct search - downgrade to basic search
                if progress_callback:
                    progress_callback(category, 0, "error")
                    progress_callback(category, 10, "falling back to basic search")
                
                # Use basic HTTP requests to get some images as a last resort
                downloaded_count = self._fallback_image_search(
                    category, 
                    download_dir, 
                    num_images, 
                    min_width, 
                    min_height,
                    progress_callback
                )
                
                # Complete the download process with fallback results
                if downloaded_count > 0:
                    if progress_callback:
                        progress_callback(category, 100, "complete")
                    with self._lock:
                        if download_id in self._active_downloads:
                            self._active_downloads[download_id]["status"] = "complete"
                            self._active_downloads[download_id]["downloaded_count"] = downloaded_count
                    print(f"Fallback download complete: {downloaded_count} images downloaded")
                    return
                else:
                    print("Fallback download failed")
                    if progress_callback:
                        progress_callback(category, 0, "failed")
                    with self._lock:
                        if download_id in self._active_downloads:
                            self._active_downloads[download_id]["status"] = "failed"
                            self._active_downloads[download_id]["error"] = "Browser initialization failed and fallback download returned no images"
                    return
            
            # If we got here, we have a working driver
            downloaded_count = 0
            downloaded_urls = set()
            
            # Check if the download has been cancelled
            def is_cancelled():
                with self._lock:
                    if download_id in self._active_downloads:
                        return self._active_downloads[download_id]["status"] == "cancelled"
                return False
            
            # Try each search term until we get enough images
            for search_term in search_terms:
                # Exit loop if we've downloaded enough images
                if downloaded_count >= num_images:
                    print(f"Reached target of {num_images} images - stopping search")
                    break
                    
                if is_cancelled():
                    print("Download cancelled - stopping search")
                    break
                    
                if progress_callback:
                    progress_callback(category, 10, f"searching for: {search_term}")
                
                try:
                    # Encode search term for URL
                    encoded_search = quote_plus(search_term)
                    # Use Bing Images instead of Google Images (less restrictive)
                    search_url = f"https://www.bing.com/images/search?q={encoded_search}&qft=+filterui:imagesize-large"
                    
                    print(f"Searching Bing Images for: {search_term}")
                    
                    if driver:
                        driver.get(search_url)
                    else:
                        print("Driver not initialized, cannot search")
                        continue
                    
                    # Wait for images to load
                    try:
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".mimg"))
                        )
                    except TimeoutException:
                        print("Timeout waiting for Bing images to load")
                        # Try fallback selector
                        try:
                            WebDriverWait(driver, 5).until(
                                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "img.nofocus"))
                            )
                        except TimeoutException:
                            print("Timeout on fallback selector too")
                            continue

                    # Take a screenshot for debugging
                    debug_dir = os.path.join(download_dir, "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    try:
                        driver.save_screenshot(os.path.join(debug_dir, f"search_{search_term.replace(' ', '_')}.png"))
                    except Exception as ss_err:
                        print(f"Failed to save screenshot: {ss_err}")
                    
                    # Scroll to load more images
                    for _ in range(3):  # Reduced from 5 to 3 to minimize errors
                        if is_cancelled():
                            break
                        try:
                            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                            time.sleep(1)
                        except Exception as scroll_error:
                            print(f"Error during scrolling: {scroll_error}")
                            break
                    
                    # Find all image elements using multiple possible selectors
                    thumbnail_elements = []
                    selectors = [
                        ".mimg", "img.nofocus", ".iusc img", ".imgpt img"
                    ]
                    
                    for selector in selectors:
                        try:
                            elements = driver.find_elements(By.CSS_SELECTOR, selector)
                            if elements:
                                thumbnail_elements.extend(elements)
                                print(f"Found {len(elements)} elements with selector '{selector}'")
                        except Exception as find_error:
                            print(f"Error finding thumbnails with selector '{selector}': {find_error}")
                    
                    print(f"Total thumbnails found: {len(thumbnail_elements)}")
                    
                    if not thumbnail_elements:
                        print(f"No thumbnails found for search term '{search_term}'")
                        continue
                    
                    # Process each thumbnail with improved URL extraction
                    for i, thumbnail in enumerate(thumbnail_elements):
                        # Exit loop if we've downloaded enough images
                        if downloaded_count >= num_images:
                            print(f"Reached target of {num_images} images - stopping thumbnail processing")
                            break
                            
                        if is_cancelled():
                            print("Download cancelled - stopping thumbnail processing")
                            break
                            
                        try:
                            # Get the image URL with improved extraction methods
                            image_url = self._extract_full_image_url(thumbnail, driver)
                            
                            # Skip if we couldn't get a URL or it's already downloaded
                            if not image_url or image_url.startswith("data:") or image_url in downloaded_urls:
                                continue
                                
                            # Create a title from the search term
                            image_title = f"{search_term.replace(' ', '_')}_{i+1}"
                            
                            # Try to get alt text as title
                            try:
                                alt_text = thumbnail.get_attribute("alt")
                                if alt_text and len(alt_text) > 5:  # Ensure it's a meaningful alt text
                                    image_title = alt_text
                            except:
                                pass
                            
                            # Download and validate the image
                            print(f"Attempting to download image from URL: {image_url[:50]}...")
                            download_success = self._download_and_validate_image(
                                image_url, 
                                image_title,
                                category, 
                                download_dir,
                                min_width,
                                min_height
                            )
                            
                            if download_success:
                                downloaded_urls.add(image_url)
                                downloaded_count += 1
                                print(f"Downloaded image {downloaded_count}/{num_images}: {image_title}")
                                
                                if progress_callback:
                                    percent = int((downloaded_count / num_images) * 100)
                                    progress_callback(category, percent, "downloading")
                            
                            # Small delay between downloads
                            time.sleep(random.uniform(0.5, 1.5))
                            
                        except Exception as e:
                            print(f"Error processing thumbnail {i}: {e}")
                            traceback.print_exc()
                            continue  # Skip to the next thumbnail
                except Exception as term_error:
                    print(f"Error processing search term '{search_term}': {term_error}")
                    traceback.print_exc()
                    continue
            
            # If we didn't get enough images, try using the fallback method
            if downloaded_count < num_images and not is_cancelled():
                if progress_callback:
                    progress_callback(category, downloaded_count * 10, "trying alternative sources")
                    
                print(f"Only found {downloaded_count} images with browser. Trying fallback methods...")
                fallback_count = self._fallback_image_search(
                    category, 
                    download_dir, 
                    num_images - downloaded_count,  # Only request the remaining number of images needed
                    min_width, 
                    min_height,
                    progress_callback
                )
                downloaded_count += fallback_count
            
            # Final status update
            if is_cancelled():
                if progress_callback:
                    progress_callback(category, 0, "cancelled")
                with self._lock:
                    if download_id in self._active_downloads:
                        self._active_downloads[download_id]["status"] = "cancelled"
                        
                print(f"Download cancelled after downloading {downloaded_count} images")
            elif downloaded_count > 0:
                if progress_callback:
                    progress_callback(category, 100, "complete")
                with self._lock:
                    if download_id in self._active_downloads:
                        self._active_downloads[download_id]["status"] = "complete"
                        self._active_downloads[download_id]["downloaded_count"] = downloaded_count
                
                files = [f for f in os.listdir(download_dir) if os.path.isfile(os.path.join(download_dir, f))]
                print(f"Download complete: {downloaded_count} images downloaded to {download_dir}")
            else:
                if progress_callback:
                    progress_callback(category, 0, "failed")
                with self._lock:
                    if download_id in self._active_downloads:
                        self._active_downloads[download_id]["status"] = "failed"
                print(f"Failed to download any images for category: {category}")
        
        except Exception as e:
            print(f"Error in download thread: {e}")
            if progress_callback:
                progress_callback(category, 0, "error")
            with self._lock:
                if download_id in self._active_downloads:
                    self._active_downloads[download_id]["status"] = "failed"
                    self._active_downloads[download_id]["error"] = str(e)
        finally:
            # Clean up browser resources
            if driver:
                try:
                    driver.quit()
                    print("Browser closed successfully")
                except Exception as e:
                    print(f"Error closing browser: {e}")
                    
            # Clean up thread references
            with self._lock:
                if download_id in self._download_threads:
                    del self._download_threads[download_id]
                    
    def _fallback_image_search(self, category, download_dir, num_images, min_width, min_height, progress_callback=None):
        """Fallback method to get images when WebDriver initialization fails"""
        print(f"Using fallback image search for category: {category} with min dimensions {min_width}x{min_height}")
        
        try:
            # Try multiple APIs to get images
            downloaded_count = 0
            search_terms = self.generate_search_terms(category, num_terms=3)
            
            # 1. Try Unsplash API first
            for search_term in search_terms:
                # Exit loop if we've downloaded enough images
                if downloaded_count >= num_images:
                    print(f"Fallback: Reached target of {num_images} images - stopping search")
                    break
                    
                if progress_callback:
                    progress_callback(category, 20, f"basic search for: {search_term}")
                
                # Use Unsplash API (no key required for basic search)
                encoded_term = quote_plus(search_term)
                unsplash_url = f"https://source.unsplash.com/1920x1080/?{encoded_term}"
                
                # Try to get a few images
                for i in range(min(5, num_images - downloaded_count)):
                    try:
                        # Add unique parameter to avoid caching
                        unique_url = f"{unsplash_url}&random={uuid.uuid4()}"
                        
                        # Create a title from search term
                        image_title = f"{search_term.replace(' ', '_')}_{i+1}"
                        
                        # Download the image
                        download_success = self._download_and_validate_image(
                            unique_url, 
                            image_title,
                            category,
                            download_dir,
                            min_width,
                            min_height
                        )
                        
                        if download_success:
                            downloaded_count += 1
                            print(f"Downloaded Unsplash image {downloaded_count}/{num_images}")
                            
                            if progress_callback:
                                percent = int((downloaded_count / num_images) * 100)
                                progress_callback(category, percent, "downloading")
                        
                        # Small delay between downloads
                        time.sleep(1.0)
                        
                        # Stop after downloading exactly the number of requested images
                        if downloaded_count >= num_images:
                            break
                            
                    except Exception as e:
                        print(f"Error downloading Unsplash image: {e}")
                        continue
            
            # 2. Try Pixabay API if Unsplash didn't get enough images
            if downloaded_count < num_images:
                pixabay_key = os.getenv('PIXABAY_API_KEY')
                if pixabay_key:
                    print("Using Pixabay API as fallback")
                    for search_term in search_terms:
                        if downloaded_count >= num_images:
                            break
                        
                        try:
                            # Make API request to Pixabay
                            encoded_term = quote_plus(search_term)
                            api_url = f"https://pixabay.com/api/?key={pixabay_key}&q={encoded_term}&image_type=photo&orientation=horizontal&min_width={min_width}&min_height={min_height}&per_page=20"
                            
                            response = requests.get(api_url)
                            if response.status_code == 200:
                                data = response.json()
                                hits = data.get('hits', [])
                                
                                for i, img_data in enumerate(hits):
                                    if downloaded_count >= num_images:
                                        break
                                        
                                    try:
                                        # Get the large image URL
                                        image_url = img_data.get('largeImageURL')
                                        if not image_url:
                                            continue
                                            
                                        # Create title from tags and search term
                                        tags = img_data.get('tags', '').replace(',', ' ')
                                        image_title = f"{search_term}_{tags}_{i+1}"
                                        
                                        # Download the image
                                        download_success = self._download_and_validate_image(
                                            image_url,
                                            image_title,
                                            category,
                                            download_dir,
                                            min_width,
                                            min_height
                                        )
                                        
                                        if download_success:
                                            downloaded_count += 1
                                            print(f"Downloaded Pixabay image {downloaded_count}/{num_images}")
                                            
                                            if progress_callback:
                                                percent = int((downloaded_count / num_images) * 100)
                                                progress_callback(category, percent, "downloading")
                                        
                                        # Small delay between downloads
                                        time.sleep(0.5)
                                        
                                    except Exception as img_err:
                                        print(f"Error with Pixabay image {i}: {img_err}")
                                        continue
                        except Exception as px_err:
                            print(f"Error using Pixabay API: {px_err}")
            
            # 3. Try Pexels API as last resort
            if downloaded_count < num_images:
                pexels_key = os.getenv('PEXELS_API_KEY')
                if pexels_key:
                    print("Using Pexels API as final fallback")
                    for search_term in search_terms:
                        if downloaded_count >= num_images:
                            break
                        
                        try:
                            # Make API request to Pexels
                            encoded_term = quote_plus(search_term)
                            api_url = f"https://api.pexels.com/v1/search?query={encoded_term}&per_page=20"
                            
                            headers = {
                                'Authorization': pexels_key
                            }
                            
                            response = requests.get(api_url, headers=headers)
                            if response.status_code == 200:
                                data = response.json()
                                photos = data.get('photos', [])
                                
                                for i, photo_data in enumerate(photos):
                                    if downloaded_count >= num_images:
                                        break
                                        
                                    try:
                                        # Get the original image URL
                                        src = photo_data.get('src', {})
                                        image_url = src.get('original') or src.get('large')
                                        if not image_url:
                                            continue
                                            
                                        # Create title from photographer and search term
                                        photographer = photo_data.get('photographer', '')
                                        image_title = f"{search_term}_{photographer}_{i+1}"
                                        
                                        # Download the image
                                        download_success = self._download_and_validate_image(
                                            image_url,
                                            image_title,
                                            category,
                                            download_dir,
                                            min_width,
                                            min_height
                                        )
                                        
                                        if download_success:
                                            downloaded_count += 1
                                            print(f"Downloaded Pexels image {downloaded_count}/{num_images}")
                                            
                                            if progress_callback:
                                                percent = int((downloaded_count / num_images) * 100)
                                                progress_callback(category, percent, "downloading")
                                        
                                        # Small delay between downloads
                                        time.sleep(0.5)
                                        
                                    except Exception as photo_err:
                                        print(f"Error with Pexels photo {i}: {photo_err}")
                                        continue
                        except Exception as pexels_err:
                            print(f"Error using Pexels API: {pexels_err}")
            
            return downloaded_count
        except Exception as e:
            print(f"Error in fallback image search: {e}")
            traceback.print_exc()
            return 0

    def _download_and_validate_image(self, 
                                    url: str, 
                                    title: str, 
                                    category: str, 
                                    download_dir: str,
                                    min_width: int,
                                    min_height: int) -> bool:
        """
        Download and validate an image
        
        Args:
            url: Image URL
            title: Image title
            category: Image category
            download_dir: Directory to save the image
            min_width: Minimum width requirement
            min_height: Minimum height requirement
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            # Create a valid filename from title
            safe_title = "".join([c if c.isalnum() or c in " ._-" else "_" for c in title]).strip()
            if not safe_title:
                safe_title = f"{category}_image"
                
            # Add unique identifier to avoid filename conflicts
            filename = f"{safe_title}_{str(uuid.uuid4())[:8]}.jpg"
            file_path = os.path.join(download_dir, filename)
            
            # Download image
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            print(f"Downloading from URL: {url[:50]}...")
            response = requests.get(url, headers=headers, stream=True, timeout=15)
            response.raise_for_status()
            
            # Check if we received an HTML response instead of an image
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                print(f"Received HTML instead of image: {content_type}")
                return False
                
            if not content_type.startswith('image/'):
                print(f"Unexpected content type: {content_type}")
                return False
            
            # Verify it's an image and check dimensions
            try:
                img = PILImage.open(io.BytesIO(response.content))
                width, height = img.size
                
                # Check if image meets minimum size requirements
                if width < min_width or height < min_height:
                    print(f"Image too small: {width}x{height}, required minimum: {min_width}x{min_height}")
                    return False
                    
                print(f"Image validated: {width}x{height}, saving to {os.path.basename(file_path)}")
                
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
            except Exception as img_error:
                print(f"Invalid image data: {img_error}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Request error downloading image: {e}")
            return False
        except Exception as e:
            print(f"Error downloading image: {e}")
            traceback.print_exc()
            return False
            
    def clean_partial_files(self, link: str, video_id: str, directory: str) -> None:
        """
        Clean up partial files - not applicable for image downloader 
        but required by interface
        """
        # Not needed for image downloads, but implementing to satisfy interface
        pass
        
    def get_status(self, download_id: str) -> Dict:
        """Get status of a download by ID"""
        with self._lock:
            if download_id in self._active_downloads:
                return self._active_downloads[download_id]
        return {"status": "not_found"}
        
    def _extract_full_image_url(self, thumbnail, driver):
        """Extract full-size image URL from a thumbnail element using multiple methods"""
        image_url = None
        
        # Method 1: Direct src attribute (usually small thumbnail)
        try:
            image_url = thumbnail.get_attribute("src")
        except Exception:
            pass
        
        # Method 2: Check if it's a Bing thumbnail and try to get the original URL
        if image_url and 'th.bing.com/th/id/' in image_url:
            try:
                # First check if we can get the parent with metadata
                parent = None
                try:
                    parent = thumbnail.find_element(By.XPATH, "./..")
                except:
                    try:
                        parent = driver.execute_script("return arguments[0].closest('.iusc')", thumbnail)
                    except:
                        pass
                
                if parent:
                    # Try to get m attribute which contains metadata with original URL
                    try:
                        data_str = parent.get_attribute("m")
                        if data_str:
                            try:
                                data = json.loads(data_str)
                                if "murl" in data:
                                    return data["murl"]  # This is the full-size image URL
                            except:
                                pass
                    except:
                        pass
                    
                # Try looking for mmComponent attribute on parent or nearby elements
                try:
                    mm_data = driver.execute_script("""
                        var element = arguments[0];
                        var parent = element.parentElement;
                        for (var i = 0; i < 4; i++) {
                            if (parent.hasAttribute('mmComponent')) {
                                return parent.getAttribute('mmComponent');
                            }
                            parent = parent.parentElement;
                            if (!parent) break;
                        }
                        return null;
                    """, thumbnail)
                    
                    if mm_data:
                        try:
                            data = json.loads(mm_data)
                            if "purl" in data:
                                return data["purl"]
                        except:
                            pass
                except:
                    pass
            except Exception as e:
                print(f"Error extracting full URL: {e}")
        
        # Method 3: If thumbnail is in a link, get the href
        try:
            parent_a = driver.execute_script(
                "return arguments[0].closest('a')", thumbnail
            )
            if parent_a:
                href = parent_a.get_attribute("href")
                if href and "bing.com" in href and "images/search" in href:
                    # Bing's image search links have a parameter with the original URL
                    if "imgurl=" in href:
                        url_part = href.split("imgurl=")[1].split("&")[0]
                        from urllib.parse import unquote
                        return unquote(url_part)
        except:
            pass
        
        # Method 4: Check for data-src attribute
        try:
            data_src = thumbnail.get_attribute("data-src")
            if data_src and not data_src.startswith("data:"):
                image_url = data_src
        except:
            pass
            
        # If we've only found a thumbnail URL but it's a Bing thumbnail URL,
        # try fixing it to get a larger version
        if image_url and 'th.bing.com/th/id/' in image_url:
            # Try to modify the URL to get a larger version
            try:
                # Some Bing image URLs can be converted to larger versions
                if '&w=' in image_url and '&h=' in image_url:
                    # Replace dimensions with larger ones
                    image_url = image_url.replace('&w=', '&w=1200').replace('&h=', '&h=800')
            except:
                pass
                
        return image_url
