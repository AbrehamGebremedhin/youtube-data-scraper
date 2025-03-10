import subprocess
import os
import time
import google.generativeai as genai
import yt_dlp
from dotenv import load_dotenv
from downloader.Downloader import Downloader
import uuid
from rest.models import get_db_session, CandidateVideo, Video
import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def initialize_video_downloader():
    """Initialize the video downloader with default settings."""
    from downloader.Downloader import Downloader
    downloader = Downloader('youtube_scraper_downloader')
    downloader.reset_to_default_downloader()
    print("Video downloader initialized for youtube_scraper")
    return downloader

def reinitialize_video_downloader():
    """Reinitialize the video downloader with default settings."""
    # Same implementation as initialize_video_downloader for now
    return initialize_video_downloader()

# Load environment variables
load_dotenv()

# Configure Gemini API key from environment variable
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

# Configure YouTube API key from environment variable
youtube_api_key = os.getenv('YOUTUBE_API_KEY')
if not youtube_api_key:
    raise ValueError("Please set YOUTUBE_API_KEY in your .env file")

# Initialize YouTube API client
youtube = build('youtube', 'v3', developerKey=youtube_api_key)

genai.configure(api_key=api_key)

# Store for candidate videos
candidate_videos_store = {}

def generate_candidate_id():
    """Generate a unique ID for a candidate video"""
    return str(uuid.uuid4())

def download_video(link, category, video_type='mp4', quality='1080', progress_callback=None, video_info=None):
    """
    Downloads a YouTube video to a category-specific directory.
    
    Args:
        link (str): YouTube video URL
        category (str): Category name for directory organization
        video_type (str): Video format (e.g., mp4, webm)
        quality (str): Video quality (e.g., 1080, 720, best)
        progress_callback (callable): Function to call with progress updates
        video_info (dict): Optional video metadata
    """
    # Use the singleton downloader instance
    downloader = Downloader()
    # Always use MP4 format
    return downloader.download_video(link, category, 'mp4', quality, progress_callback, video_info)

def get_diverse_search_terms(category, num_terms, min_duration=300, max_duration=600):
    """Generates diverse search terms using Gemini."""
    # Convert seconds to minutes for better readability in the prompt
    min_minutes = min_duration // 60
    max_minutes = max_duration // 60
    
    model = genai.GenerativeModel('gemini-2.0-flash')  # Updated model name
    prompt = f"""You are a helpful assistant designed to generate diverse search queries for YouTube video downloads.
    The videos should be professionally produced and of high quality.
    
    The user wants to download {num_terms} videos related to: "{category}".
    IMPORTANT: This input may contain multiple tags or keywords (like "nature waterfall yosemite" means videos specifically about Yosemite waterfalls).
    
    The videos must be {min_minutes}-{max_minutes} minutes long (not shorter, not longer).
    
    Please provide {num_terms} search terms that maintain ALL the important keywords/tags from the input,
    but vary slightly to help find a variety of relevant content. Do NOT change the core subject.
    
    Return ONLY the search terms, one per line, no numbering or extra text.
    
    Examples for different domains:
    
    Input: "nature waterfall yosemite"
    Example output:
    yosemite waterfall footage scenic views
    yosemite national park waterfall documentary
    powerful yosemite waterfalls in spring
    yosemite waterfall hiking trails
    yosemite waterfall time lapse footage
    
    Input: "cooking pasta carbonara authentic"
    Example output:
    authentic italian carbonara pasta recipe
    traditional carbonara pasta cooking method
    homemade pasta carbonara authentic preparation
    authentic carbonara pasta from scratch
    how to cook authentic pasta carbonara roman style
    """

    response = model.generate_content(prompt)
    search_terms = [term.strip() for term in response.text.strip().split("\n") if term.strip()]
    return search_terms[:num_terms]

def search_videos_with_api(query, max_results=5, min_duration=300, max_duration=600):
    """
    Search for videos using YouTube Data API with duration filters
    
    Args:
        query (str): The search query
        max_results (int): Maximum number of results to return
        min_duration (int): Minimum duration in seconds
        max_duration (int): Maximum duration in seconds
        
    Returns:
        list: List of video information dictionaries
    """
    try:
        # Convert duration to YouTube API format (PT5M30S for 5 minutes 30 seconds)
        min_duration_str = f'PT{min_duration}S'
        max_duration_str = f'PT{max_duration}S'
        
        # Perform the search with specific filters
        search_response = youtube.search().list(
            q=query,
            part='id,snippet',
            maxResults=max_results,
            type='video',
            videoDefinition='high',  # Only HD videos
            videoDuration='medium',  # Medium duration (4-20 minutes)
            videoEmbeddable='true',  # Only embeddable videos
            videoLicense='youtube',  # Standard YouTube license
            videoType='any'          # Any type of videos
        ).execute()
        
        video_ids = [item['id']['videoId'] for item in search_response['items']]
        
        if not video_ids:
            return []
            
        # Get detailed video information including duration and technical details
        video_response = youtube.videos().list(
            id=','.join(video_ids),
            part='contentDetails,statistics,snippet,status,player'
        ).execute()
        
        videos = []
        for item in video_response['items']:
            # Parse the duration string (format: PT5M30S for 5 minutes 30 seconds)
            duration_str = item['contentDetails']['duration']
            # Remove PT prefix
            duration_str = duration_str[2:]
            
            # Calculate duration in seconds
            duration_seconds = 0
            if 'H' in duration_str:
                h, duration_str = duration_str.split('H')
                duration_seconds += int(h) * 3600
            if 'M' in duration_str:
                m, duration_str = duration_str.split('M')
                duration_seconds += int(m) * 60
            if 'S' in duration_str:
                s = duration_str.replace('S', '')
                if s:  # Check if s is not empty
                    duration_seconds += int(s)
            
            # Skip videos that don't match our duration requirements
            if not (min_duration <= duration_seconds <= max_duration):
                continue
                
            # Check if video definition is HD (minimum 720p)
            if item['contentDetails'].get('definition', '') != 'hd':
                continue
                
            # Get video dimensions to ensure it's 1080p
            # Note: YouTube API doesn't directly provide resolution,
            # we'll check this later with yt-dlp for more accuracy
                
            videos.append({
                'video_id': item['id'],
                'url': f'https://www.youtube.com/watch?v={item["id"]}',
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'duration': duration_seconds,
                'thumbnail': item['snippet'].get('thumbnails', {}).get('high', {}).get('url', ''),
                'view_count': int(item['statistics'].get('viewCount', 0)),
                'like_count': int(item['statistics'].get('likeCount', 0)),
                'dislike_count': int(item['statistics'].get('dislikeCount', 0)) if 'dislikeCount' in item['statistics'] else 0,
                'channel_title': item['snippet']['channelTitle'],
                'published_at': item['snippet']['publishedAt'],
                'category_id': item['snippet'].get('categoryId', ''),
                'definition': item['contentDetails']['definition']  # 'sd' or 'hd'
            })
            
        return videos
        
    except HttpError as e:
        print(f"YouTube API error: {e}")
        return []
    except Exception as e:
        print(f"Error searching videos: {e}")
        return []

def validate_video_content(video_url, category, min_duration=300, max_duration=600, min_video_bitrate=1000, min_audio_bitrate=128):
    """Validates video content using YouTube API and checks technical requirements with yt-dlp."""
    try:
        # Extract video ID from URL
        video_id = None
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
            
        if not video_id:
            return False, "Could not extract video ID from URL"
            
        # Fetch video details using YouTube API
        try:
            video_response = youtube.videos().list(
                id=video_id,
                part='contentDetails,statistics,snippet,status'
            ).execute()
            
            if not video_response.get('items'):
                return False, "Video not found or not available"
                
            video_data = video_response['items'][0]
            
            # Parse duration
            duration_str = video_data['contentDetails']['duration']
            duration_str = duration_str[2:]  # Remove PT prefix
            
            # Calculate duration in seconds
            duration_seconds = 0
            if 'H' in duration_str:
                h, duration_str = duration_str.split('H')
                duration_seconds += int(h) * 3600
            if 'M' in duration_str:
                m, duration_str = duration_str.split('M')
                duration_seconds += int(m) * 60
            if 'S' in duration_str:
                s = duration_str.replace('S', '')
                if s:  # Check if s is not empty
                    duration_seconds += int(s)
                    
            # Check duration requirements
            if not (min_duration <= duration_seconds <= max_duration):
                return False, f"Video duration ({duration_seconds} seconds) not within acceptable range ({min_duration}-{max_duration} seconds)"
                
            # Check content appropriateness from YouTube metadata
            content_rating = video_data['contentDetails'].get('contentRating', {})
            if content_rating and ('ytAgeRestricted' in content_rating or content_rating.get('ytRating') == 'ytAgeRestricted'):
                return False, "Video is age-restricted and not suitable"
                
            # Validate category relevance
            video_title = video_data['snippet']['title']
            video_description = video_data['snippet']['description']
            channel_title = video_data['snippet']['channelTitle']
            video_category_id = video_data['snippet'].get('categoryId', '')
            
            # Use yt-dlp to check video resolution and bitrate
            # This is still necessary as YouTube API doesn't provide these details
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'format': 'bestvideo[height>=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(video_url, download=False)
                height = info_dict.get('height', 0) or 0  # Handle None height
                
                # Check resolution (exactly 1080p)
                if height != 1080:
                    return False, f"Video resolution ({height}p) is not 1080p"
                    
                # Extract video and audio quality information
                formats = info_dict.get('formats', [])
                best_video_bitrate = 0
                best_audio_bitrate = 0
                
                for format_info in formats:
                    format_height = format_info.get('height', 0) or 0
                    if format_height >= 1080:
                        vbr = format_info.get('vbr', 0) or 0
                        tbr = format_info.get('tbr', 0) or 0
                        video_bitrate = max(vbr, tbr)
                        
                        if video_bitrate > best_video_bitrate:
                            best_video_bitrate = video_bitrate
                    
                    acodec = format_info.get('acodec', 'none')
                    vcodec = format_info.get('vcodec', '')
                    if acodec != 'none' and vcodec == 'none':
                        abr = format_info.get('abr', 0) or 0
                        if abr > best_audio_bitrate:
                            best_audio_bitrate = abr
                
                # Check bitrate requirements
                if min_video_bitrate > 0 and best_video_bitrate < min_video_bitrate:
                    return False, f"Video bitrate ({best_video_bitrate} kbps) below required {min_video_bitrate} kbps"
                
                if min_audio_bitrate > 0 and best_audio_bitrate < min_audio_bitrate:
                    return False, f"Audio bitrate ({best_audio_bitrate} kbps) below required {min_audio_bitrate} kbps"
            
            # Check for gaming content
            gaming_keywords = ['gameplay', 'gaming', 'game', 'playthrough', 'lets play', 'walkthrough', 'speedrun']
            gaming_categories = ['20']  # YouTube category ID for Gaming
            
            title_lower = video_title.lower()
            desc_lower = video_description.lower()
            
            if video_category_id in gaming_categories:
                return False, "Gaming content is not allowed (gaming category)"
                
            for keyword in gaming_keywords:
                if keyword in title_lower or keyword in desc_lower or keyword in channel_title.lower():
                    return False, f"Gaming content detected: '{keyword}'"
            
            # Check for GoPro/action camera content
            action_cam_keywords = ['gopro', 'action camera', 'first person', 'pov', 'point of view']
            for keyword in action_cam_keywords:
                if keyword in title_lower or keyword in desc_lower:
                    return False, f"Action camera content detected: '{keyword}'"
            
            return True, f"Video meets requirements. Duration: {duration_seconds}s, Resolution: {height}p, Video bitrate: {best_video_bitrate}kbps, Audio bitrate: {best_audio_bitrate}kbps"
            
        except HttpError as e:
            return False, f"YouTube API error: {str(e)}"
            
    except Exception as e:
        return False, f"Error validating video: {e}"

def find_candidate_videos(category, num_videos=20, min_duration=300, max_duration=600):
    """Finds YouTube videos matching criteria but doesn't download them."""
    try:
        # Clear old candidates that were created more than 24 hours ago
        clear_old_candidates()
        
        search_terms_queue = get_diverse_search_terms(category, num_videos, min_duration, max_duration)
        
        candidate_videos = []  # List to hold video details that pass validation
        downloaded_ids = set()  # Track validated video IDs

        max_attempts = num_videos * 2
        attempts = 0
        while search_terms_queue and len(candidate_videos) < num_videos and attempts < max_attempts:
            search_term = search_terms_queue.pop(0)
            candidate_count_before = len(candidate_videos)
            try:
                print(f"\nSearching for: {search_term}")
                # Use YouTube Data API for search instead of yt-dlp
                found_videos = search_videos_with_api(
                    search_term, 
                    max_results=5,
                    min_duration=min_duration,
                    max_duration=max_duration
                )
                
                for video in found_videos:
                    if len(candidate_videos) >= num_videos:
                        break
                    
                    video_id = video.get('video_id')
                    if video_id in downloaded_ids:
                        print(f"Skipping duplicate video: {video_id}")
                        continue
                        
                    try:
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        is_valid, validation_message = validate_video_content(video_url, category, min_duration, max_duration)
                        print(f"Validating {video_url}: {validation_message}")
                        
                        if is_valid:
                            # Generate a unique candidate ID
                            candidate_id = generate_candidate_id()
                            
                            # Get additional details from the API response
                            video_info = {
                                'candidate_id': candidate_id,
                                'video_id': video_id,
                                'url': video_url,
                                'title': video.get('title', ''),
                                'description': video.get('description', '')[:500] + ('...' if video.get('description', '') and len(video.get('description', '')) > 500 else ''),
                                'duration': video.get('duration', 0),
                                'height': 1080,  # Already validated to be 1080p
                                'vbr': 0,  # Will be filled by yt-dlp validation
                                'abr': 0,  # Will be filled by yt-dlp validation
                                'thumbnail': video.get('thumbnail', ''),
                                'view_count': video.get('view_count', 0),
                                'validation_message': validation_message,
                                'category': category,
                                'channel_title': video.get('channel_title', ''),
                                'published_at': video.get('published_at', '')
                            }
                            
                            # Store the candidate video in our store
                            store_candidate_video(video_info)
                            
                            candidate_videos.append(video_info)
                            downloaded_ids.add(video_id)
                            print(f"Added candidate video: {video_url}")
                            break  # Stop processing further videos for this search term
                        else:
                            print(f"Video {video_url} failed validation: {validation_message}")
                    except Exception as e:
                        print(f"Error processing video: {e}")
                        continue
                
                time.sleep(2)
                # If the required number of videos is reached, stop searching
                if len(candidate_videos) >= num_videos:
                    break
                # If no video was added, generate extra search terms to try to fulfill the requested count
                if len(candidate_videos) == candidate_count_before:
                    needed = num_videos - len(candidate_videos)
                    extra_terms = get_diverse_search_terms(category, needed, min_duration, max_duration)
                    if extra_terms:
                        print(f"No videos found for '{search_term}', adding extra search terms: {', '.join(extra_terms)}")
                        search_terms_queue.extend(extra_terms)
                attempts += 1

            except Exception as e:
                print(f"Error processing search term '{search_term}': {e}")
                time.sleep(2)
                continue

        return {
            "success": len(candidate_videos) > 0,
            "message": f"Found {len(candidate_videos)} candidate videos",
            "candidates": candidate_videos
        }
        
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(error_message)
        return {
            "success": False,
            "message": error_message,
            "candidates": []
        }

def store_candidate_video(video_info):
    """Store a candidate video in the database and memory cache"""
    candidate_id = video_info['candidate_id']
    
    # Store in memory cache
    candidate_videos_store[candidate_id] = video_info
    
    # Store in database for persistence
    try:
        session = get_db_session()
        
        # Check if this candidate already exists
        existing = session.query(CandidateVideo).filter(
            CandidateVideo.video_id == video_info['video_id']
        ).first()
        
        if not existing:
            # Create new record
            candidate = CandidateVideo(
                video_id=video_info['video_id'],
                url=video_info['url'],
                title=video_info['title'],
                category=video_info['category'],
                description=video_info['description'],
                duration=video_info['duration'],
                height=video_info['height'],
                thumbnail=video_info['thumbnail'],
                vbr=video_info['vbr'],
                abr=video_info['abr'],
                view_count=video_info.get('view_count', 0),
                validation_message=video_info['validation_message'],
                status="pending"
            )
            session.add(candidate)
            session.commit()
            
        session.close()
    except Exception as e:
        print(f"Error storing candidate video: {str(e)}")
        if 'session' in locals():
            session.close()

def get_candidate_video(candidate_id):
    """Get a candidate video by its ID"""
    # First check memory cache
    if candidate_id in candidate_videos_store:
        return candidate_videos_store[candidate_id]
    
    # If not in memory, try database
    try:
        session = get_db_session()
        candidate = session.query(CandidateVideo).filter(CandidateVideo.id == candidate_id).first()
        
        if candidate:
            # Convert DB model to dictionary
            video_info = {
                'candidate_id': str(candidate.id),
                'video_id': candidate.video_id,
                'url': candidate.url,
                'title': candidate.title,
                'description': candidate.description,
                'duration': candidate.duration,
                'height': candidate.height,
                'vbr': candidate.vbr,
                'abr': candidate.abr,
                'thumbnail': candidate.thumbnail,
                'view_count': candidate.view_count,
                'validation_message': candidate.validation_message,
                'category': candidate.category
            }
            
            # Add to memory cache
            candidate_videos_store[candidate_id] = video_info
            return video_info
            
        session.close()
        return None
    except Exception as e:
        print(f"Error retrieving candidate video: {str(e)}")
        if 'session' in locals():
            session.close()
        return None

def clear_old_candidates(days=30):
    """Clear candidates that are older than the specified number of days"""
    try:
        session = get_db_session()
        older_than_date = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        
        # Get count before deletion for logging
        count = session.query(CandidateVideo).filter(CandidateVideo.creation_date < older_than_date).count()
        
        # Only delete candidates that have already been downloaded or rejected
        session.query(CandidateVideo).filter(
            CandidateVideo.creation_date < older_than_date,
            CandidateVideo.status.in_(["downloaded", "rejected"])
        ).delete()
        
        session.commit()
        session.close()
        print(f"Cleared {count} old candidates that were downloaded or rejected")
    except Exception as e:
        print(f"Error clearing old candidates: {str(e)}")
        if 'session' in locals():
            session.close()

def download_single_video(candidate_id, progress_callback=None, download_path=None):
    """Downloads a single video by its candidate ID."""
    video_info = get_candidate_video(candidate_id)
    
    if not video_info:
        return {
            "success": False,
            "message": f"Video with ID {candidate_id} not found",
            "downloaded_count": 0,
            "files": []
        }
    
    category = video_info['category']
    video_url = video_info['url']
    
    try:
        # Use custom download path if provided, otherwise default to category folder
        if download_path and os.path.isdir(download_path):
            download_dir = download_path
        else:
            download_dir = os.path.join(os.getcwd(), f"{category}_videos")
        
        os.makedirs(download_dir, exist_ok=True)
        
        # Get the downloader singleton instance
        downloader = Downloader()
        
        print(f"\nStarting download of approved video: {video_url} to {download_dir}")
        
        # Update status in database
        try:
            session = get_db_session()
            candidate = session.query(CandidateVideo).filter(
                CandidateVideo.id == candidate_id
            ).first()
            
            if candidate:
                candidate.status = "approved"
                session.commit()
            
            session.close()
        except Exception as e:
            print(f"Error updating candidate status: {str(e)}")
            if 'session' in locals():
                session.close()
        
        # Submit download to thread pool with custom download directory
        future = downloader.download_video(
            video_url, 
            category, 
            'mp4', 
            '1080', 
            progress_callback, 
            video_info, 
            custom_download_dir=download_dir
        )
        
        # Wait for download to complete
        try:
            result = future.result()  # This will block until the download completes
            
            if isinstance(result, str) and result.startswith("Error"):
                print(f"Download failed: {result}")
                return {
                    "success": False,
                    "message": f"Download failed: {result}",
                    "downloaded_count": 0,
                    "files": []
                }
            else:
                print(f"Completed download: {video_url}")
                
                # Clean up any leftover audio files
                for f in os.listdir(download_dir):
                    if f.endswith('.m4a'):
                        os.remove(os.path.join(download_dir, f))
                
                files = [f for f in os.listdir(download_dir) if os.path.isfile(os.path.join(download_dir, f))]
                
                # Update candidate status to downloaded
                try:
                    session = get_db_session()
                    candidate = session.query(CandidateVideo).filter(
                        CandidateVideo.id == candidate_id
                    ).first()
                    
                    if candidate:
                        candidate.status = "downloaded"
                        candidate.last_download_date = datetime.datetime.utcnow()
                        session.commit()
                    
                    session.close()
                except Exception as e:
                    print(f"Error updating candidate status after download: {str(e)}")
                    if 'session' in locals():
                        session.close()
                
                return {
                    "success": True,
                    "message": "Download completed successfully",
                    "downloaded_count": 1,
                    "files": files,
                    "video": video_info,
                    "download_path": download_dir
                }
                
        except Exception as e:
            error_message = f"Exception during download: {str(e)}"
            print(error_message)
            return {
                "success": False,
                "message": error_message,
                "downloaded_count": 0,
                "files": []
            }
            
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(error_message)
        return {
            "success": False,
            "message": error_message,
            "downloaded_count": 0,
            "files": []
        }

# Keep the original download_videos function for backward compatibility
def download_videos(category, approved_videos=None, num_videos=20, min_duration=300, max_duration=600, progress_callback=None):
    """Downloads YouTube videos using yt-dlp, with validation."""
    try:
        download_dir = os.path.join(os.getcwd(), f"{category}_videos")
        os.makedirs(download_dir, exist_ok=True)
        
        # If approved_videos is provided, use those instead of searching
        candidate_videos = []
        if approved_videos and isinstance(approved_videos, list) and len(approved_videos) > 0:
            # Use the pre-approved videos
            candidate_videos = approved_videos
            print(f"\nProcessing {len(candidate_videos)} pre-approved videos.")
        else:
            # Find videos using the original logic (for backward compatibility)
            result = find_candidate_videos(category, num_videos, min_duration, max_duration)
            candidate_videos = [video['url'] for video in result.get('candidates', [])]
            print(f"\nFound {len(candidate_videos)} videos automatically without approval.")

        if not candidate_videos:
            print("\nNo videos were found matching the criteria.")
            return {
                "success": False,
                "message": "No videos were found matching the criteria.",
                "downloaded_count": 0,
                "files": []
            }
        else:
            # Get the downloader singleton instance
            downloader = Downloader()
            download_futures = []
            
            print(f"\nStarting concurrent download of {len(candidate_videos)} videos...")
            
            # Submit all downloads to the thread pool with video info
            for video_url in candidate_videos:
                # Get video info before download
                video_info = {}
                try:
                    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                        info = ydl.extract_info(video_url, download=False)
                        video_info = {
                            'title': info.get('title', ''),
                            'duration': info.get('duration', 0),
                            'height': info.get('height', 0),
                            'vbr': max(info.get('vbr', 0) or 0, info.get('tbr', 0) or 0),
                            'abr': info.get('abr', 0) or 0
                        }
                except Exception as e:
                    print(f"Error getting video info: {str(e)}")
                
                future = downloader.download_video(video_url, category, 'mp4', '1080', progress_callback, video_info)
                download_futures.append((video_url, future))
                print(f"Submitted download task for: {video_url}")
            
            # Wait for all downloads to complete
            downloaded_count = 0
            failed_downloads = []
            
            for video_url, future in download_futures:
                try:
                    result = future.result()  # This will block until the download completes
                    if isinstance(result, str) and result.startswith("Error"):
                        print(f"Download failed: {result}")
                        failed_downloads.append(video_url)
                    else:
                        downloaded_count += 1
                        print(f"Completed download {downloaded_count}/{len(candidate_videos)}: {video_url}")
                except Exception as e:
                    print(f"Exception during download of {video_url}: {str(e)}")
                    failed_downloads.append(video_url)
            
            # Remove any leftover audio files
            for f in os.listdir(download_dir):
                if f.endswith('.m4a'):
                    os.remove(os.path.join(download_dir, f))
                    
            files = os.listdir(download_dir)
            if files:
                print("\nDownloaded files:")
                for file in files:
                    print(f"- {file}")
            else:
                print("\nWarning: Download directory is empty despite successful downloads.")
            
            return {
                "success": downloaded_count > 0,
                "message": f"Downloaded {downloaded_count} videos, {len(failed_downloads)} failed" if failed_downloads else f"Downloaded {downloaded_count} videos",
                "downloaded_count": downloaded_count,
                "files": files,
                "failed_downloads": failed_downloads
            }
        
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(error_message)
        return {
            "success": False,
            "message": error_message,
            "downloaded_count": 0,
            "files": []
        }

def cancel_download_by_candidate_id(candidate_id):
    """
    Cancel a download using the candidate ID
    
    Args:
        candidate_id: The ID of the candidate video to cancel
        
    Returns:
        dict: Result of the cancellation operation
    """
    video_info = get_candidate_video(candidate_id)
    
    if not video_info:
        return {
            "success": False,
            "message": f"Video with ID {candidate_id} not found"
        }
    
    video_url = video_info['url']
    
    # Get the downloader singleton instance
    downloader = Downloader()
    
    # Try to cancel the download
    cancelled = downloader.cancel_download(video_url)
    
    # Update status in database
    try:
        session = get_db_session()
        candidate = session.query(CandidateVideo).filter(
            CandidateVideo.id == candidate_id
        ).first()
        
        if candidate:
            candidate.status = "cancelled"
            session.commit()
            
            # Also update the main video record if it exists
            video = session.query(Video).filter(Video.url == video_url).first()
            if video:
                video.status = "cancelled"
                video.error_message = "Download cancelled by user"
                session.commit()
        
        session.close()
    except Exception as e:
        print(f"Error updating candidate status: {str(e)}")
        if 'session' in locals():
            session.close()
    
    if cancelled:
        return {
            "success": True,
            "message": f"Download cancelled for video {video_info['title']}",
            "video": {
                "candidate_id": candidate_id,
                "title": video_info['title'],
                "url": video_url
            }
        }
    else:
        # Check if download is not active (may be already completed or not started)
        active_downloads = downloader.get_active_downloads()
        
        if video_url in active_downloads:
            return {
                "success": False, 
                "message": f"Failed to cancel download for {video_info['title']}"
            }
        else:
            return {
                "success": False,
                "message": f"No active download found for {video_info['title']}. It may have already completed or not started yet."
            }
