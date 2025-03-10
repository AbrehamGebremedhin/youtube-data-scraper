# YouTube & Image Data Scraper

This application provides REST APIs for downloading high-quality YouTube videos and Google Images.

## Features

- Download YouTube videos by category, with customizable parameters
- Find candidate videos for review before downloading
- Download high-quality (1080p) images from Google Images by category
- Real-time progress tracking via WebSocket connections
- SQLite database for tracking download history

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with the following API keys:

   ```
   YOUTUBE_API_KEY=your_youtube_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

3. Start the server:
   ```bash
   cd rest
   uvicorn app:app --host 0.0.0.0 --port 5000 --reload
   ```

## API Documentation

Once the server is running, visit `http://localhost:5000/docs` to see the complete API documentation.

### Video Download APIs

- `POST /api/candidates` - Find candidate videos for review
- `GET /api/candidates` - List available candidates
- `POST /api/approve/{candidate_id}` - Approve and download a video
- `POST /api/cancel/{candidate_id}` - Cancel a video download
- `GET /api/history` - View video download history

### Image Download APIs

- `POST /api/images/download` - Download images by category
- `GET /api/images/status/{download_id}` - Check image download status
- `POST /api/images/cancel/{download_id}` - Cancel an image download
- `GET /api/images/history` - View image download history

## WebSocket Updates

Connect to `ws://localhost:5000/ws` to receive real-time updates on download progress.

## Requirements

- Python 3.8+
- Google Chrome (for Selenium)
- Google API keys (YouTube & Gemini)
