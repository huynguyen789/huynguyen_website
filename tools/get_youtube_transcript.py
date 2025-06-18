#!/usr/bin/env python3
"""
YouTube transcript functionality.
Provides a clean function to fetch transcripts from YouTube videos.
"""

import re
from youtube_transcript_api import YouTubeTranscriptApi
from typing import Tuple
import logging

# Set up logging for debugging
logger = logging.getLogger(__name__)

def get_youtube_transcript(video_url: str, debug: bool = False) -> Tuple[str, str]:
    """
    Input: YouTube video URL, optional debug flag
    Process: Extracts video ID, fetches available transcripts, and returns transcript text
    Output: Tuple of (transcript_text, status) where status is "success" or error message
    """
    if debug:
        logger.info(f"Attempting to fetch transcript for: {video_url}")
    
    # Extract video ID
    video_id_match = re.search(r'(?:v=|/|embed/|shorts/)([0-9A-Za-z_-]{11})', video_url)
    if not video_id_match:
        return "", "Invalid YouTube URL format"
    
    video_id = video_id_match.group(1)
    if debug:
        logger.info(f"Extracted video ID: {video_id}")
    
    try:
        # Initialize API with custom configuration
        ytt_api = YouTubeTranscriptApi()
        
        if debug:
            logger.info("Listing available transcripts...")
        
        # First, try to list available transcripts
        transcript_list = ytt_api.list(video_id)
        
        available_transcripts = []
        for transcript in transcript_list:
            transcript_info = {
                'language': transcript.language,
                'language_code': transcript.language_code,
                'is_generated': transcript.is_generated,
                'is_translatable': transcript.is_translatable
            }
            available_transcripts.append(transcript_info)
            if debug:
                logger.info(f"Found transcript: {transcript_info}")
        
        if not available_transcripts:
            return "", "No transcripts available for this video"
        
        if debug:
            logger.info(f"Fetching transcript for video {video_id}...")
        
        # Fetch the transcript
        fetched_transcript = ytt_api.fetch(video_id)
        
        if debug:
            logger.info(f"Successfully fetched transcript: {len(fetched_transcript)} segments")
        
        # Convert to raw text
        raw_data = fetched_transcript.to_raw_data()
        full_text = ' '.join([entry['text'] for entry in raw_data])
        
        if not full_text:
            return "", "Transcript is empty"
            
        if debug:
            logger.info(f"Transcript length: {len(full_text)} characters")
            
        return full_text, "success"
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        if debug:
            logger.error(f"Error fetching transcript: {error_type} - {error_msg}")
        
        # Provide specific guidance based on error type
        if "ParseError" in error_type:
            return "", f"ParseError: Video has no captions/transcripts, is private, age-restricted, or region-blocked. Raw error: {error_msg}"
        elif "TranscriptsDisabled" in error_type:
            return "", "TranscriptsDisabled: Video owner has disabled captions"
        elif "VideoUnavailable" in error_type:
            return "", "VideoUnavailable: Video is private, deleted, or doesn't exist"
        elif "RequestBlocked" in error_type or "IpBlocked" in error_type:
            return "", "RequestBlocked: YouTube is blocking your IP address"
        else:
            return "", f"{error_type}: {error_msg}"

def get_youtube_transcript_with_fallback(video_url: str, debug: bool = False) -> Tuple[str, str]:
    """
    Input: YouTube video URL, optional debug flag
    Process: Tries multiple approaches to fetch transcript, including fallback methods
    Output: Tuple of (transcript_text, status) where status is "success" or error message
    """
    # First try the standard approach
    transcript, status = get_youtube_transcript(video_url, debug)
    
    if status == "success":
        return transcript, status
    
    # If that fails, try with different language preferences
    video_id_match = re.search(r'(?:v=|/|embed/|shorts/)([0-9A-Za-z_-]{11})', video_url)
    if not video_id_match:
        return "", "Invalid YouTube URL format"
    
    video_id = video_id_match.group(1)
    
    try:
        if debug:
            logger.info("Trying fallback method with language preferences...")
        
        # Try getting transcript with specific language preferences
        ytt_api = YouTubeTranscriptApi()
        
        # Try English first, then auto-generated, then any available
        language_preferences = ['en', 'en-US', 'en-GB']
        
        for lang in language_preferences:
            try:
                if debug:
                    logger.info(f"Trying language: {lang}")
                transcript_data = ytt_api.get_transcript(video_id, languages=[lang])
                full_text = ' '.join([entry['text'] for entry in transcript_data])
                if full_text:
                    if debug:
                        logger.info(f"Success with language {lang}")
                    return full_text, "success"
            except:
                continue
        
        # If specific languages fail, try getting any available transcript
        try:
            if debug:
                logger.info("Trying to get any available transcript...")
            transcript_data = ytt_api.get_transcript(video_id)
            full_text = ' '.join([entry['text'] for entry in transcript_data])
            if full_text:
                if debug:
                    logger.info("Success with default language")
                return full_text, "success"
        except:
            pass
            
    except Exception as e:
        if debug:
            logger.error(f"Fallback method also failed: {e}")
    
    return "", f"All methods failed. Original error: {status}"

def main():
    """Test with various YouTube videos."""
    print("🧪 YouTube Transcript API Test Script")
    print("=" * 50)
    
    # Test URLs - mix of different types
    test_urls = [
        # Popular educational content (likely to have transcripts)
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Astley - Never Gonna Give You Up
        "https://www.youtube.com/watch?v=9bZkp7q19f0",  # TED Talk
        
        # You can add your own test URLs here
        # "https://www.youtube.com/watch?v=YOUR_VIDEO_ID",
    ]
    
    # Allow user to input custom URL
    custom_url = input("\n🔗 Enter a YouTube URL to test (or press Enter to skip): ").strip()
    if custom_url:
        test_urls.insert(0, custom_url)
    
    success_count = 0
    total_count = len(test_urls)
    
    for i, url in enumerate(test_urls, 1):
        print(f"\n{'=' * 20} Test {i}/{total_count} {'=' * 20}")
        
        # Use the new function
        transcript, status = get_youtube_transcript(url)
        
        if status == "success":
            print(f"✅ Success! Transcript fetched:")
            print(f"📊 Transcript length: {len(transcript)} characters")
            print(f"📄 Preview: {transcript[:200]}...")
            success_count += 1
        else:
            print(f"❌ Error: {status}")
        
        if i < total_count:
            input("\nPress Enter to continue to next test...")
    
    print(f"\n{'=' * 50}")
    print(f"🏁 Test Results: {success_count}/{total_count} successful")
    
    if success_count > 0:
        print("✅ YouTube transcript fetching is working!")
    else:
        print("❌ All tests failed. Check your internet connection and try different videos.")

if __name__ == "__main__":
    main() 