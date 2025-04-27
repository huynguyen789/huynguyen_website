# -*- coding: utf-8 -*-
"""
Consolidated tool functions for the AI Assistant.
Includes utility functions, web search capabilities, and YouTube transcript fetching.
Each function includes its own imports for easier standalone testing.
"""

#get_current_time
from datetime import datetime # Moved import inside

#get_youtube_transcript
import re #
from youtube_transcript_api import YouTubeTranscriptApi # Moved import inside
from typing import Dict, Any # Keep for return type annotation

#perform_web_search
import os # Moved import inside
from typing import Dict, Any # Moved import inside


# Note: Logger configuration remains global
import logging
logger = logging.getLogger(__name__)


# --- Utility Functions ---
def get_current_time() -> str:
    """
    Input: None
    Process: Gets the current time.
    Output: Formatted current time string (e.g., "03:30 PM, July 26, 2024").
    """
    return datetime.now().strftime("%I:%M %p, %B %d, %Y")


# --- Web Search Function ---
def perform_web_search(query: str):
    """
    Tool function wrapper for the AI search module.
    Input: Search query string.
    Process: Calls the standalone search function and extracts the summary.
            Requires SERPER_API_KEY and OPENROUTER_API_KEY environment variables.
            Imports ai_search dynamically.
    Output: Dictionary containing the search summary or an error message.
    """


    # Import ai_search dynamically within the function
    try:
        import tools.ai_search as ai_search
    except ImportError:
        try:
            import ai_search as ai_search
        except ImportError:
            logger.error("Could not import ai_search module.")
            return {"error": "Failed to import the necessary search module."}

    # Removed st.toast call
    logger.info(f"Performing web search for: '{query}'...")

    # Check if required API keys are available (as environment variables for ai_search.py)
    if not os.environ.get("SERPER_API_KEY") or not os.environ.get("OPENROUTER_API_KEY"):
         logger.warning("SERPER_API_KEY or OPENROUTER_API_KEY might be missing as environment variables for ai_search.")
         # Note: ai_search.search will return its own error if keys are truly missing.

    try:
        logger.info(f"Tool 'perform_web_search' executing ai_search.search for query: '{query}'")
        search_result = ai_search.search(
            query=query,
            model="google/gemini-2.0-flash-001",
            search_depth="fast",
            include_youtube=True
        )
        logger.info(f"Tool 'perform_web_search' received result: status={search_result['status']}")

        # Process the result for the LLM
        if search_result["status"] == "success":
            summary = search_result.get("summary", "Search completed, but no summary could be generated.")
            sources = search_result.get("visited_links", [])[:3]
            response_data: Dict[str, Any] = {
                "summary": summary if summary else "No specific summary generated.",
                "top_sources_checked": sources
            }
            return response_data
        elif search_result["status"] == "no_content":
             summary = search_result.get("summary", "Search results found, but could not retrieve content from any sources.")
             sources = search_result.get("visited_links", [])[:3]
             response_data: Dict[str, Any] = {
                 "summary": summary,
                 "top_sources_checked": sources
             }
             return response_data
        elif search_result["status"] == "no_results":
            return {"summary": "No search results found for the query."}
        else: # Error case within ai_search
            error_msg = search_result.get("error_message", "An unknown error occurred during search.")
            logger.error(f"Search tool failed with status {search_result['status']}: {error_msg}")
            return {"error": f"Search failed: {error_msg}"}

    except Exception as e:
        logger.error(f"Error calling ai_search.search from tool: {e}", exc_info=True)
        return {"error": f"Failed to execute search tool: {str(e)}"}


# --- YouTube Transcript Function ---
def get_youtube_transcript(youtube_url: str):
    """
    Tool function to fetch the transcript of a YouTube video.
    Input: YouTube video URL string.
    Process: Extracts video ID, calls YouTubeTranscriptApi, formats output and counts words.
    Output: Dictionary containing:
            - status ('success' or 'error')
            - transcript text and word count on success
            - error message on failure
    """


    logger.info(f"Tool 'get_youtube_transcript' called for URL: {youtube_url}")

    try:
        # --- Start of merged fetching logic ---
        video_id_match = re.search(r'(?:v=|/|embed/|shorts/)([0-9A-Za-z_-]{11})', youtube_url)
        if not video_id_match:
            logger.warning(f"Could not extract video ID from URL: {youtube_url}")
            response: Dict[str, Any] = {"status": "error", "message": "Could not extract video ID from URL."}
            return response

        video_id = video_id_match.group(1)

        # Try fetching the transcript
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = ' '.join([entry['text'] for entry in transcript_list])
            word_count = len(transcript_text.split())
            logger.info(f"Successfully fetched transcript for {youtube_url}")
            response: Dict[str, Any] = {
                "status": "success", 
                "transcript": transcript_text,
                "word_count": word_count
            }
            return response
        except Exception as fetch_error:
            # Log specific exceptions from the API call
            logger.warning(f"Could not fetch transcript for {youtube_url} (ID: {video_id}): {fetch_error}")
            response: Dict[str, Any] = {"status": "error", "message": f"Could not retrieve transcript: {fetch_error}"}
            return response
        # --- End of merged fetching logic ---

    except Exception as e:
        # Catch unexpected errors during the tool's execution (e.g., regex error?)
        logger.error(f"Unexpected error in get_youtube_transcript tool for {youtube_url}: {e}", exc_info=True)
        response: Dict[str, Any] = {"status": "error", "message": f"Failed to execute transcript tool due to an unexpected error: {str(e)}"}
        return response 