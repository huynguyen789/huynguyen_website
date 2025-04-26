# -*- coding: utf-8 -*-
"""
Consolidated tool functions for the AI Assistant.
Includes utility functions and web search capabilities.
"""

import os
import logging
import streamlit as st # For st.toast
from typing import Dict, Any
from datetime import datetime

# Import the core search logic (ensure ai_search.py is in the same directory or accessible)
try:
    import tools.ai_search as ai_search
except ImportError:
    # Handle cases where the script might be run from a different context
    import ai_search as ai_search # Fallback if run directly or path issues


# Configure logger for this module
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

def perform_web_search(query: str) -> Dict[str, Any]:
    """
    Tool function wrapper for the AI search module.
    Input: Search query string.
    Process: Calls the standalone search function and extracts the summary.
            Requires SERPER_API_KEY and OPENROUTER_API_KEY environment variables.
    Output: Dictionary containing the search summary or an error message.
    """
    st.toast(f"Performing web search for: '{query}'...") # Simple feedback in UI

    # Check if required API keys are available (as environment variables for ai_search.py)
    # ai_search itself handles the check now, but we can keep a log message here
    if not os.environ.get("SERPER_API_KEY") or not os.environ.get("OPENROUTER_API_KEY"):
         logger.warning("SERPER_API_KEY or OPENROUTER_API_KEY might be missing as environment variables for ai_search.")
         # Note: ai_search.search will return its own error if keys are truly missing.

    try:
        # Use a default model suitable for summarization within the tool context
        # Use 'fast' search depth for tool calls to keep them quicker.
        logger.info(f"Tool 'perform_web_search' executing ai_search.search for query: '{query}'")
        search_result = ai_search.search(
            query=query,
            model="google/gemini-2.0-flash-001", # Fast model for tool summary
            search_depth="fast",
            include_youtube=True # Include YouTube by default for tool use
        )
        logger.info(f"Tool 'perform_web_search' received result: status={search_result['status']}")

        # Process the result for the LLM
        if search_result["status"] == "success":
            summary = search_result.get("summary", "Search completed, but no summary could be generated.")
            sources = search_result.get("visited_links", [])[:3] # Get top 3 links
            response_data = {
                "summary": summary if summary else "No specific summary generated.",
                "top_sources_checked": sources
            }
            return response_data
        elif search_result["status"] == "no_content":
             # If no content, still provide the status message as summary
             summary = search_result.get("summary", "Search results found, but could not retrieve content from any sources.")
             sources = search_result.get("visited_links", [])[:3] # Usually empty here
             response_data = {
                 "summary": summary,
                 "top_sources_checked": sources
             }
             return response_data
        elif search_result["status"] == "no_results":
            return {"summary": "No search results found for the query."}
        else: # Error case within ai_search
            error_msg = search_result.get("error_message", "An unknown error occurred during search.")
            logger.error(f"Search tool failed with status {search_result['status']}: {error_msg}")
            # Return the specific error from ai_search
            return {"error": f"Search failed: {error_msg}"}

    except Exception as e:
        logger.error(f"Error calling ai_search.search from tool: {e}", exc_info=True)
        # Return a generic error if the call itself failed
        return {"error": f"Failed to execute search tool: {str(e)}"} 