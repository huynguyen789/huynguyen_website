# -*- coding: utf-8 -*-
"""
Standalone AI-Powered Search Function.

Takes a query, searches the web, fetches content, summarizes, and returns
a dictionary containing the summary, source details, and raw results.

Requires SERPER_API_KEY and OPENROUTER_API_KEY environment variables to be set.
"""

import requests
import json
from datetime import datetime
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from requests.exceptions import RequestException, Timeout
import re
import logging
from openai import OpenAI
from newspaper import Article
import html2text
import os
# Import Dict and Any for type hinting the return value
from typing import List, Dict, Tuple, Optional, Any, Union

# --- Configuration ---
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Internal Helper Functions (Keep previous helpers: _get_openai_client, _get_llm_response, _fetch_serper_results, _fetch_youtube_transcript, _clean_web_content, _fetch_and_clean_website, _gather_content_from_results, _format_content_for_llm, _generate_search_summary) ---

def _get_openai_client(api_key: str) -> OpenAI:
    """Internal helper to create OpenAI client."""
    if not api_key:
        raise ValueError("OpenRouter API Key is required.")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

def _get_llm_response(
    prompt: str,
    model_name: str,
    api_key: str,
    system_prompt: Optional[str] = None,
    client: Optional[OpenAI] = None
) -> str:
    """Internal helper to get LLM response."""
    if client is None:
        client = _get_openai_client(api_key)

    try:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return completion.choices[0].message.content

    except Exception as e:
        logger.error(f"Error generating response with {model_name}: {str(e)}", exc_info=True)
        raise Exception(f"LLM Error ({model_name}): {str(e)}")

def _fetch_serper_results(query: str, api_key: str) -> Dict[str, Any]:
    """Internal helper to fetch Serper results."""
    if not api_key:
        raise ValueError("Serper API Key is required.")
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
        results = response.json()
        if 'organic' not in results:
            logger.warning(f"Serper results missing 'organic' key for query: {query}")
            return {"organic": [], "searchParameters": {"q": query}, "type": "empty_result"}
        return results
    except Timeout:
        logger.error(f"Serper API request timed out for query: {query}")
        raise Exception("Search API request timed out.")
    except RequestException as e:
        logger.error(f"Serper API request error for query '{query}': {e}")
        raise Exception(f"Search API error: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode Serper API response for query '{query}': {e}")
        raise Exception("Failed to parse search results.")

def _fetch_youtube_transcript(video_url: str) -> Optional[str]:
    """Internal helper to fetch YouTube transcript."""
    try:
        video_id_match = re.search(r'(?:v=|/|embed/|shorts/)([0-9A-Za-z_-]{11})', video_url)
        if not video_id_match:
            logger.warning(f"Could not extract video ID from URL: {video_url}")
            return None
        video_id = video_id_match.group(1)
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([entry['text'] for entry in transcript_list])
        return transcript_text
    except Exception as e:
        logger.warning(f"Could not fetch transcript for {video_url}: {e}")
        return None

def _clean_web_content(html_content: str, url: str) -> str:
    """Internal helper to clean HTML content."""
    content = ""
    min_content_length_newspaper = 200
    min_content_length_fallback = 100
    try: # Wrap newspaper in try-except
        article = Article(url)
        article.set_html(html_content)
        article.parse()
        if article.text and len(article.text.strip()) >= min_content_length_newspaper:
            content = article.text.strip()
            logger.debug(f"Extracted content using newspaper3k for {url}")
    except Exception as e:
        logger.warning(f"Newspaper3k extraction failed for {url}: {e}")
    if not content:
        try: # Wrap html2text in try-except
            soup = BeautifulSoup(html_content, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'input', 'noscript', 'img', 'figure', 'iframe']):
                element.decompose()
            main_content_area = (soup.find('main') or soup.find('article') or
                                 soup.find('div', id=re.compile(r'main|content|body', re.I)) or
                                 soup.find('div', class_=re.compile(r'post|entry|content|article|body|text', re.I)) or
                                 soup.body)
            if main_content_area:
                h = html2text.HTML2Text()
                h.ignore_links = False; h.ignore_images = True; h.ignore_tables = False
                h.body_width = 0; h.unicode_snob = True; h.escape_snob = True
                text_content = h.handle(str(main_content_area))
                text_content = re.sub(r'\n\s*\n', '\n\n', text_content).strip()
                text_content = re.sub(r'!\[.*?\]\(.*?\)', '', text_content)
                text_content = re.sub(r'\n{3,}', '\n\n', text_content).strip()
                if len(text_content) >= min_content_length_fallback:
                    content = text_content
                    logger.debug(f"Extracted content using html2text for {url}")
                else: logger.warning(f"html2text extracted very short content ({len(text_content)} chars) for {url}")
        except Exception as e: logger.warning(f"html2text extraction failed for {url}: {e}")
    if not content:
        try: # Wrap basic BS in try-except
            if 'soup' not in locals():
                 soup = BeautifulSoup(html_content, 'html.parser')
                 for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'input', 'noscript', 'img', 'figure', 'iframe']):
                     element.decompose()
            text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'pre', 'code'])
            extracted_text = '\n'.join(elem.get_text(separator='\n', strip=True) for elem in text_elements if elem.get_text(strip=True))
            extracted_text = re.sub(r'\n{3,}', '\n\n', extracted_text).strip()
            if len(extracted_text) >= min_content_length_fallback:
                content = extracted_text
                logger.debug(f"Extracted content using basic BeautifulSoup for {url}")
            else: logger.warning(f"Basic BeautifulSoup extraction yielded very short content ({len(extracted_text)} chars) for {url}")
        except Exception as e: logger.warning(f"Basic BeautifulSoup extraction failed for {url}: {e}")
    return content if content else "No main content found."

def _fetch_and_clean_website(url: str, timeout: int = 10) -> Optional[str]:
    """Internal helper to fetch and clean website content."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8', 'Accept-Language': 'en-US,en;q=0.9', 'Connection': 'keep-alive'}
        response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type: logger.warning(f"Skipping non-HTML content type '{content_type}' for {url}"); return None
        if response.status_code == 200 and response.content:
            html_content = response.text
            cleaned_content = _clean_web_content(html_content, url)
            if cleaned_content and cleaned_content != "No main content found.": return cleaned_content
            else: logger.warning(f"Cleaning returned no significant content for {url}"); return None
        else: logger.warning(f"Received status code {response.status_code} for {url}"); return None
    except Timeout: logger.warning(f"Request timed out for {url}"); return None
    except RequestException as e: http_error_msg = f" (Status Code: {e.response.status_code})" if e.response is not None else ""; logger.warning(f"Failed to fetch {url}: {e.__class__.__name__}{http_error_msg}"); return None
    except Exception as e: logger.error(f"Unexpected error fetching/cleaning {url}: {e}", exc_info=True); return None

def _gather_content_from_results(
    organic_results: List[Dict[str, Any]], target_sources: int, include_youtube: bool, max_attempts: int
) -> Tuple[List[Dict], List[Dict], List[str], List[str], Dict]:
    """Internal helper to gather content from search results."""
    website_contents = []; youtube_contents = []; blocked_urls = []; failed_youtube_urls = []
    successful_website_count = 0; successful_youtube_count = 0; blocked_website_count = 0; failed_youtube_count = 0
    processed_urls = set(); total_attempted = 0
    for i, result in enumerate(organic_results):
        if total_attempted >= max_attempts: logger.warning(f"Reached max processing attempts ({max_attempts})."); break
        total_successful = successful_website_count + successful_youtube_count
        if total_successful >= target_sources: logger.info(f"Reached target source count ({target_sources})."); break
        url = result.get('link'); title = result.get('title', 'Untitled')
        if not url or url in processed_urls: continue
        processed_urls.add(url); total_attempted += 1
        logger.info(f"Attempt {total_attempted}/{max_attempts}: Processing URL {i+1}/{len(organic_results)}: {url}")
        is_youtube = 'youtube.com/watch?v=' in url or 'youtu.be/' in url
        try:
            if is_youtube and include_youtube:
                logger.info(f"-> Processing as YouTube video...")
                transcript = _fetch_youtube_transcript(url)
                if transcript: logger.info(f"  [SUCCESS] Fetched transcript for {url}"); youtube_contents.append({'title': title, 'link': url, 'content': transcript}); successful_youtube_count += 1
                else: logger.warning(f"  [FAILED] Could not get transcript for {url}"); failed_youtube_urls.append(url); failed_youtube_count += 1
            elif not is_youtube:
                logger.info(f"-> Processing as website...")
                content = _fetch_and_clean_website(url)
                if content: logger.info(f"  [SUCCESS] Fetched and cleaned content for {url}"); website_contents.append({'title': title, 'link': url, 'content': content}); successful_website_count += 1
                else: logger.warning(f"  [FAILED/BLOCKED] Could not get content for {url}"); blocked_urls.append(url); blocked_website_count += 1
        except Exception as e:
            logger.error(f"  [ERROR] Unexpected error processing result {url}: {e}", exc_info=True)
            if is_youtube: failed_youtube_urls.append(url); failed_youtube_count += 1
            else: blocked_urls.append(url); blocked_website_count += 1
            continue
    access_stats = {"target_sources": target_sources, "total_attempted": total_attempted, "successful_websites": successful_website_count, "successful_youtube": successful_youtube_count, "blocked_websites": blocked_website_count, "failed_youtube": failed_youtube_count, "blocked_urls": blocked_urls, "failed_youtube_urls": failed_youtube_urls}
    logger.info(f"Content gathering finished. Stats: {access_stats}")
    return website_contents, youtube_contents, blocked_urls, failed_youtube_urls, access_stats

def _format_content_for_llm(
    website_contents: List[Dict], youtube_contents: List[Dict]
) -> Tuple[str, Dict[str, str]]:
    """Formats content with XML-like tags for the LLM prompt and creates source mapping."""
    formatted_content = ""
    source_mapping = {}
    max_content_length_per_source = 15000 # Limit length per source

    for i, web_content in enumerate(website_contents, 1):
        source_id = f"website_{i}"
        source_mapping[source_id] = web_content['link']
        truncated_content = web_content['content'][:max_content_length_per_source]
        if len(web_content['content']) > max_content_length_per_source:
            truncated_content += "\n[Content truncated]"
            logger.warning(f"Truncated content for {web_content['link']}")
        formatted_content += f"<{source_id}>\nTitle: {web_content['title']}\nURL: {web_content['link']}\nSource ID: {source_id}\n\n{truncated_content}\n</{source_id}>\n\n"

    for i, yt_content in enumerate(youtube_contents, 1):
        source_id = f"youtube_video_{i}"
        source_mapping[source_id] = yt_content['link']
        truncated_content = yt_content['content'][:max_content_length_per_source]
        if len(yt_content['content']) > max_content_length_per_source:
            truncated_content += "\n[Content truncated]"
            logger.warning(f"Truncated transcript for {yt_content['link']}")
        formatted_content += f"<{source_id}>\nTitle: {yt_content['title']}\nURL: {yt_content['link']}\nSource ID: {source_id}\n\n{truncated_content}\n</{source_id}>\n\n"

    return formatted_content.strip(), source_mapping

def _generate_combined_raw_text(
    website_contents: List[Dict], youtube_contents: List[Dict]
) -> str:
    """Combines the raw text content from successful sources for display/output."""
    combined_parts = []
    for i, web_content in enumerate(website_contents, 1):
        display_entry = f"--- Source: Website {i} ---\nTitle: {web_content['title']}\nURL: {web_content['link']}\n\n{web_content['content']}"
        combined_parts.append(display_entry)
    for i, yt_content in enumerate(youtube_contents, 1):
        display_entry = f"--- Source: YouTube {i} ---\nTitle: {yt_content['title']}\nURL: {yt_content['link']}\n\n{yt_content['content']}"
        combined_parts.append(display_entry)
    return "\n\n".join(combined_parts)

def _generate_search_summary(
        query: str, formatted_content: str, source_mapping: Dict[str, str],
        model: str, openrouter_api_key: str, client: Optional[OpenAI] = None
    ) -> str:
    """Internal helper to generate the final summary."""
    today_date = datetime.now().strftime("%Y-%m-%d")
    source_references = "\n".join([f"{source_id}: {url}" for source_id, url in source_mapping.items()])
    if not source_references: source_references = "No sources successfully processed."
    system_prompt = "<role>You are a world class search engine and summarizer...</role>" # Keep full prompt
    prompt_template = f"""<instructions>
Source Reference Map (Use these URLs for citations):
{source_references}
Today's Date: {today_date}
User Query: {query}


Carefully think based on the user query and content to generate a world-class answer.
    Output format: First generate a short answer, then create a detail answer. With clear title for both.
    Be concise but include all of the important details. 
    Give examples if possible.  
    Focus on high quality and accuracy: filter, compare, select from provided content to get the best answer! 
    If you dont have enough info, state so and give users links to look by themself. Do not make up info!  
    Always cite sources with the link in the answer(embed it if you can, so it's look nicer instead of the full long links). Which part come from which source as hyperlink.
    Output nicely in Markdown with clear tittles and contents. 

    *** Important: For coding search:
    - If you found many different answers, code syntax, or approaches, alert & show them all to user. User can test out to see which one works(note to them that too)
    - Pay more attention to date on the content for the newest code version and show the user that info too.
    - Example: If you found 2 results: 
            Old and wrong way:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {{"role": "user", "content": prompt}}
                ]
            )

            New and correct way:
            from openai import OpenAI
            client = OpenAI()

            completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {{"role": "system", "content": "You are a helpful assistant."}},
                {{"role": "user", "content": "Hello!"}}
            ]
            )

            print(completion.choices[0].message.content)

            Alert & show the user both examples so they can select from! 

If you done a great job, you will get a 100k bonus this year. If not a cat will die.


<content>
{formatted_content}
</content>
""" # Keep full prompt details
    try:
        logger.info(f"Generating summary for query '{query}' using model {model}")
        summary = _get_llm_response(prompt=prompt_template, model_name=model, api_key=openrouter_api_key, system_prompt=system_prompt, client=client)
        logger.info(f"Summary generation successful.")
        return summary
    except Exception as e:
        logger.error(f"Summary generation failed: {e}", exc_info=True)
        # Return error within the summary field for consistency in the dict structure
        return f"Error: Failed to generate summary. {str(e)}"


# --- The Public Search Function ---

def search(
    query: str,
    model: str = "google/gemini-2.0-flash-001",
    search_depth: str = "fast",
    include_youtube: bool = True
) -> Dict[str, Any]:
    """
    Performs an AI-powered web search and returns a dictionary containing
    the summary, source details, and raw results.

    Requires environment variables:
    - SERPER_API_KEY: Your Serper.dev API key.
    - OPENROUTER_API_KEY: Your OpenRouter.ai API key.

    Args:
        query: The user's search query.
        model: The OpenRouter model identifier for summarization
               (default: "google/gemini-2.0-flash-001").
        search_depth: "fast" (approx 5 sources) or "deep" (approx 10 sources)
                      (default: "fast").
        include_youtube: Whether to include YouTube transcripts (default: True).

    Returns:
        A dictionary with the following keys:
        - 'status' (str): 'success', 'no_results', 'no_content', or 'error'.
        - 'query' (str): The original search query.
        - 'summary' (str | None): The AI-generated summary, or an error message, or None if skipped.
        - 'visited_links' (List[str]): List of URLs successfully fetched and processed.
        - 'combined_raw_text' (str | None): Concatenated text from visited links, or None.
        - 'serper_results' (Dict | None): Raw JSON response from Serper API, or None.
        - 'access_stats' (Dict | None): Statistics about content gathering, or None.
        - 'error_message' (str | None): Specific error message if status is 'error'.
    """
    logger.info(f"Starting search for query: '{query}'")
    logger.info(f"Parameters: model={model}, depth={search_depth}, youtube={include_youtube}")

    # Initialize result dictionary structure
    result = {
        "status": "error",
        "query": query,
        "summary": None,
        "visited_links": [],
        "combined_raw_text": None,
        "serper_results": None,
        "access_stats": None,
        "error_message": None
    }

    # --- API Key Loading ---
    serper_api_key = os.environ.get("SERPER_API_KEY")
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

    if not serper_api_key:
        msg = "SERPER_API_KEY environment variable is not set."
        logger.error(msg)
        result["error_message"] = msg
        return result
    if not openrouter_api_key:
        msg = "OPENROUTER_API_KEY environment variable is not set."
        logger.error(msg)
        result["error_message"] = msg
        return result
    # --- End API Key Loading ---

    try:
        target_sources = 5 if search_depth == "fast" else 10
        max_attempts = 15 if search_depth == "fast" else 30

        # 1. Initial Search
        logger.info("Fetching search results from Serper...")
        try:
            initial_search_results = _fetch_serper_results(query, serper_api_key)
            result["serper_results"] = initial_search_results # Store raw results
            organic_results = initial_search_results.get('organic', [])
        except Exception as e:
            msg = f"Failed to fetch search results: {str(e)}"
            logger.error(msg, exc_info=True)
            result["error_message"] = msg
            return result # Return error state

        if not organic_results:
            logger.warning(f"No organic results found for query: {query}")
            result["status"] = "no_results"
            result["summary"] = "No search results found for your query."
            return result # Return no results state

        # 2. Gather Content
        logger.info("Gathering content from search results...")
        website_contents, youtube_contents, _, _, access_stats = _gather_content_from_results(
            organic_results, target_sources, include_youtube, max_attempts
        )
        result["access_stats"] = access_stats # Store access stats

        # Populate visited links
        visited_links = [item['link'] for item in website_contents] + [item['link'] for item in youtube_contents]
        result["visited_links"] = visited_links

        total_successful = access_stats['successful_websites'] + access_stats['successful_youtube']
        logger.info(f"Gathered content from {total_successful} sources.")

        if not website_contents and not youtube_contents:
             logger.warning("Could not retrieve content from any sources.")
             result["status"] = "no_content"
             result["summary"] = "Search results found, but could not retrieve content from any sources."
             # Keep serper_results and access_stats populated
             return result # Return no content state

        # Generate combined raw text *before* formatting for LLM
        result["combined_raw_text"] = _generate_combined_raw_text(website_contents, youtube_contents)

        # 3. Format Content for LLM
        logger.info("Formatting content for LLM...")
        formatted_content, source_mapping = _format_content_for_llm(
            website_contents, youtube_contents
        )

        # 4. Generate Summary
        logger.info("Generating summary using LLM...")
        summary = _generate_search_summary(
            query=query,
            formatted_content=formatted_content,
            source_mapping=source_mapping,
            model=model,
            openrouter_api_key=openrouter_api_key,
        )
        result["summary"] = summary

        # Check if summary generation itself returned an error message
        if summary.startswith("Error:"):
            result["status"] = "error"
            result["error_message"] = summary # Put LLM error here
        else:
            result["status"] = "success" # Mark as success only if summary is generated

        logger.info("Search and summarization complete.")
        return result

    except Exception as e:
        # Catch any unexpected errors during orchestration
        msg = f"An unexpected error occurred during search: {str(e)}"
        logger.error(msg, exc_info=True)
        result["error_message"] = msg
        # Keep any data gathered so far (like serper results)
        return result # Return error state


# # --- Example Usage ---
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#     if not os.environ.get("SERPER_API_KEY") or not os.environ.get("OPENROUTER_API_KEY"):
#         print("\nERROR: Please set SERPER_API_KEY and OPENROUTER_API_KEY environment variables.")
#         exit(1)

#     test_query = "how to call openai api models python?"

#     print(f"\n--- Running search for: '{test_query}' ---")
#     search_result = search(query=test_query)

#     print("\n--- Search Result Dictionary ---")
#     # Pretty print the dictionary keys and types/lengths for clarity
#     print(f"Status: {search_result['status']}")
#     print(f"Query: {search_result['query']}")
#     print(f"Summary:\n{search_result['summary']}\n")
#     print(f"Visited Links ({len(search_result['visited_links'])}): {search_result['visited_links']}")
#     # Avoid printing huge raw text, just show length
#     raw_text_len = len(search_result['combined_raw_text']) if search_result['combined_raw_text'] else 0
#     print(f"Combined Raw Text Length: {raw_text_len} characters")
#     # Optionally print a snippet:
#     # print(f"Combined Raw Text Snippet:\n{search_result['combined_raw_text'][:500]}...\n")
#     print(f"Access Stats: {search_result['access_stats']}")
#     # Avoid printing huge serper results, just confirm presence
#     print(f"Serper Results Present: {'Yes' if search_result['serper_results'] else 'No'}")
#     if search_result['error_message']:
#         print(f"Error Message: {search_result['error_message']}")
#     print("--------------------------------\n")

