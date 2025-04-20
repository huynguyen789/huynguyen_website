# -*- coding: utf-8 -*-
"""
Main application file for the portfolio website.
Handles navigation and main layout structure.
Refactored into modular functions.
"""

import streamlit as st
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
from typing import List, Dict, Tuple, Optional, Any, Generator, Union

# --- Configuration ---

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Page configuration
st.set_page_config(
    page_title="Huy Nguyen Portfolio",
    page_icon="👨‍💻",
    layout="wide"
)

# Model configuration
MODEL_CONFIG = [
    "openai/gpt-4.1",
    "openai/o3",
    "openai/o4-mini-high",
    "anthropic/claude-3.7-sonnet",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-pro-preview-03-25",
    "google/gemini-2.0-flash-lite-001",
    "google/gemini-2.5-flash-preview",
    "google/gemini-2.5-flash-preview:thinking",
    "x-ai/grok-3-beta",
    "x-ai/grok-3-mini-beta"
]

# --- AI Interaction Modules ---

def get_openai_client() -> OpenAI:
    """
    Input: None
    Process: Creates and returns an OpenAI client configured for OpenRouter.
    Output: OpenAI client instance.
    """
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENROUTER_API_KEY"],
    )

def get_llm_response(
    prompt: str,
    model_name: str = "openai/gpt-4o",
    system_prompt: Optional[str] = None,
    client: Optional[OpenAI] = None
) -> str:
    """
    Input: prompt, optional model name, optional system prompt, optional OpenAI client.
    Process: Generates a non-streaming response using the specified model via OpenRouter.
    Output: Generated text response string.
    """
    if client is None:
        client = get_openai_client()

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
        logging.error(f"Error generating response with {model_name}: {str(e)}", exc_info=True)
        raise Exception(f"Error generating response with {model_name}: {str(e)}")


def get_streaming_llm_response(
    messages: List[Dict[str, str]],
    model_name: str = "openai/gpt-4o",
    tools: Optional[List[Dict]] = None,
    client: Optional[OpenAI] = None
) -> Generator[Tuple[str, str], None, None]:
    """
    Input: List of messages, optional model name, optional tools list, optional OpenAI client.
    Process: Generates a streaming response using OpenRouter API, handling potential function calls.
    Output: Yields tuples of (chunk, full_response_so_far). Returns the final full response string upon completion or error.
    """
    if client is None:
        client = get_openai_client()

    try:
        # Check if it's an Anthropic model (which might not support tools well or require different handling)
        is_anthropic = "anthropic" in model_name.lower()

        # --- Initial Call (Check for Tool Use) ---
        api_params = {
            "model": model_name,
            "messages": messages,
            "stream": False # First call is non-streaming to check for tool_calls
        }
        if not is_anthropic and tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto" # Explicitly allow model to choose

        response = client.chat.completions.create(**api_params)
        response_message = response.choices[0].message

        tool_calls = getattr(response_message, 'tool_calls', None)

        # --- Handle Tool Calls ---
        if tool_calls:
            # Append the assistant's response (requesting tool use) to messages
            messages.append(response_message.dict(exclude_unset=True)) # Use dict representation

            available_functions = {
                "get_current_time": get_current_time,
            }

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions.get(function_name)

                yield f"\n🔧 Using tool: {function_name}\n", f"tool_call:{function_name}" # Signal tool use

                if function_to_call:
                    # Note: Currently no function arguments are defined/expected
                    # function_args = json.loads(tool_call.function.arguments) # If args were needed
                    function_response = function_to_call()
                    tool_result_str = json.dumps(function_response) # Ensure result is JSON string

                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": tool_result_str,
                    })
                    yield f"📊 Tool Result ({function_name}): {function_response}\n", f"tool_result:{tool_result_str}" # Signal tool result
                else:
                     # Handle case where function is not found (optional)
                    yield f"⚠️ Tool '{function_name}' not found.\n", f"tool_error:not_found"
                    messages.append({ # Still need to provide a response for the tool call
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps({"error": f"Function {function_name} not found"}),
                    })


            # --- Second Call (With Tool Results) ---
            stream_params = {
                "model": model_name,
                "messages": messages,
                "stream": True
            }
            # Don't send tools again if the model already decided to use them
            # if not is_anthropic and tools:
            #     stream_params["tools"] = tools

            stream = client.chat.completions.create(**stream_params)
            full_response = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    yield content, full_response
            # return full_response # Generator implicitly returns None

        # --- No Tool Calls ---
        else:
            # Regular streaming response
            stream_params = {
                "model": model_name,
                "messages": messages,
                "stream": True
            }
            if not is_anthropic and tools: # Still offer tools if none were chosen initially
                stream_params["tools"] = tools

            stream = client.chat.completions.create(**stream_params)
            full_response = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    yield content, full_response
            # return full_response # Generator implicitly returns None

    except Exception as e:
        error_msg = f"Error generating streaming response with {model_name}: {str(e)}"
        logging.error(error_msg, exc_info=True)
        yield error_msg, error_msg # Yield error message
        # return error_msg # Generator implicitly returns None


# --- Content Fetching & Processing Modules ---

def fetch_serper_results(query: str, api_key: str) -> Dict[str, Any]:
    """
    Input: Search query string, Serper API key.
    Process: Performs a search using the Serper Google Search API.
    Output: Dictionary containing the raw search results from Serper API.
    """
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        results = response.json()
        if 'organic' not in results:
            logging.warning(f"Serper results missing 'organic' key for query: {query}")
            # Return a structure consistent with success but empty
            return {"organic": [], "searchParameters": {"q": query}, "type": "empty_result"}
        return results
    except Timeout:
        logging.error(f"Serper API request timed out for query: {query}")
        raise Exception("Search API request timed out.")
    except RequestException as e:
        logging.error(f"Serper API request error for query '{query}': {e}")
        raise Exception(f"Search API error: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode Serper API response for query '{query}': {e}")
        raise Exception("Failed to parse search results.")


def fetch_youtube_transcript(video_url: str) -> Optional[str]:
    """
    Input: YouTube video URL.
    Process: Extracts video ID and fetches the transcript using youtube_transcript_api.
    Output: Transcript text string, or None if transcript is unavailable or an error occurs.
    """
    try:
        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', video_url)
        if not video_id_match:
            logging.warning(f"Could not extract video ID from URL: {video_url}")
            return None

        video_id = video_id_match.group(1)
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([entry['text'] for entry in transcript_list])
        return transcript_text
    except Exception as e:
        # Handles cases like transcripts disabled, video unavailable, etc.
        logging.warning(f"Could not fetch transcript for {video_url}: {e}")
        return None


def clean_web_content(html_content: str, url: str) -> str:
    """
    Input: Raw HTML content string, the source URL.
    Process: Uses newspaper3k and html2text to extract and clean the main text content from HTML.
              Prioritizes newspaper3k, falls back to html2text, then basic BeautifulSoup.
    Output: Cleaned text content string. Returns "No main content found." if extraction fails.
    """
    content = ""

    # Method 1: newspaper3k
    try:
        article = Article(url)
        article.set_html(html_content)
        article.parse()
        if article.text and len(article.text.strip()) > 200: # Check for meaningful content length
            content = article.text.strip()
            logging.info(f"Extracted content using newspaper3k for {url}")
    except Exception as e:
        logging.warning(f"Newspaper3k extraction failed for {url}: {e}")

    # Method 2: html2text (if newspaper failed or produced short content)
    if not content:
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove common noise elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'input']):
                element.decompose()

            # Try finding common main content containers
            main_content_area = (
                soup.find('main') or
                soup.find('article') or
                soup.find('div', id=re.compile(r'main|content', re.I)) or
                soup.find('div', class_=re.compile(r'post|entry|content|article', re.I)) or
                soup.body # Fallback
            )

            if main_content_area:
                h = html2text.HTML2Text()
                h.ignore_links = True # Keep links potentially useful later, maybe make configurable
                h.ignore_images = True
                h.ignore_tables = False
                h.body_width = 0 # Preserve formatting, especially for code
                h.unicode_snob = True # Try to get cleaner unicode chars
                h.escape_snob = True # Escape special markdown chars

                text_content = h.handle(str(main_content_area))
                # Further clean-up: remove excessive newlines
                text_content = re.sub(r'\n\s*\n', '\n\n', text_content).strip()
                if len(text_content) > 100: # Basic check for actual content
                    content = text_content
                    logging.info(f"Extracted content using html2text for {url}")
                else:
                     logging.warning(f"html2text extracted very short content for {url}")

        except Exception as e:
            logging.warning(f"html2text extraction failed for {url}: {e}")

    # Method 3: Basic BeautifulSoup text extraction (last resort)
    if not content:
        try:
            # Reuse soup if created, else parse again
            if 'soup' not in locals():
                 soup = BeautifulSoup(html_content, 'html.parser')
                 for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'input']):
                     element.decompose()

            # Get text from paragraph and heading tags
            text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
            extracted_text = '\n'.join(elem.get_text(strip=True) for elem in text_elements if elem.get_text(strip=True))

            if len(extracted_text) > 100:
                content = extracted_text
                logging.info(f"Extracted content using basic BeautifulSoup for {url}")
            else:
                logging.warning(f"Basic BeautifulSoup extraction yielded very short content for {url}")

        except Exception as e:
            logging.warning(f"Basic BeautifulSoup extraction failed for {url}: {e}")


    return content if content else "No main content found."


def fetch_and_clean_website(url: str, timeout: int = 10) -> Optional[str]:
    """
    Input: Website URL, request timeout duration.
    Process: Fetches the website's HTML content and cleans it using clean_web_content.
    Output: Cleaned text content string, or None if fetching/cleaning fails or content is empty.
    """
    try:
        headers = { # Add a user-agent to mimic a browser
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status() # Check for HTTP errors

        if response.status_code == 200 and response.content:
            # Decode content carefully
            content_type = response.headers.get('content-type', '').lower()
            encoding = response.encoding if response.encoding else response.apparent_encoding
            try:
                html_content = response.content.decode(encoding if encoding else 'utf-8', errors='replace')
            except (UnicodeDecodeError, LookupError):
                 html_content = response.text # Fallback to requests' auto-decoded text

            cleaned_content = clean_web_content(html_content, url)
            if cleaned_content and cleaned_content != "No main content found.":
                return cleaned_content
            else:
                logging.warning(f"Cleaning returned no content for {url}")
                return None
        else:
            logging.warning(f"Received status code {response.status_code} for {url}")
            return None

    except Timeout:
        logging.warning(f"Request timed out for {url}")
        return None
    except RequestException as e:
        logging.warning(f"Failed to fetch {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error fetching/cleaning {url}: {e}", exc_info=True)
        return None


# --- Search Orchestration Module ---

def gather_content_from_results(
    organic_results: List[Dict[str, Any]],
    target_sources: int,
    include_youtube: bool,
    max_attempts: int,
    progress_callback: Optional[callable] = None
) -> Tuple[List[Dict], List[Dict], List[str], List[str], Dict]:
    """
    Input: List of organic search results from Serper, target number of sources,
           YouTube inclusion flag, max processing attempts, optional progress callback.
    Process: Iterates through search results, fetches and cleans website content,
             fetches YouTube transcripts, until target source count or max attempts is reached.
    Output: Tuple containing:
            - website_contents (List[Dict]): Successfully processed websites.
            - youtube_contents (List[Dict]): Successfully processed YouTube videos.
            - blocked_urls (List[str]): URLs that failed or were blocked.
            - failed_youtube_urls (List[str]): YouTube URLs that failed.
            - access_stats (Dict): Statistics about the fetching process.
    """
    website_contents = []
    youtube_contents = []
    blocked_urls = []
    failed_youtube_urls = []

    successful_website_count = 0
    successful_youtube_count = 0
    blocked_website_count = 0
    failed_youtube_count = 0

    processed_urls = set()
    total_attempted = 0

    for result in organic_results:
        if total_attempted >= max_attempts:
            logging.warning("Reached max processing attempts.")
            break

        total_successful = successful_website_count + successful_youtube_count
        if total_successful >= target_sources:
            break # Target reached

        url = result.get('link')
        title = result.get('title', 'Untitled')

        if not url or url in processed_urls:
            continue # Skip if no URL or already processed

        processed_urls.add(url)
        total_attempted += 1

        # Calculate progress
        progress = 0.2 + (0.6 * min(total_successful, target_sources) / target_sources) # Progress based on successful finds

        is_youtube = 'youtube.com/watch?v=' in url or 'youtu.be/' in url

        try:
            if is_youtube and include_youtube:
                if progress_callback:
                    progress_callback(f"Processing YouTube video {successful_youtube_count + 1}/{target_sources} (Attempt {total_attempted})...", progress)

                transcript = fetch_youtube_transcript(url)
                if transcript:
                    youtube_contents.append({
                        'title': title,
                        'link': url,
                        'content': transcript
                    })
                    successful_youtube_count += 1
                else:
                    failed_youtube_urls.append(url)
                    failed_youtube_count += 1

            elif not is_youtube:
                if progress_callback:
                    progress_callback(f"Processing website {successful_website_count + 1}/{target_sources} (Attempt {total_attempted})...", progress)

                content = fetch_and_clean_website(url)
                if content:
                    website_contents.append({
                        'title': title,
                        'link': url,
                        'content': content
                    })
                    successful_website_count += 1
                else:
                    blocked_urls.append(url)
                    blocked_website_count += 1
            # else: # Is YouTube but include_youtube is False
            #     pass # Just skip

        except Exception as e:
            # Catch unexpected errors during processing a specific result
            logging.error(f"Error processing result {url}: {e}", exc_info=True)
            if is_youtube:
                failed_youtube_urls.append(url)
                failed_youtube_count += 1
            else:
                blocked_urls.append(url)
                blocked_website_count += 1
            continue # Move to the next result

    access_stats = {
        "target_sources": target_sources,
        "total_attempted": total_attempted,
        "successful_websites": successful_website_count,
        "successful_youtube": successful_youtube_count,
        "blocked_websites": blocked_website_count, # Renamed for clarity
        "failed_youtube": failed_youtube_count,
        "blocked_urls": blocked_urls,
        "failed_youtube_urls": failed_youtube_urls
    }

    return website_contents, youtube_contents, blocked_urls, failed_youtube_urls, access_stats


def format_content_for_llm(
    website_contents: List[Dict],
    youtube_contents: List[Dict]
) -> Tuple[str, str, Dict[str, str]]:
    """
    Input: Lists of processed website and YouTube content dictionaries.
    Process: Formats the content into a single string for the LLM prompt,
             creates a combined string for display, and generates a source mapping.
    Output: Tuple containing:
            - formatted_content (str): Content formatted with XML-like tags for the LLM.
            - combined_content_display (str): Content formatted for user display.
            - source_mapping (Dict[str, str]): Maps source IDs (e.g., "website_1") to URLs.
    """
    formatted_content = ""
    combined_content_display_parts = []
    source_mapping = {}

    # Add website content
    for i, web_content in enumerate(website_contents, 1):
        source_id = f"website_{i}"
        source_mapping[source_id] = web_content['link']
        formatted_content += f"<{source_id}>\nTitle: {web_content['title']}\nURL: {web_content['link']}\nSource ID: {source_id}\n\n{web_content['content']}\n</{source_id}>\n\n"
        display_entry = f"--- Source: Website {i} ---\nTitle: {web_content['title']}\nURL: {web_content['link']}\nSource ID: {source_id}\n\n{web_content['content']}"
        combined_content_display_parts.append(display_entry)

    # Add YouTube content
    for i, yt_content in enumerate(youtube_contents, 1):
        source_id = f"youtube_video_{i}"
        source_mapping[source_id] = yt_content['link']
        formatted_content += f"<{source_id}>\nTitle: {yt_content['title']}\nURL: {yt_content['link']}\nSource ID: {source_id}\n\n{yt_content['content']}\n</{source_id}>\n\n"
        display_entry = f"--- Source: YouTube {i} ---\nTitle: {yt_content['title']}\nURL: {yt_content['link']}\nSource ID: {source_id}\n\n{yt_content['content']}"
        combined_content_display_parts.append(display_entry)

    combined_content_display = "\n\n".join(combined_content_display_parts)
    return formatted_content.strip(), combined_content_display, source_mapping


def generate_search_summary(
        query: str,
        formatted_content: str,
        source_mapping: Dict[str, str],
        model_choice: str,
        client: Optional[OpenAI] = None
    ) -> str:
    """
    Input: User query, LLM-formatted content, source mapping, chosen model name, optional OpenAI client.
    Process: Creates the final prompt and calls the LLM to generate a summary.
    Output: AI-generated summary string.
    """
    today_date = datetime.now().strftime("%Y-%m-%d")
    source_references = "\n".join([f"{source_id}: {url}" for source_id, url in source_mapping.items()])

    system_prompt = """<role>You are a world class search engine and summarizer.
Based on the provided web content and YouTube transcripts, create a comprehensive and accurate summary to answer the user's query.</role>
"""

    prompt_template = f"""<instructions>
The content below is organized with tags indicating the source:
- <website_1>, <website_2>, etc.: Content from different websites.
- <youtube_video_1>, <youtube_video_2>, etc.: Transcripts from different YouTube videos.

Source Reference Map (Use these URLs for citations):
{source_references}

Today's Date: {today_date}

User Query: {query}

Task:
Carefully analyze the user query and the provided content to generate a world-class answer.

Output Format:
1.  **Short Answer:** A concise summary addressing the core of the query.
2.  **Detailed Answer:** A more comprehensive explanation, elaborating on key points.
    - Use clear titles (e.g., `### Short Answer`, `### Detailed Answer`).
    - Structure the detailed answer logically using headings and bullet points where appropriate.
    - Include examples if relevant and supported by the content.

Content Guidelines:
- Focus on high quality and accuracy. Synthesize information from the provided sources.
- Filter out irrelevant information. Compare sources if they conflict, and note discrepancies if significant.
- If the provided content is insufficient to answer the query fully, state that clearly. Do *not* invent information.
- Be objective and neutral in tone.

**CRITICAL Citation Requirements:**
1.  Cite sources using Markdown links: `[descriptive text](URL)`.
2.  Use the actual URLs from the 'Source Reference Map' above for the links.
3.  Integrate citations naturally within the text where information is used.
4.  **NEVER** use the internal source IDs like `[website_1]` or `[youtube_video_1]` in your final output. Cite with the URL.
    *   *Example Correct:* "According to [this analysis](https://example.com/analysis)..."
    *   *Example Incorrect:* "According to [website_1]..."

**Special Instructions for Coding-Related Queries:**
- If multiple code snippets, syntaxes, or approaches are found for the same task, present them clearly.
- Highlight potential differences (e.g., old vs. new syntax, different libraries).
- Note if source dates suggest one approach is more current. Advise the user to test the code.
- Format code blocks correctly using Markdown triple backticks (```).

*Self-Correction/Improvement:* Review your generated response before finalizing. Ensure all instructions, especially citation rules, are followed meticulously. A high-quality, well-cited answer is paramount.
</instructions>

<content>
{formatted_content}
</content>
"""
    try:
        summary = get_llm_response(
            prompt=prompt_template,
            model_name=model_choice,
            system_prompt=system_prompt,
            client=client # Pass client if provided
        )
        return summary
    except Exception as e:
        logging.error(f"Summary generation failed: {e}", exc_info=True)
        raise Exception(f"Summary generation error: {str(e)}")


def search_and_summarize_orchestrator(
    query: str,
    model_choice: str,
    search_depth: str, # "fast" or "deep"
    include_youtube: bool,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Input: Search query, model choice, search depth, YouTube inclusion flag, optional progress callback.
    Process: Orchestrates the entire search and summarization process:
             1. Fetches initial search results via Serper.
             2. Gathers content from results.
             3. (Optional) Fetches additional results if needed.
             4. Formats content for LLM.
             5. Generates summary using LLM.
             6. Compiles final results object.
    Output: Dictionary containing the summary, source details, stats, and raw results.
    """
    if progress_callback: progress_callback("Initiating search...", 0.05)

    client = get_openai_client() # Create one client for potential reuse
    target_sources = 5 if search_depth == "fast" else 10
    max_attempts = 15 if search_depth == "fast" else 30 # Allow more attempts for deep search

    # 1. Initial Search
    if progress_callback: progress_callback("Fetching search results...", 0.1)
    try:
        initial_search_results = fetch_serper_results(query, st.secrets["SERPER_API_KEY"])
        organic_results = initial_search_results.get('organic', [])
    except Exception as e:
        logging.error(f"Initial search failed: {e}", exc_info=True)
        raise # Re-raise to be caught by the UI

    if not organic_results:
        logging.warning(f"No organic results found for query: {query}")
        # Return a result indicating no sources found
        return {
            "summary": "No search results found for your query.",
            "combined_content_display": "N/A",
            "source_mapping": {},
            "access_stats": { "target_sources": target_sources, "total_attempted": 0, "successful_websites": 0, "successful_youtube": 0, "blocked_websites": 0, "failed_youtube": 0, "blocked_urls": [], "failed_youtube_urls": [] },
            "serper_results": initial_search_results,
            "word_count": 0,
            "status": "No Results"
        }


    # 2. Gather Content (Initial Pass)
    if progress_callback: progress_callback("Processing results...", 0.2)
    website_contents, youtube_contents, blocked_urls, failed_youtube_urls, access_stats = gather_content_from_results(
        organic_results, target_sources, include_youtube, max_attempts, progress_callback
    )

    total_successful = access_stats['successful_websites'] + access_stats['successful_youtube']
    additional_search_results = None # Initialize

    # 3. (Optional) Additional Search if needed
    if total_successful < target_sources and access_stats['total_attempted'] < max_attempts:
        if progress_callback: progress_callback(f"Found {total_successful}/{target_sources}. Fetching more results...", 0.7)
        try:
            # Simple refinement - could be more sophisticated
            refined_query = f"{query} detailed information"
            additional_search_results = fetch_serper_results(refined_query, st.secrets["SERPER_API_KEY"])
            additional_organic = additional_search_results.get('organic', [])

            if additional_organic:
                 # Pass only the *remaining* target and attempts
                remaining_target = target_sources - total_successful
                remaining_attempts = max_attempts - access_stats['total_attempted']

                # Filter out already processed URLs before passing to gather_content
                processed_in_first_pass = set(wc['link'] for wc in website_contents) | set(yc['link'] for yc in youtube_contents) | set(blocked_urls) | set(failed_youtube_urls)
                new_organic_results = [res for res in additional_organic if res.get('link') not in processed_in_first_pass]


                if new_organic_results and remaining_target > 0 and remaining_attempts > 0:
                    add_websites, add_youtube, add_blocked, add_failed_yt, add_stats = gather_content_from_results(
                        new_organic_results, remaining_target, include_youtube, remaining_attempts, progress_callback
                    )
                    # Combine results
                    website_contents.extend(add_websites)
                    youtube_contents.extend(add_youtube)
                    blocked_urls.extend(add_blocked)
                    failed_youtube_urls.extend(add_failed_yt)
                    # Update stats carefully
                    access_stats['successful_websites'] += add_stats['successful_websites']
                    access_stats['successful_youtube'] += add_stats['successful_youtube']
                    access_stats['blocked_websites'] += add_stats['blocked_websites']
                    access_stats['failed_youtube'] += add_stats['failed_youtube']
                    access_stats['total_attempted'] += add_stats['total_attempted'] # Add attempts from the second round
                    access_stats['blocked_urls'].extend(add_stats['blocked_urls'])
                    access_stats['failed_youtube_urls'].extend(add_stats['failed_youtube_urls'])

        except Exception as e:
            logging.warning(f"Additional search or processing failed: {e}")
            # Continue with the results obtained so far

    # Check if any content was gathered
    if not website_contents and not youtube_contents:
         return {
            "summary": "Could not retrieve content from any sources for your query.",
            "combined_content_display": "N/A",
            "source_mapping": {},
            "access_stats": access_stats,
            "serper_results": {"initial": initial_search_results, "additional": additional_search_results},
            "word_count": 0,
            "status": "No Content Found"
        }


    # 4. Format Content
    if progress_callback: progress_callback("Formatting content for AI...", 0.75)
    formatted_content, combined_content_display, source_mapping = format_content_for_llm(
        website_contents, youtube_contents
    )

    # 5. Generate Summary
    if progress_callback: progress_callback("Generating summary...", 0.8)
    try:
        summary = generate_search_summary(
            query, formatted_content, source_mapping, model_choice, client
        )
    except Exception as e:
        logging.error(f"Summary generation failed: {e}", exc_info=True)
        raise # Re-raise to be caught by the UI

    if progress_callback: progress_callback("Complete!", 1.0)

    # 6. Compile Final Results
    word_count = len(combined_content_display.split())
    final_results = {
        "summary": summary,
        "combined_content_display": combined_content_display,
        "source_mapping": source_mapping,
        "access_stats": access_stats,
        "serper_results": {"initial": initial_search_results, "additional": additional_search_results},
        "word_count": word_count,
        "status": "Success"
    }
    return final_results


# --- Utilities ---

def get_current_time() -> str:
    """
    Input: None
    Process: Gets the current time.
    Output: Formatted current time string (e.g., "03:30 PM, July 26, 2024").
    """
    return datetime.now().strftime("%I:%M %p, %B %d, %Y")

# Tool definition for chat function calling
CHAT_TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current date and time.",
        "parameters": { # Even if no params, structure is needed
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}]


# --- UI Page Modules ---

def home_page():
    """ Renders the Home page content. """
    st.markdown("## Welcome to My Portfolio!")
    st.markdown("""
    I am a passionate developer exploring the intersection of AI, web technologies, and data.
    This site showcases some experimental tools built with Streamlit and various APIs.

    Navigate using the sidebar to explore:
    - **Search:** An AI-powered search assistant that synthesizes information from multiple web and YouTube sources.
    - **Chat:** A conversational AI interface using different language models.
    """)

def search_page():
    """ Renders the Search page UI and handles search logic. """
    st.title("AI-Powered Search Assistant 🔍")
    st.write("Get comprehensive answers synthesized from multiple web sources and YouTube videos.")

    # Initialize session state for search page specific items if not present
    if 'include_youtube' not in st.session_state:
        st.session_state.include_youtube = True
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'selected_history_item_search' not in st.session_state: # Use specific key
        st.session_state.selected_history_item_search = None

    # --- Search Input Area ---
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("🔍 Enter your search query:",
                              placeholder="e.g., Latest advancements in LLM function calling",
                              key="search_query_input")
    with col2:
        search_type = st.selectbox("Search Depth:",
                                   ["Quick (approx 5 sources)", "Deep (approx 10 sources)"],
                                   index=0, # Default to Quick
                                   key="search_depth_select",
                                   help="Deep search attempts to find more sources but takes longer.")

    col3, col4, col5 = st.columns([2, 2, 1])
    with col3:
        # Find index safely, default to 0 if not found
        try:
            default_model_index = MODEL_CONFIG.index("google/gemini-2.0-flash-lite-001")
        except ValueError:
            default_model_index = 0
        model_choice = st.selectbox(
            "AI Model for Summarization:",
            MODEL_CONFIG,
            index=default_model_index,
            key="search_model_select",
            help="Select the AI model to generate the summary."
        )
    with col4:
        # Use a unique key and manage state explicitly
        include_youtube_checkbox = st.checkbox("Include YouTube Transcripts",
                                               value=st.session_state.include_youtube,
                                               key="include_youtube_checkbox_search",
                                               help="Include relevant YouTube video transcripts in the search.")
        # Update session state based on checkbox interaction
        if include_youtube_checkbox != st.session_state.include_youtube:
            st.session_state.include_youtube = include_youtube_checkbox
            # No rerun needed here, state is updated for the *next* search

    with col5:
        search_button = st.button("🔎 Search", use_container_width=True, key="search_button")

    # --- Search Execution and Results Display ---
    results_container = st.container() # Placeholder for results

    if search_button and query:
        search_depth_val = "deep" if "Deep" in search_type else "fast"
        try:
            with st.spinner("Performing search and generating summary..."): # Use spinner for better UX
                 # Use status for progress updates if needed, spinner is simpler
                 with st.status("🔍 Searching and Processing...", expanded=True) as status_indicator:
                    search_progress_text = st.empty()
                    search_progress_bar = st.progress(0.0)

                    def update_search_progress(message, progress_value):
                        status_indicator.update(label=message)
                        search_progress_text.text(message)
                        search_progress_bar.progress(progress_value)

                    # Call the orchestrator function
                    search_results_data = search_and_summarize_orchestrator(
                        query,
                        model_choice,
                        search_depth_val,
                        st.session_state.include_youtube, # Use current state
                        update_search_progress
                    )

                    status_indicator.update(label="Search complete!", state="complete")


            # Store results in history before displaying
            st.session_state.search_history.append({
                "query": query,
                "model_used": model_choice,
                "search_depth": search_depth_val,
                "included_youtube": st.session_state.include_youtube,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **search_results_data # Unpack the results dict
            })
            # Set the latest search as the selected one to display immediately
            st.session_state.selected_history_item_search = len(st.session_state.search_history) - 1
            st.rerun() # Rerun to display the new result from history


        except Exception as e:
            st.error(f"An error occurred during the search process: {str(e)}")
            logging.error(f"Search page error for query '{query}': {str(e)}", exc_info=True)

    # --- Display Selected Search Result (from history) ---
    if st.session_state.selected_history_item_search is not None:
         # Check index validity
        if 0 <= st.session_state.selected_history_item_search < len(st.session_state.search_history):
            selected_index = st.session_state.selected_history_item_search
            # Access history using the index (newest is at the end)
            search_data = st.session_state.search_history[selected_index]

            with results_container:
                st.markdown(f"### Results for: \"{search_data['query']}\"")
                st.caption(f"Searched on: {search_data['timestamp']} | Model: {search_data['model_used']} | Depth: {search_data['search_depth']} | YouTube: {'Included' if search_data['included_youtube'] else 'Excluded'}")

                access_stats = search_data['access_stats']
                total_successful = access_stats['successful_websites'] + access_stats['successful_youtube']
                target = access_stats['target_sources']
                success_percentage = round((total_successful / target) * 100) if target > 0 else 0

                if search_data['status'] == "Success":
                    if total_successful >= target:
                        st.success(f"Successfully gathered content from {total_successful} sources ({access_stats['successful_websites']} websites, {access_stats['successful_youtube']} YouTube).")
                    else:
                        st.warning(f"Found {total_successful} of {target} target sources ({success_percentage}%). {access_stats['blocked_websites']} websites blocked/inaccessible, {access_stats['failed_youtube']} YouTube videos failed.")
                elif search_data['status'] == "No Content Found":
                     st.error("Could not retrieve content from any sources for this query.")
                elif search_data['status'] == "No Results":
                     st.error("No search results found for this query.")


                # Display content only if successful or partially successful
                if search_data['status'] in ["Success", "No Content Found"]: # Show tabs even if no content retrieved, but summary might be empty
                    tabs = st.tabs(["📝 Summary", "🔍 Sources", "📊 Access Stats", "🌐 Raw Search API", "🔧 Debug Info"])

                    with tabs[0]: # Summary
                        if search_data['summary']:
                            st.markdown(search_data['summary'])
                        else:
                            st.info("Summary could not be generated (likely due to lack of content).")

                    with tabs[1]: # Sources
                        st.text_area("Combined Source Content", search_data['combined_content_display'], height=400, key=f"sources_{selected_index}")

                    with tabs[2]: # Access Stats
                        st.write(f"**Target Sources:** {target}")
                        st.write(f"**Total Attempts:** {access_stats['total_attempted']}")
                        st.write(f"**Successfully Accessed:** {total_successful} ({success_percentage}%)")
                        st.write(f"- Websites: {access_stats['successful_websites']}")
                        st.write(f"- YouTube Videos: {access_stats['successful_youtube']}")
                        st.write(f"**Failed/Blocked:**")
                        st.write(f"- Websites: {access_stats['blocked_websites']}")
                        st.write(f"- YouTube Videos: {access_stats['failed_youtube']}")

                        if access_stats['blocked_urls']:
                            with st.expander("Blocked/Inaccessible Website URLs"):
                                st.json(access_stats['blocked_urls'])
                        if access_stats['failed_youtube_urls']:
                            with st.expander("Failed YouTube URLs"):
                                st.json(access_stats['failed_youtube_urls'])

                    with tabs[3]: # Serper API
                        st.json(search_data['serper_results'])

                    with tabs[4]: # Debug
                        st.write("#### Source ID to URL Mapping")
                        if search_data['source_mapping']:
                             st.json(search_data['source_mapping'])
                             # Show citation example using the first source
                             first_id = list(search_data['source_mapping'].keys())[0]
                             first_url = search_data['source_mapping'][first_id]
                             st.write("---")
                             st.write("**Proper Citation Example:**")
                             st.markdown(f"Text referencing the source should look like: `[description]({first_url})`")
                             st.markdown(f"**Incorrect:** Using `[{first_id}]`")
                        else:
                            st.info("No sources were successfully processed to create a mapping.")

                # Button to clear the displayed result area
                if st.button("Clear Displayed Result", key=f"clear_search_{selected_index}"):
                    st.session_state.selected_history_item_search = None
                    st.rerun()
        else:
             # Handle invalid index if necessary (e.g., history was cleared)
             st.session_state.selected_history_item_search = None
             st.warning("Selected history item is no longer valid. Please select another.")
             # Optionally rerun: st.rerun()


    # --- Search History Display ---
    if st.session_state.search_history:
        with st.expander("📚 Search History", expanded=False):
            # Display history items (newest first)
            for i, search in enumerate(reversed(st.session_state.search_history)):
                 history_index = len(st.session_state.search_history) - 1 - i # Original index
                 col1_hist, col2_hist = st.columns([4, 1])
                 with col1_hist:
                     st.markdown(f"**{search['timestamp']}**: `{search['query']}`")
                     st.caption(f"Model: {search['model_used']}, Depth: {search['search_depth']}, YouTube: {'Yes' if search['included_youtube'] else 'No'}")
                 with col2_hist:
                     # Highlight the currently viewed item
                     button_type = "primary" if history_index == st.session_state.selected_history_item_search else "secondary"
                     if st.button(f"View Result #{i+1}", key=f"view_search_{history_index}", use_container_width=True, type=button_type):
                         st.session_state.selected_history_item_search = history_index
                         st.rerun() # Rerun to display the selected item above
                 st.divider()

            if st.button("Clear Search History", key="clear_search_history_button"):
                 st.session_state.search_history = []
                 st.session_state.selected_history_item_search = None
                 st.rerun()


def chat_page():
    """ Renders the Chat page UI and handles chat interactions. """
    st.title("AI Chat 💬")

    # Initialize chat history in session state if not present
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = {} # Store multiple sessions {session_id: {history: [], model: ""}}
    if 'current_chat_session_id' not in st.session_state:
        st.session_state.current_chat_session_id = None
    if 'chat_session_counter' not in st.session_state:
        st.session_state.chat_session_counter = 0 # To generate unique IDs

    # --- Session Management ---
    st.sidebar.title("Chat Sessions")

    # Button to start a new chat
    if st.sidebar.button("➕ New Chat"):
        st.session_state.chat_session_counter += 1
        new_session_id = f"session_{st.session_state.chat_session_counter}"
        st.session_state.chat_sessions[new_session_id] = {
            "history": [{"role": "assistant", "content": "👋 Hi! How can I help you today?"}],
            "model": MODEL_CONFIG[MODEL_CONFIG.index("google/gemini-2.5-flash-preview")] # Default model for new chat
        }
        st.session_state.current_chat_session_id = new_session_id
        st.rerun()

    # List existing chat sessions
    session_ids = list(st.session_state.chat_sessions.keys())
    session_display_names = {}
    for sid in session_ids:
         # Try to get the first user message as the name, fallback to ID
        history = st.session_state.chat_sessions[sid]['history']
        first_user_message = next((msg['content'] for msg in history if msg['role'] == 'user'), None)
        display_name = f"{first_user_message[:30]}..." if first_user_message else sid
        session_display_names[sid] = display_name

    # Select current session
    st.session_state.current_chat_session_id = st.sidebar.radio(
        "Select Chat:",
        options=session_ids,
        format_func=lambda sid: session_display_names.get(sid, sid),
        key="chat_session_selector",
        index=session_ids.index(st.session_state.current_chat_session_id) if st.session_state.current_chat_session_id in session_ids else 0
    )

    # --- Chat Interface for the selected session ---
    if st.session_state.current_chat_session_id and st.session_state.current_chat_session_id in st.session_state.chat_sessions:
        current_session = st.session_state.chat_sessions[st.session_state.current_chat_session_id]
        chat_history = current_session["history"]

        # Model selection for the current chat
        try:
            current_model_index = MODEL_CONFIG.index(current_session["model"])
        except ValueError:
            current_model_index = MODEL_CONFIG.index("google/gemini-2.5-flash-preview") # Fallback

        selected_model = st.selectbox(
            "Select Model for this Chat:",
            MODEL_CONFIG,
            index=current_model_index,
            key=f"model_select_{st.session_state.current_chat_session_id}" # Key specific to session
        )
        # Update model for the session if changed
        if selected_model != current_session["model"]:
            current_session["model"] = selected_model
            # No rerun needed, just updates the state for the next message

        # Display chat history
        chat_container = st.container() # To hold messages
        with chat_container:
            for message in chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Type your message here...", key=f"chat_input_{st.session_state.current_chat_session_id}"):
            # Add user message to the current session's history
            chat_history.append({"role": "user", "content": prompt})

            # Rerun to display the user message immediately
            st.rerun()

        # --- Generate response if the last message is from the user ---
        if chat_history and chat_history[-1]["role"] == "user":
            user_prompt = chat_history[-1]["content"] # Get the latest user prompt

            # Display user message (already done by rerun, but good practice)
            # with st.chat_message("user"):
            #    st.markdown(user_prompt)

            # Prepare message history for API (excluding initial greeting)
            api_messages = [msg for msg in chat_history if not (msg == chat_history[0] and msg["role"] == "assistant")]

            # Display assistant response with streaming
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response_content = ""
                try:
                    # Use the streaming function
                    response_generator = get_streaming_llm_response(
                        messages=api_messages,
                        model_name=current_session["model"], # Use session's model
                        tools=CHAT_TOOLS # Pass defined tools
                    )

                    for chunk, current_response_or_signal in response_generator:
                        # Check if it's a signal or actual content
                        if isinstance(current_response_or_signal, str) and current_response_or_signal.startswith(("tool_call:", "tool_result:", "tool_error:")):
                             # Display tool usage signals/results if desired
                             # st.write(chunk) # Optional: show tool activity in chat
                             pass # Or just ignore these signals in the main chat display
                        else:
                            # It's content, update the placeholder
                            full_response_content = current_response_or_signal
                            message_placeholder.markdown(full_response_content + "▌") # Blinking cursor

                    # Final display without cursor
                    message_placeholder.markdown(full_response_content)

                    # Add final assistant response to history
                    chat_history.append({"role": "assistant", "content": full_response_content})
                    # No rerun needed here, message is already displayed

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    message_placeholder.error(error_msg)
                    chat_history.append({"role": "assistant", "content": f"Error: {error_msg}"}) # Log error in history
                    logging.error(f"Chat page streaming error: {e}", exc_info=True)

        # Option to clear the current chat session
        if st.button("Clear Current Chat", key=f"clear_chat_{st.session_state.current_chat_session_id}"):
            current_session["history"] = [{"role": "assistant", "content": "Chat cleared. How can I help?"}]
            st.rerun()

        # Option to delete the current chat session
        if st.sidebar.button(f"🗑️ Delete '{session_display_names[st.session_state.current_chat_session_id]}'", key=f"delete_chat_{st.session_state.current_chat_session_id}"):
             del st.session_state.chat_sessions[st.session_state.current_chat_session_id]
             # Select the first available session or set to None if empty
             available_sessions = list(st.session_state.chat_sessions.keys())
             st.session_state.current_chat_session_id = available_sessions[0] if available_sessions else None
             st.rerun()

    else:
        st.info("Click 'New Chat' in the sidebar to start a conversation.")


# --- Main Application ---

def main():
    """ Main function to run the Streamlit application and handle navigation. """
    # Initialize session state for navigation if it doesn't exist
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home' # Default page

    # Sidebar for navigation
    with st.sidebar:
        st.markdown("---") # Separator before navigation
        st.title("Navigation")
        # Use st.session_state.current_page to set the default selected radio button
        pages = ["Home", "Search", "Chat"]
        try:
            current_page_index = pages.index(st.session_state.current_page)
        except ValueError:
            current_page_index = 0 # Default to Home if state is invalid

        selected_page = st.radio(
            "Go to",
            options=pages,
            key="navigation_radio",
            index=current_page_index
        )

        # Update state if selection changes
        if selected_page != st.session_state.current_page:
            st.session_state.current_page = selected_page
            # Clear selected history item when switching pages to avoid confusion
            st.session_state.selected_history_item_search = None
            st.rerun() # Rerun to load the new page

    # Display the selected page title dynamically
    # st.title(f"{st.session_state.current_page} Page") # Title is set within each page function now

    # Page content based on selection
    if st.session_state.current_page == "Home":
        home_page()
    elif st.session_state.current_page == "Search":
        search_page()
    elif st.session_state.current_page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()