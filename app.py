"""
Main application file for the portfolio website.
Handles navigation and main layout structure.
Logic:
- Manages page navigation through session state
- Provides search functionality with web and YouTube sources
- Uses synchronous functions for operations
- Uses OpenRouter API for AI model access
"""

import streamlit as st
import requests
import json
from datetime import datetime
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from requests.exceptions import Timeout
import re
import logging
from openai import OpenAI
# Import newspaper library for better content extraction
from newspaper import Article
import html2text

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


def get_response(prompt: str, model_name: str = "openai/gpt-4o", system_prompt: str = None):
    """
    Input: prompt, optional model name, and optional system prompt
    Process: Generates response using OpenRouter API with specified model
    Output: Generated text response
    """
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=st.secrets["OPENROUTER_API_KEY"],
        )
        
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        
        return completion.choices[0].message.content
            
    except Exception as e:
        raise Exception(f"Error generating response with {model_name}: {str(e)}")


#SEARCH PAGE:
def clean_content(soup, url):
    """
    Input: BeautifulSoup object and the URL
    Process: Uses multiple extraction methods to get the best content, preserving code blocks
    Output: Cleaned text content
    """
    content = ""
    html_content = str(soup)
    
    # Method 1: Try newspaper3k extraction first (good for articles)
    try:
        article = Article(url)
        # Set html manually since we already have it
        article.set_html(html_content)
        article.parse()
        if article.text and len(article.text) > 300:  # Only use if we got substantial content
            content = article.text
    except Exception as e:
        logging.error(f"Newspaper extraction error: {str(e)}")
    
    # Method 2: Use html2text directly on the main content area
    if not content or len(content) < 300:
        try:
            # Remove unnecessary elements first
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Try to find the main content area
            main_content = (
                soup.find('div', id='main-outlet') or  
                soup.find('main') or
                soup.find('article') or
                soup.find('div', class_='content') or
                soup.find('div', id='content') or
                soup.body  # Fallback to entire body if no specific content area found
            )
            
            if main_content:
                # Create an HTML2Text instance for converting HTML to markdown
                h = html2text.HTML2Text()
                h.ignore_links = False
                h.ignore_images = True 
                h.ignore_tables = False
                h.preserve_newlines = True
                # This is crucial for code blocks - prevents wrapping
                h.body_width = 0
                
                # Convert HTML to markdown-like text (preserves code blocks)
                content = h.handle(str(main_content))
        except Exception as e:
            logging.error(f"HTML2Text extraction error: {str(e)}")
    
    # Method 3: Fall back to a simpler BeautifulSoup extraction if others failed
    if not content or len(content) < 300:
        try:
            # Find the main content again (since we already processed soup)
            main_content = (
                soup.find('div', id='main-outlet') or  
                soup.find('main') or
                soup.find('article') or
                soup.find('div', class_='content') or
                soup.find('div', id='content') or
                soup.body
            )
            
            if main_content:
                # Extract code blocks first
                code_blocks = []
                for code_elem in main_content.find_all(['pre', 'code']):
                    code_text = code_elem.get_text(strip=False)  # Preserve whitespace in code
                    if code_text:
                        code_blocks.append(f"```\n{code_text}\n```")
                
                # Extract text content
                text_elements = []
                for elem in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'blockquote']):
                    text = elem.get_text(strip=True)
                    if text:
                        text_elements.append(text)
                
                if text_elements:
                    content = '\n\n'.join(text_elements)
                
                # Add extracted code blocks
                if code_blocks:
                    content += '\n\n' + '\n\n'.join(code_blocks)
        except Exception as e:
            logging.error(f"BeautifulSoup extraction error: {str(e)}")
    
    return content if content else "No main content found."

def search_and_summarize(query, model_choice, search_type, include_youtube=True, progress_callback=None):
    """
    Input: Search query, model choice, search type, YouTube inclusion flag, and optional progress callback
    Process: Searches web and YouTube for content and summarizes results, continuing until target number of sources is reached
    Output: Number of sources used, blocked websites, combined content, AI summary, raw Serper API response, and word count
    """
    if progress_callback:
        progress_callback("Searching for web content...", 0.1)
    
    # Serper API call
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': st.secrets["SERPER_API_KEY"],
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        search_results = response.json()
    except Exception as e:
        raise Exception(f"Search API error: {str(e)}")

    if 'organic' not in search_results:
        raise Exception("Invalid search results format")

    if progress_callback:
        progress_callback("Processing search results...", 0.2)

    # Lists to store content from different sources
    website_contents = []
    youtube_contents = []
    
    # Counters
    successful_website_count = 0
    successful_youtube_count = 0
    blocked_websites_count = 0
    failed_youtube_count = 0
    
    # Lists to store blocked/failed URLs
    blocked_websites = []
    failed_youtube = []
    
    # Define target number of successful sources (websites + YouTube videos)
    target_sources = 5 if search_type == "fast" else 10
    
    # Safety limit to prevent infinite processing if most sites are blocked
    max_attempts = 30
    total_attempted = 0
    processed_urls = set()  # Track URLs we've already processed to avoid duplicates

    # Process search results until we reach target number of sources or max attempts
    for rank, result in enumerate(search_results['organic'], 1):
        # Skip if we've already processed this URL
        if result['link'] in processed_urls:
            continue
            
        processed_urls.add(result['link'])
        total_attempted += 1
        
        # Safety check to avoid infinite loops
        if total_attempted >= max_attempts:
            break
            
        # Calculate progress percentage (adjust to account for more possible attempts)
        progress = 0.2 + (0.6 * min(total_attempted, target_sources) / target_sources)
        
        try:
            if 'youtube.com' in result['link'] and include_youtube:
                if progress_callback:
                    progress_callback(f"Processing YouTube video {successful_youtube_count + 1} (attempt {total_attempted})", progress)
                
                video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', result['link'])
                if video_id_match:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id_match.group(1))
                    transcript_text = ' '.join([entry['text'] for entry in transcript])
                    youtube_contents.append({
                        'title': result.get('title', 'YouTube Video'),
                        'link': result['link'],
                        'content': transcript_text
                    })
                    successful_youtube_count += 1
                else:
                    failed_youtube.append(result['link'])
                    failed_youtube_count += 1
            else:
                if progress_callback:
                    progress_callback(f"Processing website {successful_website_count + 1} (attempt {total_attempted})", progress)
                
                try:
                    page_response = requests.get(result['link'], timeout=5)
                    page_response.raise_for_status()  # Check for HTTP errors
                    
                    if page_response.status_code == 200:
                        soup = BeautifulSoup(page_response.content, 'html.parser')
                        content = clean_content(soup, result['link'])
                        if content and content != "No main content found.":
                            website_contents.append({
                                'title': result.get('title', 'Website'),
                                'link': result['link'],
                                'content': content
                            })
                            successful_website_count += 1
                        else:
                            blocked_websites.append(result['link'])
                            blocked_websites_count += 1
                    else:
                        blocked_websites.append(result['link'])
                        blocked_websites_count += 1
                except requests.exceptions.RequestException as e:
                    # Handle various request exceptions (timeout, connection error, etc.)
                    blocked_websites.append(result['link'])
                    blocked_websites_count += 1
                
        except Exception as e:
            if 'youtube.com' in result.get('link', ''):
                failed_youtube.append(result['link'])
                failed_youtube_count += 1
            else:
                blocked_websites.append(result['link'])
                blocked_websites_count += 1
            continue
            
        # Check if we've reached our target
        total_successful = successful_website_count + successful_youtube_count
        if total_successful >= target_sources:
            break
            
    # If we didn't reach our target and there might be more results, consider making another search with a modified query
    if (successful_website_count + successful_youtube_count < target_sources) and (total_attempted < max_attempts):
        if progress_callback:
            progress_callback(f"Not enough sources found. Attempting to find more results...", 0.7)
        
        # You could implement pagination or modified queries here if needed
        # This is a simple example - you might want to add more sophisticated handling
        try:
            # Add a refinement to the query to get different results
            refined_query = f"{query} more information"
            payload = json.dumps({"q": refined_query})
            response = requests.post(url, headers=headers, data=payload)
            additional_results = response.json()
            
            if 'organic' in additional_results:
                # Process additional results (similar logic as above)
                for rank, result in enumerate(additional_results['organic'], 1):
                    # Skip if we've already processed this URL
                    if result['link'] in processed_urls:
                        continue
                        
                    processed_urls.add(result['link'])
                    total_attempted += 1
                    
                    # Safety check
                    if total_attempted >= max_attempts:
                        break
                        
                    # Calculate progress
                    progress = 0.2 + (0.6 * min(total_attempted, target_sources) / target_sources)
                    
                    # Same processing logic as above
                    try:
                        if 'youtube.com' in result['link'] and include_youtube:
                            if progress_callback:
                                progress_callback(f"Processing additional YouTube video {successful_youtube_count + 1} (attempt {total_attempted})", progress)
                            
                            video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', result['link'])
                            if video_id_match:
                                transcript = YouTubeTranscriptApi.get_transcript(video_id_match.group(1))
                                transcript_text = ' '.join([entry['text'] for entry in transcript])
                                youtube_contents.append({
                                    'title': result.get('title', 'YouTube Video'),
                                    'link': result['link'],
                                    'content': transcript_text
                                })
                                successful_youtube_count += 1
                            else:
                                failed_youtube.append(result['link'])
                                failed_youtube_count += 1
                        else:
                            if progress_callback:
                                progress_callback(f"Processing additional website {successful_website_count + 1} (attempt {total_attempted})", progress)
                            
                            try:
                                page_response = requests.get(result['link'], timeout=5)
                                page_response.raise_for_status()
                                
                                if page_response.status_code == 200:
                                    soup = BeautifulSoup(page_response.content, 'html.parser')
                                    content = clean_content(soup, result['link'])
                                    if content and content != "No main content found.":
                                        website_contents.append({
                                            'title': result.get('title', 'Website'),
                                            'link': result['link'],
                                            'content': content
                                        })
                                        successful_website_count += 1
                                    else:
                                        blocked_websites.append(result['link'])
                                        blocked_websites_count += 1
                                else:
                                    blocked_websites.append(result['link'])
                                    blocked_websites_count += 1
                            except requests.exceptions.RequestException as e:
                                blocked_websites.append(result['link'])
                                blocked_websites_count += 1
                            
                    except Exception as e:
                        if 'youtube.com' in result.get('link', ''):
                            failed_youtube.append(result['link'])
                            failed_youtube_count += 1
                        else:
                            blocked_websites.append(result['link'])
                            blocked_websites_count += 1
                        continue
                        
                    # Check if we've reached our target
                    total_successful = successful_website_count + successful_youtube_count
                    if total_successful >= target_sources:
                        break
        except Exception as e:
            # If the additional search fails, just continue with what we have
            logging.error(f"Additional search error: {str(e)}")

    if progress_callback:
        progress_callback("Generating summary...", 0.8)
        
    # Format the content for the prompt
    formatted_content = ""
    
    # Create a source mapping for easier reference
    source_mapping = {}
    
    # Add website content
    for i, web_content in enumerate(website_contents, 1):
        source_id = f"website_{i}"
        source_mapping[source_id] = web_content['link']
        formatted_content += f"<{source_id}>\nTitle: {web_content['title']}\nURL: {web_content['link']}\nSource ID: {source_id}\n\n{web_content['content']}\n</{source_id}>\n\n"
    
    # Add YouTube content
    for i, yt_content in enumerate(youtube_contents, 1):
        source_id = f"youtube_video_{i}"
        source_mapping[source_id] = yt_content['link']
        formatted_content += f"<{source_id}>\nTitle: {yt_content['title']}\nURL: {yt_content['link']}\nSource ID: {source_id}\n\n{yt_content['content']}\n</{source_id}>\n\n"
    
    # Create a combined content for display
    combined_content = []
    for i, web_content in enumerate(website_contents, 1):
        source_id = f"website_{i}"
        formatted_entry = f"[Website {i}] {web_content['title']}\nURL: {web_content['link']}\n\nSource ID: {source_id}\n\n{web_content['content']}"
        combined_content.append(formatted_entry)
    
    for i, yt_content in enumerate(youtube_contents, 1):
        source_id = f"youtube_video_{i}"
        formatted_entry = f"[YouTube {i}] {yt_content['title']}\nURL: {yt_content['link']}\n\nSource ID: {source_id}\n\n{yt_content['content']}"
        combined_content.append(formatted_entry)
    
    # Get today's date
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    # Create source reference section for clear citations
    source_references = "\n".join([f"{source_id}: {url}" for source_id, url in source_mapping.items()])
    
    # Create the prompt with the template
    system_prompt = """<role>You are a world class search engine. 
Based on the web content and youtube transcripts, create a world-class summary to answer the user query.</role>
"""

    prompt_template = f"""<instructions>The content is organized with tags to indicate different sources:
- <website_1>, <website_2>, etc.: Content from different websites
- <youtube_video_1>, <youtube_video_2>, etc.: Transcripts from different YouTube videos

Source Reference Map:
{source_references}

Today date: {today_date}

User Query: {query}

Carefully think based on the user query and content to generate a world-class answer.
Output format: First generate a short answer, then create a detail answer. With clear title for both.
Be concise but include all of the important details. 
Give examples if possible.  
Focus on high quality and accuracy: filter, compare, select from provided content to get the best answer! 
If you dont have enough info, state so and give users links to look by themself. Do not make up info!  

IMPORTANT - For citations: 
1. When citing a source, use proper markdown links: [text](URL)
2. Replace references like "website_1" or "youtube_video_1" with actual clickable links
3. Example: Instead of writing "According to [website_1]..." write "According to [this source](https://example.com)..."
4. Use the source URLs in the references section above for your links
5. NEVER use bracketed references like [website_1] in your final output

Output nicely in Markdown with clear titles and contents.

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
</instructions>

<content>
{formatted_content}
</content>
"""

    # Use the model to generate a summary
    try:
        summary = get_response(
            prompt_template,
            model_name=model_choice,
            system_prompt=system_prompt
        )
    except Exception as e:
        raise Exception(f"Summary generation error: {str(e)}")

    if progress_callback:
        progress_callback("Complete!", 1.0)

    # Create a structured result object
    access_stats = {
        "target_sources": target_sources,
        "total_attempted": total_attempted,
        "successful_websites": successful_website_count,
        "successful_youtube": successful_youtube_count,
        "blocked_websites": blocked_websites_count,
        "failed_youtube": failed_youtube_count,
        "blocked_urls": blocked_websites,
        "failed_youtube_urls": failed_youtube
    }

    # Include both original and any additional search results
    combined_search_results = search_results
    if 'additional_results' in locals():
        combined_search_results['additional_results'] = additional_results

    return successful_website_count, successful_youtube_count, blocked_websites_count, '\n\n'.join(combined_content), summary, len('\n\n'.join(combined_content).split()), combined_search_results, access_stats, source_mapping

def search_page():
    """
    Input: None
    Process: Handles the search page UI and functionality
    Output: Renders the search page interface
    """
    st.title("AI-Powered Search Assistant 🔍")
    st.write("Get comprehensive answers from multiple web sources and YouTube videos.")
    
    # Initialize include_youtube in session state if not present
    if 'include_youtube' not in st.session_state:
        st.session_state.include_youtube = True
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("🔍 Enter your search query:", 
                            placeholder="How's the weather in San Francisco?",
                            help="Be specific for better results")
    with col2:
        search_type = st.selectbox("Search Depth:", 
                                 ["Quick (5 sources)", "Deep (10 sources)"],
                                 help="Deep search takes longer but provides more comprehensive results")
    
    # Model selection and search button in same row
    col3, col4, col5 = st.columns([2, 2, 1])
    with col3:
        default_model_index = MODEL_CONFIG.index("google/gemini-2.0-flash-lite-001")
        model_choice = st.selectbox(
            "AI Model:", 
            MODEL_CONFIG,
            index=default_model_index,
            help="Different models may provide different perspectives"
        )
    with col4:
        include_youtube = st.checkbox("Include YouTube content", 
                                    value=st.session_state.include_youtube,
                                    help="Include transcripts from relevant YouTube videos",
                                    key="include_youtube_checkbox")
        # Update session state
        st.session_state.include_youtube = include_youtube
    with col5:
        search_button = st.button("🔎 Search", use_container_width=True)
    
    # Initialize session state for search history
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    if search_button and query:
        try:
            # Create placeholder containers for results
            results_container = st.container()
            
            # Show searching status in a temporary status indicator
            with st.status("🔍 Searching...") as status:
                # Search progress tracking
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                def update_progress(message, progress):
                    progress_text.text(message)
                    progress_bar.progress(progress)
                    status.update(label=message)
                
                # Perform search
                search_depth = "deep" if search_type == "Deep (10 sources)" else "fast"
                websites_used, youtube_videos_used, blocked_count, combined_content, response, word_count, serper_results, access_stats, source_mapping = search_and_summarize(
                    query, model_choice, search_depth, include_youtube, update_progress
                )
                
                # Add to search history
                st.session_state.search_history.append({
                    "query": query,
                    "response": response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "access_stats": access_stats,
                    "combined_content": combined_content,
                    "serper_results": serper_results,
                    "source_mapping": source_mapping
                })
            
            # Display results OUTSIDE the status block
            with results_container:
                total_successful = websites_used + youtube_videos_used
                success_percentage = round((total_successful / access_stats['target_sources']) * 100)
                
                if total_successful >= access_stats['target_sources']:
                    st.success(f"Success! Found all {access_stats['target_sources']} target sources ({websites_used} websites and {youtube_videos_used} YouTube videos).")
                else:
                    st.warning(f"Found {total_successful} of {access_stats['target_sources']} target sources ({success_percentage}%). {blocked_count} websites were blocked/inaccessible.")
                
                # Use tabs to separate results and sources
                tabs = st.tabs(["📝 Summary", "🔍 Sources", "🚫 Blocked Sites", "🌐 Serper API", "🔧 Debug"])
                
                # First tab: Summary
                with tabs[0]:
                    st.markdown(response)
                
                # Second tab: Source details
                with tabs[1]:
                    st.text_area("Raw Source Content", combined_content, height=400)
                
                # Third tab: Blocked sites
                with tabs[2]:
                    st.write(f"#### Access Statistics")
                    
                    # Calculate success rate
                    success_rate = 0
                    if access_stats['target_sources'] > 0:
                        success_rate = round((access_stats['successful_websites'] + access_stats['successful_youtube']) / access_stats['target_sources'] * 100)
                    
                    # Display target information
                    st.write(f"Target sources: {access_stats['target_sources']}")
                    st.write(f"Successfully accessed: {access_stats['successful_websites'] + access_stats['successful_youtube']} ({success_rate}%)")
                    
                    # Display detailed statistics
                    st.write(f"Total attempts: {access_stats['total_attempted']}")
                    st.write(f"Successful websites: {access_stats['successful_websites']}")
                    st.write(f"Successful YouTube videos: {access_stats['successful_youtube']}")
                    st.write(f"Blocked/inaccessible websites: {access_stats['blocked_websites']}")
                    st.write(f"Failed YouTube videos: {access_stats['failed_youtube']}")
                    
                    if access_stats['blocked_urls']:
                        st.write("#### Blocked Website URLs:")
                        for url in access_stats['blocked_urls']:
                            st.write(f"- {url}")
                    
                    if access_stats['failed_youtube_urls']:
                        st.write("#### Failed YouTube URLs:")
                        for url in access_stats['failed_youtube_urls']:
                            st.write(f"- {url}")
                
                # Fourth tab: Serper API results
                with tabs[3]:
                    st.json(serper_results)
                
                # Fifth tab: Debug information
                with tabs[4]:
                    st.write("#### Source Reference Map")
                    
                    # Use the source mapping directly
                    for source_id, url in source_mapping.items():
                        st.write(f"- **{source_id}**: [{url}]({url})")
                    
                    # Show example of correct citation format
                    st.write("#### Proper Citation Example")
                    if source_mapping:
                        sample_source_id = list(source_mapping.keys())[0]
                        sample_url = source_mapping[sample_source_id]
                        st.write(f"Instead of writing: According to [{sample_source_id}]...")
                        st.write(f"Write: According to [this source]({sample_url})...")
                    
                    # Display combined content as markdown
                    st.write("#### Combined Content as Markdown")
                    st.text_area("Raw Content for Debugging", combined_content, height=300)
                    
        except Exception as e:
            st.error(f"An error occurred during search: {str(e)}")
            logging.error(f"Search error: {str(e)}", exc_info=True)  # Log the full exception
            
    # Show search history
    if st.session_state.search_history:
        with st.expander("📚 Search History", expanded=False):
            # Create a column for search history
            history_col = st.container()
            
            # Create a column for displaying selected history item
            result_col = st.container()
            
            # Set up session state for selected history item if not present
            if 'selected_history_item' not in st.session_state:
                st.session_state.selected_history_item = None
            
            
            # Display search history in first column
            with history_col:
                for i, search in enumerate(reversed(st.session_state.search_history)):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{search['timestamp']}**: {search['query']}")
                    with col2:
                        if st.button(f"View #{i+1}", key=f"view_{i}"):
                            st.session_state.selected_history_item = i
                    st.divider()
            
            # Display selected history item in second column
            with result_col:
                if st.session_state.selected_history_item is not None:
                    i = st.session_state.selected_history_item
                    search = list(reversed(st.session_state.search_history))[i]
                    
                    # Display success message
                    st.success(f"Results for: {search['query']}")
                    
                    # Display the response directly without nested tabs or expanders
                    st.markdown(search['response'])
                    
                    # Add tabs for additional information
                    if 'combined_content' in search and 'serper_results' in search:
                        details_tabs = st.tabs(["🔍 Sources", "🌐 Serper API", "🔧 Debug"])
                        with details_tabs[0]:
                            st.text_area("Raw Source Content", search['combined_content'], height=400)
                        with details_tabs[1]:
                            st.json(search['serper_results'])
                        with details_tabs[2]:
                            st.write("#### Source Reference Map")
                            if 'source_mapping' in search:
                                for source_id, url in search['source_mapping'].items():
                                    st.write(f"- **{source_id}**: [{url}]({url})")
                                    
                                    # Show example of correct citation format
                                    st.write("#### Proper Citation Example")
                                    if search['source_mapping']:
                                        sample_source_id = list(search['source_mapping'].keys())[0]
                                        sample_url = search['source_mapping'][sample_source_id]
                                        st.write(f"Instead of writing: According to [{sample_source_id}]...")
                                        st.write(f"Write: According to [this source]({sample_url})...")
                            else:
                                st.warning("Source mapping not available for this search (likely an older search).")
                        
                        if st.button("Clear Results", key="clear_results"):
                            st.session_state.selected_history_item = None

#===============================================================



#CHAT PAGE:
def get_current_time():
    """
    Input: None
    Process: Gets current time using datetime
    Output: Returns formatted current time string
    """
    current_time = datetime.now()
    return current_time.strftime("%I:%M %p, %B %d, %Y")

def get_streaming_response(prompt: str, model_name: str = "openai/gpt-4o", system_prompt: str = None):
    """
    Input: prompt, optional model name, and optional system prompt
    Process: Generates streaming response using OpenRouter API with function calling support
    Output: Yields response chunks and returns complete response
    """
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=st.secrets["OPENROUTER_API_KEY"],
        )

        # Define tools in the simpler format
        tools = [{
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            }
        }]

        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Check if it's an Anthropic model
        is_anthropic = "anthropic" in model_name.lower()
        
        # For Anthropic models, don't include tools
        api_params = {
            "model": model_name,
            "messages": messages,
            "stream": False
        }
        
        if not is_anthropic:
            api_params["tools"] = tools
        
        # First call to check for function calling
        response = client.chat.completions.create(**api_params)

        # Handle function calling if present
        if not is_anthropic and hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
            # Add the assistant's response to messages
            messages.append(response.choices[0].message.dict())
            
            # Process each tool call
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                
                # Yield information about which tool is being used
                yield f"\n🔧 Using tool: {function_name}\n", function_name
                
                # Execute the tool
                if function_name == "get_current_time":
                    tool_result = datetime.now().strftime("%I:%M %p, %B %d, %Y")
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(tool_result)
                    })

                    # Yield the tool result for streaming display
                    yield f"📊 Tool Result: {tool_result}\n", tool_result

            # Get final response with tool result
            stream_params = {
                "model": model_name,
                "messages": messages,
                "stream": True
            }
            if not is_anthropic:
                stream_params["tools"] = tools
                
            stream = client.chat.completions.create(**stream_params)

            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content, full_response

        else:
            # Regular streaming response if no function call
            stream_params = {
                "model": model_name,
                "messages": messages,
                "stream": True
            }
            if not is_anthropic:
                stream_params["tools"] = tools
                
            stream = client.chat.completions.create(**stream_params)

            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content, full_response
                    
        return full_response
            
    except Exception as e:
        error_msg = f"Error generating response with {model_name}: {str(e)}"
        yield error_msg, error_msg
        return error_msg

def chat_page():
    """
    Input: None
    Process: Creates a basic chat interface with model selection and streaming responses
    Output: Displays chat interface and handles message exchange
    """
    st.title("AI Chat 💬")
    
    # Initialize chat history in session state if not present
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": "👋 Hi! I'm your AI assistant. How can I help you today?"
        }]
    
    # Model selection - use raw model names directly
    default_model_index = MODEL_CONFIG.index("google/gemini-2.5-flash-preview")
    
    selected_model = st.selectbox(
        "Select Model:",
        MODEL_CONFIG,
        index=default_model_index
    )
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Add reset button
    if st.button("Reset Chat"):
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": "👋 Hi! I'm your AI assistant. How can I help you today?"
        }]
        st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Format the entire conversation for the API call
        message_history = []
        for msg in st.session_state.chat_history:
            # Skip the first assistant greeting when sending to API
            if msg == st.session_state.chat_history[0] and msg["role"] == "assistant":
                continue
            message_history.append({"role": msg["role"], "content": msg["content"]})
        
        # Create the full conversation context
        full_prompt = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in message_history])
        
        # Display assistant response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Stream the response
                for chunk, current_response in get_streaming_response(
                    prompt=full_prompt,
                    model_name=selected_model,
                    system_prompt="You are a helpful, friendly assistant. Provide concise and accurate responses to the latest user message in the conversation."
                ):
                    full_response = current_response
                    # Display response with a blinking cursor
                    message_placeholder.markdown(full_response + "▌")
                
                # After streaming completes, display the final message
                message_placeholder.markdown(full_response)
                
                # Add assistant response to history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})



#===============================================================


# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

def home_page():
    """
    Input: None
    Process: Handles the home page UI and functionality
    Output: Renders the home page interface
    """
    st.write("## Welcome to My Portfolio!")
    st.write("I am a passionate developer with expertise in...")

def main():
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        selected_page = st.radio(
            "Go to",
            ["Home", "Search", "Chat"],
            key="navigation"
        )
        st.session_state.current_page = selected_page

    # Main content area
    st.title(f"{st.session_state.current_page}")
    
    # Page content based on selection
    if st.session_state.current_page == "Home":
        home_page()
    elif st.session_state.current_page == "Search":
        search_page()
    elif st.session_state.current_page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()
