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

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

st.set_page_config(
    page_title="Huy Nguyen Portfolio",
    page_icon="👨‍💻",
    layout="wide"
)

# Model configuration
MODEL_CONFIG = {
    "openai/gpt-4o": {
        "display_name": "OpenAI GPT-4o"
    },
    "anthropic/claude-3.7-sonnet": {
        "display_name": "Anthropic Claude 3.7 Sonnet"
    },
    "google/gemini-2.0-flash-001": {
        "display_name": "Google Gemini 2.0 Flash"
    },
    "google/gemini-2.5-pro-preview-03-25": {
        "display_name": "Google Gemini 2.5 Pro"
    },
    "google/gemini-2.0-flash-lite-001": {
        "display_name": "Google Gemini 2.0 Flash Lite"
    }
}

# Page configuration


# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

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

def clean_content(soup):
    """
    Input: BeautifulSoup object
    Process: Cleans HTML content by removing unnecessary elements and extracting relevant text
    Output: Cleaned text content
    """
    # Remove unnecessary elements
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
        # Extract text from relevant elements
        content = []
        for elem in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'blockquote']):
            text = elem.get_text(strip=True)
            if text:
                content.append(text)
        return '\n\n'.join(content)
    return "No main content found."

def search_and_summarize(query, model_choice, search_type, progress_callback=None):
    """
    Input: Search query, model choice, search type, and optional progress callback
    Process: Searches web and YouTube for content and summarizes results
    Output: Number of sources used, combined content, and AI summary
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

    combined_content = []
    successful_website_count = 0
    successful_youtube_count = 0
    total_links = 5 if search_type == "fast" else 10

    for rank, result in enumerate(search_results['organic'][:total_links], 1):
        progress = 0.2 + (0.5 * rank / total_links)
        
        try:
            if 'youtube.com' in result['link']:
                if progress_callback:
                    progress_callback(f"Processing YouTube video {successful_youtube_count + 1}", progress)
                
                video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', result['link'])
                if video_id_match:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id_match.group(1))
                    transcript_text = ' '.join([entry['text'] for entry in transcript])
                    combined_content.append(f"[YouTube] {result['title']}\n{transcript_text}")
                    successful_youtube_count += 1
            else:
                if progress_callback:
                    progress_callback(f"Processing website {successful_website_count + 1}", progress)
                
                page_response = requests.get(result['link'], timeout=5)
                soup = BeautifulSoup(page_response.content, 'html.parser')
                content = clean_content(soup)
                combined_content.append(f"[Website] {result['link']}\n{content}")
                successful_website_count += 1
                
        except Exception as e:
            continue

    if progress_callback:
        progress_callback("Generating summary...", 0.8)

    # Use the model to generate a summary
    try:
        summary = get_response(
            f"Summarize this information:\n\n{'\n\n'.join(combined_content)}",
            model_name=model_choice,
            system_prompt="You are a helpful AI assistant that provides clear, accurate summaries of information."
        )
    except Exception as e:
        raise Exception(f"Summary generation error: {str(e)}")

    if progress_callback:
        progress_callback("Complete!", 1.0)

    return successful_website_count, successful_youtube_count, '\n\n'.join(combined_content), summary, len('\n\n'.join(combined_content).split())





def main():
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        selected_page = st.radio(
            "Go to",
            ["Home", "Search"],
            key="navigation"
        )
        st.session_state.current_page = selected_page

    # Main content area
    st.title(f"{st.session_state.current_page}")
    
    # Page content based on selection
    if st.session_state.current_page == "Home":
        st.write("## Welcome to My Portfolio!")
        st.write("I am a passionate developer with expertise in...")


    elif st.session_state.current_page == "Search":
        st.title("AI-Powered Search Assistant 🔍")
        st.write("Get comprehensive answers from multiple web sources and YouTube videos.")
        
        # Example queries section
        with st.expander("📝 Example Queries", expanded=False):
            st.markdown("""
            Try these example queries:
            - "Latest developments in quantum computing 2024"
            - "Best practices for cybersecurity in cloud computing"
            - "How to implement zero trust architecture"
            - "Recent advancements in AI for healthcare"
            """)
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("🔍 Enter your search query:", 
                                placeholder="What would you like to learn about?",
                                help="Be specific for better results")
        with col2:
            search_type = st.selectbox("Search Depth:", 
                                     ["Quick (5 sources)", "Deep (10 sources)"],
                                     help="Deep search takes longer but provides more comprehensive results")
        
        # Model selection and search button in same row
        col3, col4, col5 = st.columns([2, 2, 1])
        with col3:
            model_options = list(MODEL_CONFIG.keys())
            model_labels = [MODEL_CONFIG[model]["display_name"] for model in model_options]
            model_index = st.selectbox(
                "AI Model:", 
                range(len(model_options)),
                format_func=lambda i: model_labels[i],
                help="Different models may provide different perspectives"
            )
            model_choice = model_options[model_index]
        with col4:
            include_youtube = st.checkbox("Include YouTube content", 
                                        value=True,
                                        help="Include transcripts from relevant YouTube videos")
        with col5:
            search_button = st.button("🔎 Search", use_container_width=True)
        
        # Initialize session state for search history
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        if search_button and query:
            try:
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
                    websites_used, youtube_videos_used, combined_content, response, word_count = search_and_summarize(
                        query, model_choice, search_depth, update_progress
                    )
                    
                    # Add to search history
                    st.session_state.search_history.append({
                        "query": query,
                        "response": response,
                        "response_escaped": response.replace('#', '\\#').replace('>', '\\>'),  # Escape markdown characters
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Display results
                    st.success(f"Found information from {websites_used} websites and {youtube_videos_used} YouTube videos")
                    
                    # Use tabs to separate results and sources
                    tabs = st.tabs(["📝 Summary", "🔍 Sources"])
                    
                    # First tab: Summary
                    with tabs[0]:
                        st.markdown(response)
                    
                    # Second tab: Source details
                    with tabs[1]:
                        st.text_area("Raw Source Content", combined_content, height=400)
                        
            except Exception as e:
                st.error(f"An error occurred during search: {str(e)}")
                logging.error(f"Search error: {str(e)}", exc_info=True)  # Log the full exception
                
        # Show search history
        if st.session_state.search_history:
            # Create a column for search history
            history_col = st.container()
            
            # Create a column for displaying selected history item
            result_col = st.container()
            
            # Set up session state for selected history item if not present
            if 'selected_history_item' not in st.session_state:
                st.session_state.selected_history_item = None
            
            # Display search history in first column
            with history_col:
                with st.expander("📚 Search History", expanded=False):
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
                    st.subheader(f"Results for: {search['query']}")
                    
                    # Handle older search history items that don't have response_escaped
                    response_text = search['response']
                    
                    # Use text area instead of markdown to prevent rendering nested UI elements
                    st.markdown(response_text)
                 
                    
                    if st.button("Clear Results", key="clear_results"):
                        st.session_state.selected_history_item = None

if __name__ == "__main__":
    main()
