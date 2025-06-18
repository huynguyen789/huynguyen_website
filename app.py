# -*- coding: utf-8 -*-
"""
Main application file for the portfolio website.
Handles navigation and main layout structure.
Refactored into modular functions.
Integrates standalone ai_search module.
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
import os # Needed for environment variables check in tool (optional)

# --- Import the standalone search module ---
import tools.ai_search as ai_search
# --- Import the voice chat module ---
# from voice_chat import voice_chat_page # Removed
# --- Import Tool Functions ---
from tools.tools import get_current_time, perform_web_search # Updated import
from tools.get_youtube_transcript import get_youtube_transcript_with_fallback # Updated import

# --- Configuration ---

# Configure logging
logging.basicConfig(
    level=logging.INFO, # Set to INFO to see logs from ai_search
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__) # Logger for app.py

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
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash-lite-preview-06-17",
    "google/gemini-2.5-flash",
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
    # Check if the key exists in secrets
    if "OPENROUTER_API_KEY" not in st.secrets:
        st.error("🚨 OPENROUTER_API_KEY not found in Streamlit Secrets!")
        st.stop()
        return None # Should not be reached due to st.stop()
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENROUTER_API_KEY"],
    )



# --- Tool Definitions and Mapping ---
CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform_web_search",
            "description": "Use this tool to find current information, answer questions about recent events, or look up topics not covered in your training data. Performs a web search using multiple sources (websites, YouTube) and returns a concise summary of the findings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The specific search query to use. Should be targeted and descriptive.",
                    }
                },
                "required": ["query"],
            },
        }
    }
]

# Map tool names to functions - THIS MUST BE DEFINED BEFORE get_streaming_llm_response USES IT
# Make it global or ensure definition order. Global is simpler here.
AVAILABLE_FUNCTIONS = {
    "get_current_time": get_current_time, # Uses function from tools.tools
    "perform_web_search": perform_web_search, # Uses function from tools.tools
}

# --- LLM Response Generation (Modified get_streaming_llm_response) ---

def get_streaming_llm_response(
    messages: List[Dict[str, str]],
    model_name: str = "openai/gpt-4o",
    tools: Optional[List[Dict]] = None,
    client: Optional[OpenAI] = None
) -> Generator[Tuple[str, str], None, None]:
    """
    Input: List of messages, optional model name, optional tools list, optional OpenAI client.
    Process: Generates a streaming response using OpenRouter API, handling potential function calls,
             including the web search tool.
    Output: Yields tuples of (chunk, full_response_so_far_or_signal).
            Signals can be 'tool_call:...' or 'tool_feedback:...'
    """
    if client is None:
        client = get_openai_client()
        if client is None: # Handle case where client creation failed
             yield "Error: Could not initialize AI client.", "error:client_init_failed"
             return

    try:
        is_anthropic = "anthropic" in model_name.lower() # Anthropic might handle tools differently

        # --- Initial Call (Check for Tool Use) ---
        api_params = {
            "model": model_name,
            "messages": messages,
            "stream": False # First call is non-streaming to check for tool_calls
        }
        # Only send tools if the model likely supports them and tools are provided
        if not is_anthropic and tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"

        logger.debug(f"Making initial non-streaming call to {model_name} with tools: {bool(tools)}")
        response = client.chat.completions.create(**api_params)
        response_message = response.choices[0].message

        tool_calls = getattr(response_message, 'tool_calls', None)

        # --- Handle Tool Calls ---
        if tool_calls:
            logger.info(f"Tool calls requested by {model_name}: {[tc.function.name for tc in tool_calls]}")
            # Append the assistant's response (requesting tool use) to messages
            # Use .model_dump() for newer Pydantic versions with OpenAI client
            messages.append(response_message.model_dump(exclude_unset=True))

 
            tool_results_for_api = [] # Collect results before the next API call

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = AVAILABLE_FUNCTIONS.get(function_name)
                tool_call_id = tool_call.id

                yield f"🔧 Requesting Tool: `{function_name}`\n", f"tool_call:{function_name}" # Signal tool use

                if function_to_call:
                    try:
                        # Parse arguments
                        function_args = json.loads(tool_call.function.arguments)
                        logger.info(f"Calling tool '{function_name}' with args: {function_args}")

                        # --- Execute the tool ---
                        # Use a generic feedback mechanism before/after
                        yield f"⏳ Executing tool: '{function_name}'...", f"tool_feedback:Executing {function_name}"

                        function_response = function_to_call(**function_args) # Pass parsed args

                        yield f"✅ Tool '{function_name}' finished.", f"tool_feedback:{function_name} Complete"

                        # Ensure result is JSON string for the API
                        tool_result_str = json.dumps(function_response)
                        logger.info(f"Tool '{function_name}' result: {tool_result_str}")

                        # Prepare result message for the next API call
                        tool_results_for_api.append({
                            "tool_call_id": tool_call_id,
                            "role": "tool",
                            "name": function_name,
                            "content": tool_result_str,
                        })

                    except Exception as e:
                        logger.error(f"Error executing tool {function_name}: {e}", exc_info=True)
                        yield f"⚠️ Error executing tool '{function_name}': {e}\n", f"tool_error:{str(e)}"
                        # Provide an error response for the tool call
                        tool_results_for_api.append({
                            "tool_call_id": tool_call_id,
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps({"error": str(e)}),
                        })
                else:
                     # Handle case where function is not found
                     logger.warning(f"Tool '{function_name}' requested by model but not found in AVAILABLE_FUNCTIONS.")
                     yield f"⚠️ Tool '{function_name}' not found.\n", f"tool_error:not_found"
                     tool_results_for_api.append({
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps({"error": f"Function {function_name} not found"}),
                     })

            # Append all tool results to messages
            messages.extend(tool_results_for_api)

            # --- Second Call (With Tool Results) ---
            yield "🧠 Processing tool results and generating final answer...", "tool_feedback:Generating final answer..."
            logger.debug(f"Making second streaming call to {model_name} with tool results.")
            stream_params = {
                "model": model_name,
                "messages": messages,
                "stream": True
            }
            # Don't send tools definition again in the second call
            stream = client.chat.completions.create(**stream_params)
            full_response = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    yield content, full_response
            logger.info("Streaming response with tool results finished.")
            # Generator implicitly returns None

        # --- No Tool Calls ---
        else:
            logger.debug(f"No tool calls requested by {model_name}. Proceeding with direct streaming.")
            # Regular streaming response if no tools were called
            stream_params = {
                "model": model_name,
                "messages": messages, # Use original messages
                "stream": True
            }
            # Still offer tools if none were chosen initially, if applicable
            if not is_anthropic and tools:
                stream_params["tools"] = tools
                # No tool_choice needed if we just want to stream the direct answer

            stream = client.chat.completions.create(**stream_params)
            full_response = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    yield content, full_response
            logger.info("Direct streaming response finished.")
            # Generator implicitly returns None

    except Exception as e:
        error_msg = f"Error generating streaming response with {model_name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        yield error_msg, error_msg # Yield error message as content and signal
        # Generator implicitly returns None


# --- UI Page Modules ---

def home_page():
    """ Renders the Home page content. """
    st.markdown("## Welcome to My Portfolio!")
    st.markdown("""
    I am a passionate developer exploring the intersection of AI, web technologies, and data.
    This site showcases some experimental tools built with Streamlit and various APIs.

    Navigate using the sidebar to explore:
    - **Search:** An AI-powered search assistant that synthesizes information from multiple web and YouTube sources.
    - **Chat:** A conversational AI interface using different language models, capable of using tools like web search.
    """)

def search_page():
    """ Renders the Search page UI and handles search logic using ai_search module. """
    st.title("AI-Powered Search Assistant 🔍")
    st.write("Get comprehensive answers synthesized from multiple web sources and YouTube videos.")

    # Check for required API keys (as environment variables for ai_search.py)
    # Provide a clear message if missing, as the search won't work.
    if not os.environ.get("SERPER_API_KEY") or not os.environ.get("OPENROUTER_API_KEY"):
        st.error("🚨 Configuration Error: `SERPER_API_KEY` or `OPENROUTER_API_KEY` environment variable is not set. The search function requires these to operate.")
        st.warning("Please ensure these are set in your environment or Streamlit secrets (and mapped to environment variables if deployed).")
        st.stop() # Stop execution of this page if keys are missing

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
            default_model_index = MODEL_CONFIG.index("google/gemini-2.0-flash-001")
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
            # Use spinner for simpler progress indication
            with st.spinner(f"Performing '{search_depth_val}' search and generating summary with {model_choice}..."):
                # Call the imported search function
                search_results_data = ai_search.search(
                    query=query,
                    model=model_choice,
                    search_depth=search_depth_val,
                    include_youtube=st.session_state.include_youtube # Use current state
                )

            # Store results in history before displaying
            history_entry = {
                "query": query,
                "model_used": model_choice,
                "search_depth": search_depth_val,
                "included_youtube": st.session_state.include_youtube,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **search_results_data # Unpack the results dict from ai_search.search
            }
            st.session_state.search_history.append(history_entry)
            # Set the latest search as the selected one to display immediately
            st.session_state.selected_history_item_search = len(st.session_state.search_history) - 1
            st.rerun() # Rerun to display the new result from history

        except Exception as e:
            st.error(f"An error occurred during the search process: {str(e)}")
            logger.error(f"Search page error for query '{query}': {str(e)}", exc_info=True)

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

                access_stats = search_data.get('access_stats') # Use .get for safety
                serper_results = search_data.get('serper_results') # Use .get for safety
                combined_raw_text = search_data.get('combined_raw_text', '') # Use .get for safety

                # Display status message based on access_stats if available
                if access_stats:
                    total_successful = access_stats.get('successful_websites', 0) + access_stats.get('successful_youtube', 0)
                    target = access_stats.get('target_sources', 0)
                    success_percentage = round((total_successful / target) * 100) if target > 0 else 0
                    blocked_websites = access_stats.get('blocked_websites', 0)
                    failed_youtube = access_stats.get('failed_youtube', 0)

                    if search_data['status'] == "success":
                        if total_successful >= target:
                            st.success(f"Successfully gathered content from {total_successful} sources ({access_stats.get('successful_websites', 0)} websites, {access_stats.get('successful_youtube', 0)} YouTube).")
                        else:
                            st.warning(f"Found {total_successful} of {target} target sources ({success_percentage}%). {blocked_websites} websites blocked/inaccessible, {failed_youtube} YouTube videos failed.")
                    elif search_data['status'] == "no_content":
                         st.error(f"Could not retrieve content from any sources ({blocked_websites} websites blocked/inaccessible, {failed_youtube} YouTube videos failed).")
                    elif search_data['status'] == "no_results":
                         st.error("No search results found for this query.")
                    elif search_data['status'] == "error":
                         st.error(f"An error occurred: {search_data.get('error_message', 'Unknown error')}")
                else:
                    # Fallback status display if access_stats are missing
                    st.info(f"Search status: {search_data['status']}")
                    if search_data['status'] == "error":
                        st.error(f"Error: {search_data.get('error_message', 'Unknown error')}")


                # Display content tabs only if the search didn't fail catastrophically before content gathering
                if search_data['status'] in ["success", "no_content"]:
                    # Define tabs - remove Debug Info for now as source_mapping isn't returned
                    tab_list = ["📝 Summary", "🔍 Sources", "📊 Access Stats"]
                    if serper_results: # Only show Raw Search if results exist
                        tab_list.append("🌐 Raw Search API")

                    tabs = st.tabs(tab_list)
                    tab_index = 0

                    with tabs[tab_index]: # Summary
                        tab_index += 1
                        if search_data['summary']:
                            st.markdown(search_data['summary'])
                        elif search_data['status'] == "no_content":
                            st.info("Summary could not be generated as no content was retrieved.")
                        elif search_data['status'] == "error" and search_data.get('error_message', '').startswith("Error: Failed to generate summary"):
                             st.warning(f"Summary generation failed: {search_data['error_message']}")
                        else:
                            st.info("Summary not available.")

                    with tabs[tab_index]: # Sources (Combined Raw Text)
                        tab_index += 1
                        st.text_area("Combined Source Content", combined_raw_text, height=400, key=f"sources_{selected_index}")

                    with tabs[tab_index]: # Access Stats
                        tab_index += 1
                        if access_stats:
                            st.write(f"**Target Sources:** {access_stats.get('target_sources', 'N/A')}")
                            st.write(f"**Total Attempts:** {access_stats.get('total_attempted', 'N/A')}")
                            st.write(f"**Successfully Accessed:** {total_successful} ({success_percentage}%)")
                            st.write(f"- Websites: {access_stats.get('successful_websites', 0)}")
                            st.write(f"- YouTube Videos: {access_stats.get('successful_youtube', 0)}")
                            st.write(f"**Failed/Blocked:**")
                            st.write(f"- Websites: {access_stats.get('blocked_websites', 0)}")
                            st.write(f"- YouTube Videos: {access_stats.get('failed_youtube', 0)}")

                            blocked_urls = access_stats.get('blocked_urls', [])
                            failed_youtube_urls = access_stats.get('failed_youtube_urls', [])

                            if blocked_urls:
                                with st.expander("Blocked/Inaccessible Website URLs"):
                                    st.json(blocked_urls)
                            if failed_youtube_urls:
                                with st.expander("Failed YouTube URLs"):
                                    st.json(failed_youtube_urls)
                        else:
                            st.info("Access statistics not available for this search.")

                    if serper_results: # Raw Search API Tab
                         with tabs[tab_index]:
                             tab_index += 1
                             st.json(serper_results)

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
                     st.caption(f"Model: {search['model_used']}, Depth: {search['search_depth']}, YouTube: {'Yes' if search['included_youtube'] else 'No'}, Status: {search['status']}")
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
        # Find a default model index safely
        try:
            default_model_index = MODEL_CONFIG.index("google/gemini-2.5-flash-preview")
        except ValueError:
            default_model_index = 0 # Fallback to the first model
        st.session_state.chat_sessions[new_session_id] = {
            "history": [{"role": "assistant", "content": "👋 Hi! How can I help you today? I can also perform web searches if needed."}],
            "model": MODEL_CONFIG[default_model_index]
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

    # Select current session - handle case where no sessions exist
    if session_ids:
        try:
            current_session_index = session_ids.index(st.session_state.current_chat_session_id) if st.session_state.current_chat_session_id in session_ids else 0
        except ValueError:
             current_session_index = 0 # Should not happen if session_ids is not empty

        selected_session_id = st.sidebar.radio(
            "Select Chat:",
            options=session_ids,
            format_func=lambda sid: session_display_names.get(sid, sid),
            key="chat_session_selector",
            index=current_session_index
        )
        # Update current session ID if selection changed
        if selected_session_id != st.session_state.current_chat_session_id:
            st.session_state.current_chat_session_id = selected_session_id
            st.rerun() # Rerun to load the selected chat
    else:
        st.session_state.current_chat_session_id = None # Ensure it's None if no sessions


    # --- Chat Interface for the selected session ---
    if st.session_state.current_chat_session_id and st.session_state.current_chat_session_id in st.session_state.chat_sessions:
        current_session = st.session_state.chat_sessions[st.session_state.current_chat_session_id]
        chat_history = current_session["history"]

        # Model selection for the current chat
        try:
            current_model_index = MODEL_CONFIG.index(current_session["model"])
        except ValueError:
            # Fallback if model name is somehow invalid
            try:
                current_model_index = MODEL_CONFIG.index("google/gemini-2.5-flash-preview")
            except ValueError:
                current_model_index = 0 # Absolute fallback
            current_session["model"] = MODEL_CONFIG[current_model_index] # Correct the stored model name


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
        chat_container = st.container( border=False) # Set height for scrollability
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

            # Prepare message history for API (excluding initial greeting if desired, but often good to keep)
            # Let's keep the initial greeting for context unless it causes issues.
            api_messages = chat_history # Send the whole history

            # Display assistant response with streaming
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response_content = ""
                tool_activity_log = [] # To store tool messages temporarily

                try:
                    # Use the streaming function
                    response_generator = get_streaming_llm_response(
                        messages=api_messages,
                        model_name=current_session["model"], # Use session's model
                        tools=CHAT_TOOLS # Pass defined tools
                    )

                    for chunk, current_response_or_signal in response_generator:
                        # Check if it's a signal or actual content
                        if isinstance(current_response_or_signal, str):
                            if current_response_or_signal.startswith("tool_call:"):
                                # Keep this - shows the tool is being called
                                tool_activity_log.append(f"```\n🔧 Calling Tool: {current_response_or_signal.split(':', 1)[1]}\n```")
                                # Update placeholder immediately to show the call
                                message_placeholder.markdown(full_response_content + "\n" + "\n".join(tool_activity_log) + "▌")
                            elif current_response_or_signal.startswith("tool_feedback:"):
                                # --- CHANGE HERE ---
                                # Show feedback as a toast, don't add to permanent log
                                feedback_message = current_response_or_signal.split(':', 1)[1]
                                st.toast(f"⏳ {feedback_message}")
                                # --- END CHANGE ---
                            elif current_response_or_signal.startswith("tool_error:"):
                                # Keep this - show errors
                                tool_activity_log.append(f"```\n⚠️ Tool Error: {current_response_or_signal.split(':', 1)[1]}\n```")
                                message_placeholder.markdown(full_response_content + "\n" + "\n".join(tool_activity_log) + "▌")
                            elif current_response_or_signal.startswith("error:"): # General errors
                                 tool_activity_log.append(f"```\n🚨 Error: {current_response_or_signal.split(':', 1)[1]}\n```")
                                 message_placeholder.markdown(full_response_content + "\n" + "\n".join(tool_activity_log) + "▌")
                            else:
                                # It's content, update the placeholder
                                full_response_content = current_response_or_signal
                                # Display content and only the relevant tool logs (calls/errors)
                                message_placeholder.markdown(full_response_content + "\n" + "\n".join(tool_activity_log) + "▌")
                        else:
                             # Should not happen based on generator output type hint, but handle defensively
                             logger.warning(f"Unexpected item type from generator: {type(current_response_or_signal)}")


                    # Final display without cursor
                    final_display_text = full_response_content + "\n" + "\n".join(tool_activity_log)
                    message_placeholder.markdown(final_display_text.strip())

                    # Add final assistant response to history (including tool activity for record)
                    chat_history.append({"role": "assistant", "content": final_display_text.strip()})
                    # No rerun needed here, message is already displayed

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    message_placeholder.error(error_msg)
                    chat_history.append({"role": "assistant", "content": f"Error: {error_msg}"}) # Log error in history
                    logger.error(f"Chat page streaming error: {e}", exc_info=True)

        # Option to clear the current chat session
        if st.button("Clear Current Chat", key=f"clear_chat_{st.session_state.current_chat_session_id}"):
            current_session["history"] = [{"role": "assistant", "content": "Chat cleared. How can I help?"}]
            st.rerun()

        # Option to delete the current chat session (in sidebar)
        if st.sidebar.button(f"🗑️ Delete '{session_display_names[st.session_state.current_chat_session_id]}'", key=f"delete_chat_{st.session_state.current_chat_session_id}"):
             del st.session_state.chat_sessions[st.session_state.current_chat_session_id]
             # Select the first available session or set to None if empty
             available_sessions = list(st.session_state.chat_sessions.keys())
             st.session_state.current_chat_session_id = available_sessions[0] if available_sessions else None
             st.rerun()

    elif not session_ids:
         st.info("Click '➕ New Chat' in the sidebar to start a conversation.")
    else:
         # This case might occur if the selected session ID becomes invalid somehow
         st.warning("Please select a chat session from the sidebar.")

def youtube_summary_page():
    """ Renders the YouTube Summary page UI and handles video summarization. """
    st.title("YouTube Video Summarizer 📺")
    st.write("Get AI-generated summaries of YouTube videos using their transcripts.")

    # Check for required API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        st.error("🚨 Configuration Error: `OPENROUTER_API_KEY` environment variable is not set.")
        st.warning("Please ensure this is set in your environment or Streamlit secrets.")
        st.stop()

    # Initialize session state for YouTube summary page
    if 'youtube_history' not in st.session_state:
        st.session_state.youtube_history = []

    # Default system prompt
    default_system_prompt = """You are an expert video content summarizer. Create a comprehensive yet concise summary of the YouTube video transcript provided. 

Format your response with:
- A brief overview (2-3 sentences)
- Key points covered (bullet points)
- Important details or insights
- Main takeaways

Keep the summary informative but readable."""

    # --- Input Section ---
    st.subheader("📝 Video Summary Settings")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        youtube_url = st.text_input(
            "🔗 YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            key="youtube_url_input"
        )
    
    with col2:
        summarize_button = st.button("🔄 Summarize Video", use_container_width=True, key="summarize_button")

    # Custom system prompt
    custom_system_prompt = st.text_area(
        "🎯 Custom Summary Instructions (optional):",
        value=default_system_prompt,
        height=120,
        help="Customize how you want the video to be summarized. Leave as default or modify as needed.",
        key="custom_system_prompt"
    )

    # --- Processing and Results ---
    if summarize_button and youtube_url:
        try:
            with st.spinner("Fetching video transcript..."):
                # Import the fallback function
                from tools.get_youtube_transcript import get_youtube_transcript_with_fallback
                
                # Use the fallback function with debugging enabled
                transcript, status = get_youtube_transcript_with_fallback(youtube_url, debug=True)
                
                if status != "success":
                    st.error(f"❌ {status}")
                    
                    # Add debugging info
                    with st.expander("🔍 Debug Information", expanded=False):
                        st.write("If this video works in the standalone script but not here, try:")
                        st.write("1. Check if the video is the same one you tested")
                        st.write("2. Try a different video (like a popular TED talk)")
                        st.write("3. The issue might be YouTube treating web apps differently")
                        st.code(f"Video URL: {youtube_url}")
                        st.code(f"Error: {status}")
                    
                    return

            # Trim transcript to context limit (120k characters)
            max_context_length = 120000
            if len(transcript) > max_context_length:
                transcript = transcript[:max_context_length]
                st.warning(f"⚠️ Transcript was truncated to {max_context_length:,} characters to fit context limits.")

            # Prepare messages for LLM
            system_prompt = custom_system_prompt.strip() if custom_system_prompt.strip() else default_system_prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the YouTube video transcript:\n\n{transcript}\n\nPlease provide a comprehensive summary following the instructions given."}
            ]

            # Generate streaming summary
            st.subheader("📊 Video Summary")
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_summary = ""

                try:
                    # Use streaming response
                    response_generator = get_streaming_llm_response(
                        messages=messages,
                        model_name="google/gemini-2.5-flash",
                        tools=None  # No tools needed for summarization
                    )

                    for chunk, current_response_or_signal in response_generator:
                        # Handle signals (though we don't expect tool calls)
                        if isinstance(current_response_or_signal, str) and current_response_or_signal.startswith(("tool_", "error:")):
                            continue  # Skip tool signals for this use case
                        else:
                            # It's content, update the placeholder
                            full_summary = current_response_or_signal
                            message_placeholder.markdown(full_summary + "▌")

                    # Final display without cursor
                    message_placeholder.markdown(full_summary)

                    # Store in history
                    history_entry = {
                        "url": youtube_url,
                        "custom_prompt": system_prompt,
                        "summary": full_summary,
                        "transcript": transcript,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.youtube_history.append(history_entry)

                    # Show full transcript in expandable section
                    with st.expander("📜 Full Transcript (click to show)", expanded=False):
                        st.text_area(
                            "Video Transcript:",
                            value=transcript,
                            height=400,
                            key=f"transcript_display_{len(st.session_state.youtube_history)}"
                        )

                except Exception as e:
                    error_msg = f"Error generating summary: {str(e)}"
                    message_placeholder.error(error_msg)
                    logger.error(f"YouTube summary error: {e}", exc_info=True)

        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
            logger.error(f"YouTube processing error: {e}", exc_info=True)

    # --- History Section ---
    if st.session_state.youtube_history:
        st.markdown("---")
        st.subheader("📚 Summary History")
        st.write(f"**{len(st.session_state.youtube_history)} video(s) summarized**")
        
        # Display history items (newest first)
        for i, entry in enumerate(reversed(st.session_state.youtube_history)):
            history_index = len(st.session_state.youtube_history) - 1 - i
            
            # Use a container instead of nested expanders
            with st.container(border=True):
                st.markdown(f"**Summary #{i+1} - {entry['timestamp']}**")
                st.markdown(f"🔗 [{entry['url']}]({entry['url']})")
                
                # Show summary directly
                st.markdown("**Summary:**")
                st.markdown(entry['summary'])
                
                # Use columns for transcript access
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write("📜 **Transcript Available**")
                with col2:
                    # Use a button to show/hide transcript
                    show_transcript_key = f"show_transcript_{history_index}"
                    if show_transcript_key not in st.session_state:
                        st.session_state[show_transcript_key] = False
                    
                    if st.button(
                        "Show Transcript" if not st.session_state[show_transcript_key] else "Hide Transcript",
                        key=f"transcript_toggle_{history_index}"
                    ):
                        st.session_state[show_transcript_key] = not st.session_state[show_transcript_key]
                
                # Show transcript if toggled
                if st.session_state.get(show_transcript_key, False):
                    st.text_area(
                        "Transcript:",
                        value=entry['transcript'],
                        height=300,
                        key=f"history_transcript_{history_index}"
                    )

        # Clear history button
        if st.button("🗑️ Clear History", key="clear_youtube_history"):
            # Also clear all transcript toggle states
            keys_to_remove = [key for key in st.session_state.keys() if key.startswith("show_transcript_")]
            for key in keys_to_remove:
                del st.session_state[key]
            st.session_state.youtube_history = []
            st.rerun()

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
        pages = ["Home", "Search", "Chat", "YouTube Summary"] # Added YouTube Summary
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
            if 'selected_history_item_search' in st.session_state:
                 st.session_state.selected_history_item_search = None
            st.rerun() # Rerun to load the new page

    # Page content based on selection
    if st.session_state.current_page == "Home":
        home_page()
    elif st.session_state.current_page == "Search":
        search_page()
    elif st.session_state.current_page == "Chat":
        chat_page()
    elif st.session_state.current_page == "YouTube Summary": # Added YouTube Summary page
        youtube_summary_page()

if __name__ == "__main__":
    # Crucial: Ensure API keys are loaded from secrets into environment variables
    # This is often needed when running locally or deploying if ai_search.py
    # strictly uses os.environ.get()
    # In Streamlit Cloud, you map secrets to environment variables in settings.
    # For local testing, you might do:
    if "SERPER_API_KEY" in st.secrets:
        os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
    if "OPENROUTER_API_KEY" in st.secrets:
        os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
    # Also load GEMINI_API_KEY for the Voice Chat page # Removed
    # if "GEMINI_API_KEY" in st.secrets: # Removed
    #     os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"] # Removed

    main()