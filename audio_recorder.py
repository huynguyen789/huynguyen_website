# audio_recorder.py
# Purpose: Record audio using Streamlit's audio_input, display it, transcribe using OpenAI API,
# send transcription to LLM, stream response, and speak the final response using OpenAI TTS.

import streamlit as st
from openai import OpenAI, AsyncOpenAI
import tempfile
import os
import asyncio
# LocalAudioPlayer requires ffplay or mpv installed on the system.
# It also relies on the 'sounddevice' Python package: uv pip install sounddevice
from openai.helpers import LocalAudioPlayer
import sys
from typing import List, Dict, Generator, Tuple, Optional # Added typing imports
import logging # Added logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- OpenAI Client Initialization ---
# Note: Ensure OPENAI_API_KEY environment variable is set.
try:
    # Keep the synchronous client for transcription and initial LLM call
    sync_client = OpenAI()
    # Create an async client for TTS streaming
    async_client = AsyncOpenAI()
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
    st.error(f"Failed to initialize OpenAI client. Have you set the OPENAI_API_KEY environment variable? Error: {e}")
    sync_client = None
    async_client = None

# --- Helper function to run async code in Streamlit ---
# This attempts to get or create an event loop for the current thread,
# which is necessary for running async functions within Streamlit's sync environment.
def run_async(func):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            raise
    return loop.run_until_complete(func)

# --- OpenAI Transcription Function ---
def transcribe_audio(file_path: str, client: OpenAI) -> str:
    """
    Transcribe an audio file using OpenAI's API.

    Input:
        file_path (str): Path to the audio file.
        client (OpenAI): Synchronous OpenAI client instance.
    Process:
        Sends the audio file to OpenAI's transcription API. Handles potential errors.
    Output:
        str: Transcribed text, or an error message if transcription fails.
    """
    if not client:
        logger.warning("Transcription skipped: OpenAI client not initialized.")
        return "Error: OpenAI client not initialized."
    try:
        with open(file_path, "rb") as audio_file:
            logger.info(f"Transcribing audio file: {os.path.basename(file_path)}")
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=(os.path.basename(file_path), audio_file)
            )
            logger.info("Transcription successful.")
        return transcription.text
    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        error_msg = f"Error during transcription: {e}"
        st.error(error_msg)
        return f"Error: Could not transcribe audio. {e}"

# --- OpenAI TTS Streaming Function ---
async def speak_text_streaming_async(
    text: str,
    client: AsyncOpenAI,
    voice: str = "alloy", # Changed default voice slightly
    model: str = "tts-1", # Standard TTS model
    instructions: str = "Speak in a clear, natural tone.",
    player: LocalAudioPlayer = None
) -> None:
    """
    Asynchronously stream spoken audio from input text using OpenAI TTS.

    Input:
        text (str): The text to be spoken.
        client (AsyncOpenAI): Asynchronous OpenAI client instance.
        voice (str): The voice to use.
        model (str): The TTS model to use.
        instructions (str): Instructions for the TTS model.
        player (LocalAudioPlayer): An instance of LocalAudioPlayer.
    Process:
        Streams audio from OpenAI and plays it using the provided player.
    Output:
        None
    """
    if not client:
        logger.warning("TTS skipped: Async OpenAI client not initialized.")
        st.error("Error: Async OpenAI client not initialized.")
        return
    if not text:
        logger.warning("TTS skipped: No text provided.")
        st.warning("No text provided to speak.")
        return
    if player is None:
        logger.warning("TTS skipped: Audio player not provided.")
        st.error("Audio player not provided.")
        return

    try:
        logger.info(f"Starting TTS streaming for text: \"{text[:50]}...\"")
        st.info("Disclosure: The following voice is AI-generated.")

        async with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
            instructions=instructions,
            response_format="pcm",  # PCM is generally good for low-latency streaming
        ) as response:
            logger.info("Playing TTS stream...")
            await player.play(response)
            logger.info("TTS stream finished playing.")
    except Exception as e:
        logger.error(f"Error during text-to-speech streaming: {e}", exc_info=True)
        st.error(f"Error during text-to-speech streaming: {e}")

# --- OpenAI LLM Streaming Function (Simplified) ---
def get_llm_response_stream(
    messages: List[Dict[str, str]],
    client: Optional[OpenAI] = None,
    model_name: str = "gpt-4o" # Default model
) -> Generator[Tuple[str, str], None, None]:
    """
    Get a streaming response from OpenAI Chat Completion API.

    Input:
        messages (List[Dict[str, str]]): Conversation history/prompt.
        client (Optional[OpenAI]): Synchronous OpenAI client instance.
        model_name (str): The model to use (e.g., "gpt-4o", "gpt-3.5-turbo").
    Process:
        Sends the messages to the specified OpenAI model and streams the response.
    Output:
        Generator[Tuple[str, str], None, None]: Yields (chunk, full_response_so_far).
                                                 Yields ("Error: ...", "Error: ...") on failure.
    """
    if client is None:
        logger.error("LLM call failed: OpenAI client is None.")
        yield "Error: OpenAI client not initialized.", "Error: OpenAI client not initialized."
        return

    try:
        logger.info(f"Sending request to LLM model: {model_name}")
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
        )
        full_response = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
                yield content, full_response # Yield chunk and accumulated response
        logger.info("LLM streaming finished.")
    except Exception as e:
        error_msg = f"Error generating LLM stream from {model_name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg) # Show error in Streamlit
        yield error_msg, error_msg # Yield error as chunk and full response

# --- Streamlit App ---
st.set_page_config(layout="wide") # Use wider layout
st.title("Voice Note AI Assistant")

# Initialize session state keys
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'llm_response' not in st.session_state:
    st.session_state.llm_response = ""
if 'audio_player' not in st.session_state:
    # Initialize player once, requires ffplay/mpv
    try:
        st.session_state.audio_player = LocalAudioPlayer()
        logger.info("LocalAudioPlayer initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize LocalAudioPlayer: {e}", exc_info=True)
        st.error(f"Failed to initialize audio player. Ensure 'ffplay' or 'mpv' is installed and in your PATH, and you have 'sounddevice' installed (`uv pip install sounddevice`). Error: {e}")
        st.session_state.audio_player = None

# --- Recording and Transcription Column ---
col1, col2 = st.columns(2)

with col1:
    st.header("1. Record & Transcribe")
    audio_input = st.audio_input(
        "Record a voice message (click microphone):",
        key="audio_recorder"
    )

    if audio_input:
        st.subheader("Recorded Audio")
        audio_bytes = audio_input.read()
        st.audio(audio_bytes)

        # --- Transcription ---
        # Use a flag to avoid re-transcribing after LLM interaction within the same audio upload
        if 'transcribed_current_audio' not in st.session_state or not st.session_state.transcribed_current_audio:
            st.subheader("Transcription")
            with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_audio_file:
                temp_audio_file.write(audio_bytes)
                temp_audio_file.flush()
                audio_path = temp_audio_file.name

                with st.spinner("Transcribing audio..."):
                    transcript = transcribe_audio(audio_path, sync_client)

                if transcript.startswith("Error:"):
                    st.session_state.transcription = "" # Reset on error
                    st.session_state.transcribed_current_audio = False
                else:
                    st.success("Transcription complete!")
                    st.session_state.transcription = transcript # Store result
                    st.session_state.llm_response = "" # Clear previous LLM response
                    st.session_state.transcribed_current_audio = True # Mark as transcribed
        
        # Display transcription in a text area for potential editing
        if st.session_state.transcription and not st.session_state.transcription.startswith("Error:"):
             st.session_state.transcription = st.text_area(
                 "Editable Transcription:",
                 value=st.session_state.transcription,
                 height=150,
                 key="transcription_edit"
             )

    else:
        st.info("Click the microphone icon above to record your message.")
        # Clear state when no audio is present
        st.session_state.transcription = ""
        st.session_state.llm_response = ""
        st.session_state.transcribed_current_audio = False

# --- LLM Interaction and Speaking Column ---
with col2:
    st.header("2. AI Response & Speaking")

    # Only show this section if we have a valid transcription
    if st.session_state.transcription and not st.session_state.transcription.startswith("Error:"):

        # System prompt (optional, could be user-configurable)
        system_prompt = "You are a helpful assistant. Respond clearly and concisely to the user's transcribed message." 
        # st.text_input("System Prompt:", value=system_prompt, key="system_prompt_input")
        
        ai_model = st.selectbox("Select AI Model:", ["gpt-4o", "gpt-3.5-turbo"], key="model_select")

        if st.button(f"💬 Get {ai_model} Response", key="get_response_button"):
            st.session_state.llm_response = "" # Clear previous response
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": st.session_state.transcription}
            ]
            
            st.subheader("AI Response Stream")
            response_placeholder = st.empty() # Placeholder for the streaming text
            full_response = ""
            
            with st.spinner(f"Waiting for {ai_model}..."):
                response_generator = get_llm_response_stream(messages, sync_client, ai_model)
            
                for chunk, accumulated_response in response_generator:
                    if chunk.startswith("Error:"):
                        st.session_state.llm_response = chunk # Store error message
                        break # Stop processing on error
                    full_response = accumulated_response
                    response_placeholder.markdown(full_response + "▌") # Display stream with cursor
                response_placeholder.markdown(full_response) # Final display without cursor
                st.session_state.llm_response = full_response # Store final response

        # --- Speak LLM Response Button ---
        if st.session_state.llm_response and not st.session_state.llm_response.startswith("Error:"):
            st.write("---")
            st.subheader("Speak AI Response")
            if st.session_state.audio_player and async_client:
                if st.button("🔊 Speak AI Response", key="speak_response_button"):
                     with st.spinner("Generating and streaming audio..."):
                        run_async(speak_text_streaming_async(
                            st.session_state.llm_response,
                            async_client,
                            player=st.session_state.audio_player
                        ))
            elif not async_client:
                st.error("Cannot speak: Async OpenAI client not initialized.")
            elif not st.session_state.audio_player:
                 st.error("Cannot speak: Audio player (ffplay/mpv) not initialized or found.")

    else:
        st.info("Record a message and get a transcription first.")

# --- Footer Notes ---
st.write("---")
if not os.getenv("OPENAI_API_KEY"):
    st.warning("OpenAI API key not found. Please set the `OPENAI_API_KEY` environment variable for transcription, AI responses, and TTS to work.")

st.caption("Text-to-Speech requires 'ffplay' or 'mpv' to be installed and accessible in your system's PATH, and the `sounddevice` Python package (`uv pip install sounddevice`).")
