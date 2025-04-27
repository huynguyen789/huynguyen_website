# audio_recorder.py
# Purpose: Record audio using Streamlit's audio_input, display it, transcribe using OpenAI API,
# send transcription to LLM, stream response, and automatically speak the final response using OpenAI TTS.

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
import time

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
    model: str = "tts-1", # "gpt-4o-mini-tts" or tts-1 Do not change the model! 
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
    model_name: str = "gpt-4.1-nano-2025-04-14" # Default model
) -> Generator[Tuple[str, str], None, None]:
    """
    Get a streaming response from OpenAI Chat Completion API.

    Input:
        messages (List[Dict[str, str]]): Conversation history/prompt.
        client (Optional[OpenAI]): Synchronous OpenAI client instance.
        model_name (str): The model to use (e.g., "gpt-4o"
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
st.set_page_config(layout="wide", page_title="Voice Chat Assistant")
st.title("Seamless Voice Chat Assistant")

# Initialize session state keys
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'llm_response' not in st.session_state:
    st.session_state.llm_response = ""
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'transcribed_audio_id' not in st.session_state:
    st.session_state.transcribed_audio_id = None
if 'audio_player' not in st.session_state:
    # Initialize player once, requires ffplay/mpv
    try:
        st.session_state.audio_player = LocalAudioPlayer()
        logger.info("LocalAudioPlayer initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize LocalAudioPlayer: {e}", exc_info=True)
        st.error(f"Failed to initialize audio player. Ensure 'ffplay' or 'mpv' is installed and in your PATH, and you have 'sounddevice' installed (`uv pip install sounddevice`). Error: {e}")
        st.session_state.audio_player = None

# Config options in sidebar
with st.sidebar:
    st.header("Configuration")
    ai_model = st.selectbox("AI Model:", ["gpt-4.1-nano-2025-04-14", "gpt-4o"], key="model_select")
    voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    tts_voice = st.selectbox("Voice:", voice_options, index=0, key="voice_select")
    auto_speak = st.checkbox("Auto-speak responses", value=True, key="auto_speak")
    system_prompt = st.text_area(
        "System Prompt:", 
        value="You are a world-class communicator.Prefer concise and direct responses.",
        height=100,
        key="system_prompt_input"
    )
    
    # Display API key status
    if os.getenv("OPENAI_API_KEY"):
        st.success("OpenAI API key detected ✓")
    else:
        st.error("OpenAI API key not found! Set the `OPENAI_API_KEY` environment variable.")
        
    # Display player status
    if st.session_state.audio_player:
        st.success("Audio player initialized ✓")
    else:
        st.error("Audio player not found! Install ffplay/mpv and sounddevice.")

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.header("🎤 Record Your Message")
    
    # Audio input widget
    audio_input = st.audio_input("Click microphone to record:", key="audio_recorder")
    
    # When new audio is recorded
    if audio_input:
        audio_bytes = audio_input.read()
        
        # Check if this is new audio or previously processed audio
        audio_id = hash(audio_bytes)
        is_new_recording = (audio_id != st.session_state.transcribed_audio_id)
        
        if is_new_recording:
            # Reset state for new audio
            st.session_state.transcription = ""
            st.session_state.llm_response = ""
            st.session_state.processing_complete = False
            
            # Display recorded audio
            st.subheader("Recorded Audio")
            st.audio(audio_bytes)
            
            # STEP 1: Automatic Transcription
            st.subheader("Transcription")
            with st.spinner("Transcribing your message..."):
                with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_audio_file:
                    temp_audio_file.write(audio_bytes)
                    temp_audio_file.flush()
                    
                    # Get transcription
                    transcript = transcribe_audio(temp_audio_file.name, sync_client)
                    
            if transcript.startswith("Error:"):
                st.error("Failed to transcribe audio. Please try again.")
            else:
                # Display transcription
                st.success("✓ Transcription complete")
                st.write(transcript)
                st.session_state.transcription = transcript
                st.session_state.transcribed_audio_id = audio_id
                
                # STEP 2: Automatic LLM Processing
                with col2:
                    st.header("🤖 AI Response")
                    with st.spinner(f"Getting response from {ai_model}..."):
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": transcript}
                        ]
                        
                        # Stream the response
                        response_placeholder = st.empty()
                        full_response = ""
                        
                        response_generator = get_llm_response_stream(messages, sync_client, ai_model)
                        for chunk, accumulated_response in response_generator:
                            if chunk.startswith("Error:"):
                                st.error("Failed to get AI response.")
                                break
                            full_response = accumulated_response
                            response_placeholder.markdown(full_response + "▌")
                        
                        # Final response display
                        response_placeholder.markdown(full_response)
                        st.session_state.llm_response = full_response
                        
                        # Mark processing as complete
                        st.session_state.processing_complete = True
                    
                    # STEP 3: Automatic TTS if enabled
                    if st.session_state.processing_complete and auto_speak and st.session_state.llm_response:
                        if st.session_state.audio_player and async_client:
                            speak_button = st.button("🔊 Speak Response Again", key="speak_again_button")
                            
                            # Either auto-speak or speak when button is clicked
                            if auto_speak or speak_button:
                                with st.spinner("Converting text to speech..."):
                                    run_async(speak_text_streaming_async(
                                        st.session_state.llm_response,
                                        async_client,
                                        voice=tts_voice,
                                        player=st.session_state.audio_player
                                    ))
                        else:
                            if not async_client:
                                st.error("Cannot speak: OpenAI client not initialized.")
                            elif not st.session_state.audio_player:
                                st.error("Cannot speak: Audio player not available.")
        else:
            # This is previously processed audio, just show the results
            st.subheader("Recorded Audio")
            st.audio(audio_bytes)
            
            if st.session_state.transcription:
                st.subheader("Transcription")
                st.write(st.session_state.transcription)
    else:
        # No audio recorded yet
        st.info("👆 Click the microphone icon above to start recording your message.")

# Display footer only if no audio recorded (to save space)
if not audio_input:
    st.write("---")
    st.caption("This app uses OpenAI for speech-to-text, AI responses, and text-to-speech. For TTS playback, it requires ffplay/mpv and the sounddevice package.")
