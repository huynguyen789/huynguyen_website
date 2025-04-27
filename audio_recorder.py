# audio_recorder.py
# Purpose: Record audio using Streamlit's audio_input, display it, transcribe using OpenAI API,
# and speak the transcription using OpenAI TTS streaming.

import streamlit as st
from openai import OpenAI, AsyncOpenAI
import tempfile
import os
import asyncio
# LocalAudioPlayer requires ffplay or mpv installed on the system.
# It also relies on the 'sounddevice' Python package: uv pip install sounddevice
from openai.helpers import LocalAudioPlayer
import sys

# --- OpenAI Client Initialization ---
# Note: Ensure OPENAI_API_KEY environment variable is set.
try:
    # Keep the synchronous client for transcription
    sync_client = OpenAI()
    # Create an async client for TTS streaming
    async_client = AsyncOpenAI()
except Exception as e:
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
        return "Error: OpenAI client not initialized."
    try:
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=(os.path.basename(file_path), audio_file)
            )
        return transcription.text
    except Exception as e:
        # Use st.error for Streamlit display, but also return the error string
        error_msg = f"Error during transcription: {e}"
        st.error(error_msg)
        return f"Error: Could not transcribe audio. {e}" # Return the specific error

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
        st.error("Error: Async OpenAI client not initialized.")
        return
    if not text:
        st.warning("No text provided to speak.")
        return
    if player is None:
        st.error("Audio player not provided.")
        return

    try:
        # Display disclosure in Streamlit
        st.info("Disclosure: The following voice is AI-generated.")

        async with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
            instructions=instructions,
            response_format="pcm",  # PCM is generally good for low-latency streaming
        ) as response:
            # Play the streamed audio chunks
            await player.play(response) # Removed await as play is not async
    except Exception as e:
        st.error(f"Error during text-to-speech streaming: {e}")


# --- Streamlit App ---

st.title("Audio Recorder, Transcriber, and Speaker")

# Initialize session state for transcription result
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""

audio_input = st.audio_input("Record a voice message (click the microphone icon):", key="audio_recorder")

if audio_input:
    st.write("---")
    st.subheader("Recorded Audio")
    audio_bytes = audio_input.read()
    st.audio(audio_bytes) # Display the recorded audio

    st.write("---")
    st.subheader("Transcription")

    # Only transcribe if we haven't already for this audio input
    # This prevents re-transcribing on every interaction after recording
    # Comparing audio_bytes directly might be inefficient for large files,
    # but works for typical voice messages. A hash could be better.
    # We'll simply check if transcription is empty for now.
    if not st.session_state.transcription:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(audio_bytes)
            temp_audio_file.flush()
            audio_path = temp_audio_file.name

            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio(audio_path, sync_client)

            if transcript.startswith("Error:"):
                # Error already displayed by transcribe_audio
                st.session_state.transcription = "" # Reset on error
            else:
                st.success("Transcription complete!")
                st.write(transcript)
                st.session_state.transcription = transcript # Store result

    # --- Speak Button ---
    # Show transcription and speak button only if transcription exists
    if st.session_state.transcription and not st.session_state.transcription.startswith("Error:"):
        st.write("---")
        st.subheader("Speak Transcription")

        # Create the player instance here, maybe cache it?
        # For now, create it on demand. Ensure backend (ffplay/mpv) is installed.
        try:
            # Check if ffplay or mpv exists before creating player
            # This requires checking system paths or using shutil.which
            # For simplicity, we'll let LocalAudioPlayer handle the check internally for now.
            audio_player = LocalAudioPlayer()
        except Exception as e:
            st.error(f"Failed to initialize audio player. Ensure 'ffplay' or 'mpv' is installed and in your PATH. Error: {e}")
            audio_player = None # Set to None if init fails

        if audio_player and st.button("🔊 Speak Transcription"):
            if async_client: # Check if async client is initialized
                with st.spinner("Generating and streaming audio..."):
                     # Use the helper to run the async function
                    run_async(speak_text_streaming_async(
                        st.session_state.transcription,
                        async_client,
                        player=audio_player
                    ))
            else:
                st.error("Cannot speak: Async OpenAI client not initialized.")

else:
    st.info("Click the microphone icon above to record your message.")
    st.session_state.transcription = "" # Clear transcription if no audio input

# Add a note about API Key and TTS dependencies
if not os.getenv("OPENAI_API_KEY"):
    st.warning("OpenAI API key not found. Please set the `OPENAI_API_KEY` environment variable for transcription and TTS to work.")
