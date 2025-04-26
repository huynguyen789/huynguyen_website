import asyncio
import os
import tempfile
import streamlit as st
import pyaudio
import wave
from google import genai
from google.genai import types
import time
import base64

# Set page config
st.set_page_config(page_title="Gemini Voice Chat", page_icon="🎤")

# Initialize session state variables if they don't exist
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "last_response_played" not in st.session_state:
    st.session_state.last_response_played = False

# Streamlit UI
st.title("Gemini Voice Chat")

# API key input
# Load API key from environment or secrets
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("🚨 GEMINI_API_KEY not found in Streamlit Secrets!")
    st.stop()

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    # Show recording indicator
    with st.spinner("Recording..."):
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save as raw PCM
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pcm')
    with open(temp_file.name, 'wb') as f:
        for frame in frames:
            f.write(frame)

    return temp_file.name

def get_audio_html(file_path, autoplay=False):
    """Generate HTML for an audio element with optional autoplay."""
    audio_file = open(file_path, 'rb')
    audio_bytes = audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_file.close()
    
    autoplay_attr = "autoplay" if autoplay else ""
    html = f"""
    <audio controls {autoplay_attr}>
        <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
    </audio>
    """
    return html

async def process_voice_chat(api_key, input_file):
    client = genai.Client(api_key=api_key)
    model = "gemini-2.0-flash-live-001"

    config = {
        "response_modalities": ["AUDIO"],
        "speech_config": types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
            )
        )
    }

    async with client.aio.live.connect(model=model, config=config) as session:
        # Read the recorded audio
        audio_bytes = open(input_file, 'rb').read()

        # Send audio to Gemini
        await session.send_realtime_input(
            audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
        )

        # Signal end of audio stream
        await session.send_realtime_input(audio_stream_end=True)

        # Prepare to save Gemini's response
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        wf = wave.open(output_file, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)  # Output is 24kHz

        with st.spinner("Waiting for response..."):
            async for response in session.receive():
                if response.data is not None:
                    wf.writeframes(response.data)

                # Check if generation is complete
                if hasattr(response, 'server_content') and hasattr(response.server_content, 'generation_complete'):
                    if response.server_content.generation_complete:
                        break

        wf.close()
        return output_file

# Display conversation history
for i, (role, audio_file) in enumerate(st.session_state.conversation_history):
    with st.chat_message(role):
        st.write(f"{role.capitalize()} audio:")
        
        # For the most recent assistant response that hasn't been played yet
        if role == "assistant" and i == len(st.session_state.conversation_history) - 1 and not st.session_state.last_response_played:
            st.components.v1.html(get_audio_html(audio_file, autoplay=True), height=50)
            st.session_state.last_response_played = True
        else:
            st.audio(audio_file)

# Record button
if st.button("Record (5 seconds)"):
    if not api_key:
        st.error("Please enter your Gemini API key first.")
    else:
        # Record user's voice
        input_file = record_audio()
        st.session_state.conversation_history.append(("user", input_file))

        # Process with Gemini
        output_file = asyncio.run(process_voice_chat(api_key, input_file))
        st.session_state.conversation_history.append(("assistant", output_file))
        
        # Reset flag to indicate the new response should be auto-played
        st.session_state.last_response_played = False
        
        # Force a rerun to update the conversation display
        st.rerun()

# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state.conversation_history = []
    st.session_state.last_response_played = False
    st.rerun()