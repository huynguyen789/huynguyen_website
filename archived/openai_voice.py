# voice_agent.py
# Purpose: Record audio from the user's microphone, save to a temp file, and transcribe using OpenAI API.

from openai import OpenAI
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import numpy as np
import threading
import queue
import sys
import time
import asyncio
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

def record_audio_interactive(sample_rate: int = 44100) -> str:
    """
    Record audio from the microphone interactively (press Enter to start/stop) and save to a temporary WAV file.

    Input:
        sample_rate (int): Sampling rate in Hz.
    Process:
        Waits for user to press Enter to start, records until Enter is pressed again, then saves to temp WAV file.
    Output:
        str: Path to the temporary WAV file.
    """
    print("Press Enter to start recording...")
    input()
    print("Recording... Press Enter to stop.")

    frames = []

    def callback(indata, frames_count, time_info, status):
        frames.append(indata.copy())

    # Open the stream and record until Enter is pressed again
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', callback=callback):
        input()  # Wait for Enter to stop recording

    print("Recording stopped.")

    if not frames:
        print("No audio recorded.")
        return ""

    audio = np.concatenate(frames, axis=0)
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    write(temp_file.name, sample_rate, audio)
    print(f"Audio recorded and saved to {temp_file.name}")
    return temp_file.name

def transcribe_audio(file_path: str, client: OpenAI) -> str:
    """
    Transcribe an audio file using OpenAI's API.
    
    Input:
        file_path (str): Path to the audio file.
        client (OpenAI): OpenAI client instance.
    Process:
        Sends the audio file to OpenAI's transcription API.
    Output:
        str: Transcribed text.
    """
    try:
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file
            )
        return transcription.text
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise

def speak_text_streaming(text: str, voice: str = "coral", model: str = "gpt-4o-mini-tts", instructions: str = "Speak in a clear, natural tone.") -> None:
    """
    Stream spoken audio from input text directly to speakers using OpenAI TTS.

    Input:
        text (str): The text to be spoken.
        voice (str): The voice to use (e.g., "coral").
        model (str): The TTS model to use.
        instructions (str): Instructions for the TTS model.
    Process:
        Streams audio from OpenAI and plays it in real time.
    Output:
        None
    """
    async def _speak():
        openai = AsyncOpenAI()
        print("Disclosure: The following voice is AI-generated, not a human voice.")
        async with openai.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
            instructions=instructions,
            response_format="pcm",  # Fastest for streaming
        ) as response:
            await LocalAudioPlayer().play(response)
    asyncio.run(_speak())

if __name__ == "__main__":
    client = OpenAI()
    audio_path = record_audio_interactive()
    if audio_path:
        text = transcribe_audio(audio_path, client)
        print("Transcription:", text)
        if text:  # Only speak if transcription is not empty
            speak_text_streaming(text,instructions="speak softly like a story teller")
        else:
            print("No transcription to speak.")