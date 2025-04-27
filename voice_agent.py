# voice_agent.py
# Purpose: Record audio from the user's microphone, save to a temp file, and transcribe using OpenAI API.

from openai import OpenAI
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import numpy as np
import keyboard
import threading
import time

def record_audio(duration: int, sample_rate: int = 44100) -> str:
    """
    Record audio from the microphone and save to a temporary WAV file.
    
    Input:
        duration (int): Duration of the recording in seconds.
        sample_rate (int): Sampling rate in Hz.
    Process:
        Records audio from the default microphone and writes it to a temp WAV file.
    Output:
        str: Path to the temporary WAV file.
    """
    try:
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        write(temp_file.name, sample_rate, audio)
        print(f"Audio recorded and saved to {temp_file.name}")
        return temp_file.name
    except Exception as e:
        print(f"Error during recording: {e}")
        raise

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

def record_audio_manual(sample_rate: int = 44100) -> str:
    """
    Record audio from the microphone, starting and stopping with keyboard input.
    
    Input:
        sample_rate (int): Sampling rate in Hz.
    Process:
        Waits for user to press 'r' to start recording and 's' to stop.
        Records audio from the default microphone and writes it to a temp WAV file.
    Output:
        str: Path to the temporary WAV file.
    """
    print("Press 'r' to start recording...")
    keyboard.wait('r')
    print("Recording... Press 's' to stop.")
    recording = []
    is_recording = True

    def record_thread():
        nonlocal recording, is_recording
        audio = sd.rec(int(60 * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        while is_recording:
            time.sleep(0.1)
        sd.stop()
        recording = audio[:sd.get_stream().time * sample_rate]

    is_recording = True
    t = threading.Thread(target=record_thread)
    t.start()
    keyboard.wait('s')
    is_recording = False
    t.join()

    # Save the recorded audio
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    write(temp_file.name, sample_rate, sd.get_stream().read()[0])
    print(f"Audio recorded and saved to {temp_file.name}")
    return temp_file.name

if __name__ == "__main__":
    client = OpenAI()
    # audio_path = record_audio(duration=5)  # Old fixed-duration method
    audio_path = record_audio_manual()        # New manual control method
    text = transcribe_audio(audio_path, client)
    print("Transcription:", text)