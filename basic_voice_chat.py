"""
basic_voice_chat.py
-------------------

A minimal voice chat script using Google Gemini Live API.

Input:
    - Microphone audio recorded at 16 kHz, 16-bit, mono PCM.
Process:
    1. Records the user's voice.
    2. Sends audio to the Gemini Live API via `send_realtime_input`.
    3. Listens for streamed audio response (24 kHz, 16-bit PCM).
    4. Plays the response back to the user.
Output:
    - Spoken response from Gemini played through speakers.

Prerequisites:
    pip install -U google-genai sounddevice simpleaudio numpy
    export GEMINI_API_KEY="YOUR_API_KEY"

Usage:
    python basic_voice_chat.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Final

import numpy as np  # type: ignore
import sounddevice as sd  # type: ignore
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

API_KEY: Final[str | None] = os.getenv("GEMINI_API_KEY")
MODEL_NAME: Final[str] = "gemini-2.0-flash-live-001"
RESPONSE_SAMPLE_RATE: Final[int] = 24000  # Hz, per documentation
INPUT_SAMPLE_RATE: Final[int] = 16000  # Hz, per documentation

if not API_KEY:
    raise EnvironmentError(
        "Missing GEMINI_API_KEY environment variable. Set it before running the script."
    )

client = genai.Client(api_key=API_KEY)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def record_audio(duration_sec: float = 4.0) -> bytes:
    """Record audio from the default microphone.

    Input:
        duration_sec: Desired length of the recording in seconds.
    Process:
        • Captures mono audio at 16 kHz, 16-bit using sounddevice.
    Output:
        Raw little-endian PCM bytes suitable for Live API.
    """

    print(f"Recording for {duration_sec:.1f}s… Speak now.")
    frames = sd.rec(
        int(duration_sec * INPUT_SAMPLE_RATE),
        samplerate=INPUT_SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    print("Recording finished.\n")
    return frames.flatten().tobytes()


def play_audio(pcm_data: bytes) -> None:
    """Play raw PCM audio through the system default output.

    Input:
        pcm_data: 16-bit little-endian mono PCM at 24 kHz.
    Process:
        • Converts to NumPy int16 array and streams via sounddevice.
    Output:
        None (audio is played back).
    """

    if not pcm_data:
        print("[Warning] Empty audio data received; nothing to play.")
        return

    # Ensure even number of bytes (whole int16 samples).
    if len(pcm_data) % 2 != 0:
        pcm_data = pcm_data[:-1]

    try:
        audio_np = np.frombuffer(pcm_data, dtype=np.int16)
        sd.play(audio_np, samplerate=RESPONSE_SAMPLE_RATE, blocking=True)
        sd.stop()
    except Exception as exc:
        # Fallback for environments where PortAudio crashes.
        print(f"[Warning] sounddevice playback failed ({exc}); falling back to afplay.")

        try:
            import tempfile, wave, subprocess, sys

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                with wave.open(tmp.name, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(RESPONSE_SAMPLE_RATE)
                    wf.writeframes(pcm_data)

                # macOS: use afplay; Linux: aplay; Windows: start /min wmplayer maybe.
                if sys.platform == "darwin":
                    subprocess.run(["afplay", tmp.name], check=False)
                else:
                    subprocess.run(["aplay", tmp.name], check=False)
        except Exception as exc2:
            print(f"[Error] Fallback playback also failed: {exc2}")


# -----------------------------------------------------------------------------
# Core async chat routine
# -----------------------------------------------------------------------------


async def chat() -> None:
    """Manage an interactive, bidirectional voice chat session with Gemini."""

    config = {"response_modalities": ["AUDIO"]}

    async with client.aio.live.connect(model=MODEL_NAME, config=config) as session:
        while True:
            user_input = input("Press <Enter> to record, or type 'exit' to quit: ").strip()
            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            audio_bytes = record_audio()

            # Send the recorded audio to Gemini.
            await session.send_realtime_input(
                audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
            )
            # Indicate that the audio stream has ended for this turn.
            await session.send_realtime_input(audio_stream_end=True)

            response_audio = bytearray()
            async for response in session.receive():
                # Collect the streamed audio chunks.
                if response.data is not None:
                    response_audio.extend(response.data)

                # Stop collecting when Gemini signals the turn is complete.
                if (
                    response.server_content is not None
                    and response.server_content.turn_complete is True
                ):
                    break

            print("Gemini is replying…")
            play_audio(bytes(response_audio))


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> None:
    """Run the async chat wrapper via asyncio."""

    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        print("\nSession terminated by user.")


if __name__ == "__main__":
    main()
