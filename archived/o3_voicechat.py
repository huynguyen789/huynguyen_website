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
import sys
import time

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

API_KEY: Final[str | None] = os.getenv("GEMINI_API_KEY")
MODEL_NAME: Final[str] = "gemini-2.0-flash-live-001"
RESPONSE_SAMPLE_RATE: Final[int] = 24000  # Hz, per documentation
INPUT_SAMPLE_RATE: Final[int] = 16000  # Hz, per documentation
_CHUNK_MS: Final[int] = 50            # Buffer size in milliseconds (~800 frames)
_SILENCE_MS: Final[int] = 1200        # Stop after this many ms of silence *after speech started*
_SILENCE_THRESH: Final[int] = 500     # RMS level (~0–32767) considered "silent"

if not API_KEY:
    raise EnvironmentError(
        "Missing GEMINI_API_KEY environment variable. Set it before running the script."
    )

client = genai.Client(api_key=API_KEY)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _rms_level(data: np.ndarray) -> float:
    """Compute the root-mean-square (volume) of an int16 NumPy chunk."""

    return float(np.sqrt((data.astype(np.int64) ** 2).mean()))


def record_audio_vad() -> bytes:
    """Record audio until *after* the speaker finishes (>=1s silence).

    Input   : None (user presses <Enter> to start recording).
    Process :
        1. Capture 50-ms chunks from the default microphone (16-kHz, mono).
        2. Wait until at least one chunk exceeds the silence threshold –
           this marks the start of speech.
        3. Keep recording; once speech has started, stop when we observe
           `_SILENCE_MS` of consecutive silence.
    Output  : Raw little-endian 16-bit PCM bytes ready for Gemini.
    """

    print("Press <Enter> to start recording…", end="", flush=True)
    input()  # Wait for the user to start
    print("Recording – speak now. (Auto-stop after ~1 s silence)")

    # Storage for chunks while we are recording
    chunks: list[np.ndarray] = []

    # State for silence detection
    speech_started = False
    silence_accum_ms = 0

    blocksize = int(INPUT_SAMPLE_RATE * _CHUNK_MS / 1000)

    def _callback(indata, frames, time_info, status):  # noqa: D401
        """Stream callback: just add data to our list."""

        if status:
            print(status, file=sys.stderr)
        chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=INPUT_SAMPLE_RATE,
        channels=1,
        dtype="int16",
        callback=_callback,
        blocksize=blocksize,
    ):
        # The stream runs in the background; we poll length of chunks list.
        while True:
            if not chunks:
                time.sleep(_CHUNK_MS / 1000)  # Wait for first data
                continue

            # Work on the *latest* chunk only.
            current_chunk = chunks[-1]
            rms = _rms_level(current_chunk)

            if rms >= _SILENCE_THRESH:
                speech_started = True
                silence_accum_ms = 0  # reset when we hear speech
            else:
                if speech_started:
                    silence_accum_ms += _CHUNK_MS
                    if silence_accum_ms >= _SILENCE_MS:
                        break

            # Avoid tight loop
            time.sleep(_CHUNK_MS / 1000)

    print("Recording finished.\n")

    # Concatenate chunks into a single PCM byte string
    full_audio = np.concatenate(chunks).flatten()
    return full_audio.tobytes()


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

            audio_bytes = record_audio_vad()

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
