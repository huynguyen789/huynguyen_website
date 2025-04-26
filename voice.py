import asyncio
import wave
import pyaudio
from google import genai
from google.genai import types

# Initialize the Gemini client
client = genai.Client(api_key="GEMINI_API_KEY")
model = "gemini-2.0-flash-live-001"
config = {"response_modalities": ["AUDIO"]}

# Audio I/O settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000            # Input sample rate (pcm @16 kHz)
OUTPUT_RATE = 24000     # Gemini returns pcm @24 kHz
CHUNK = 1024
RECORD_SECONDS = 5      # how long to record each turn

async def main():
    pa = pyaudio.PyAudio()
    # open microphone stream
    stream_in = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    async with client.aio.live.connect(model=model, config=config) as session:
        print("Voice chat started. Press Ctrl+C to exit.")
        try:
            while True:
                # 1) Record user audio
                print(f"Recording for {RECORD_SECONDS} seconds…")
                frames = [
                    stream_in.read(CHUNK)
                    for _ in range(int(RATE / CHUNK * RECORD_SECONDS))
                ]
                audio_bytes = b"".join(frames)

                # 2) Send it and signal end-of-stream
                await session.send_realtime_input(
                    audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
                )
                await session.send_realtime_input(audio_stream_end=True)

                # 3) Collect Gemini's audio response
                print("Waiting for response…")
                response_audio = b""
                async for response in session.receive():
                    if response.data is not None:
                        response_audio += response.data
                    # stop when generation is complete
                    if response.server_content and getattr(response.server_content, "generation_complete", False):
                        break

                # 4) Write response to WAV file
                wf = wave.open("response.wav", "wb")
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pa.get_sample_size(FORMAT))
                wf.setframerate(OUTPUT_RATE)
                wf.writeframes(response_audio)
                wf.close()

                # 5) Play it back
                wf2 = wave.open("response.wav", "rb")
                stream_out = pa.open(
                    format=pa.get_format_from_width(wf2.getsampwidth()),
                    channels=wf2.getnchannels(),
                    rate=wf2.getframerate(),
                    output=True,
                )
                data = wf2.readframes(CHUNK)
                while data:
                    stream_out.write(data)
                    data = wf2.readframes(CHUNK)
                stream_out.stop_stream()
                stream_out.close()

        except KeyboardInterrupt:
            print("\nExiting…")
        finally:
            stream_in.stop_stream()
            stream_in.close()
            pa.terminate()

if __name__ == "__main__":
    asyncio.run(main())