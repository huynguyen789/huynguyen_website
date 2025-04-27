"""
Simple Desktop App for Gemini Live API
Input: User selects mode, enters text, clicks buttons.
Process: Starts/stops AudioLoop, sends messages, displays responses.
Output: Model responses shown in the app.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import asyncio
import threading

from geminiLive import AudioLoop

class GeminiApp:
    def __init__(self, root):
        """
        Input: root (tk.Tk): Tkinter root window.
        Process: Sets up UI and event handlers.
        Output: None
        """
        self.root = root
        self.root.title("Gemini Live Desktop App")

        # Mode selection
        self.mode_var = tk.StringVar(value="camera")
        ttk.Label(root, text="Video Mode:").pack(anchor="w")
        ttk.Combobox(root, textvariable=self.mode_var, values=["camera", "screen", "none"]).pack(fill="x")

        # Output display
        self.output = scrolledtext.ScrolledText(root, height=10, state="disabled")
        self.output.pack(fill="both", expand=True, padx=5, pady=5)

        # Text input
        self.input_var = tk.StringVar()
        input_frame = ttk.Frame(root)
        input_frame.pack(fill="x", padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.input_var).pack(side="left", fill="x", expand=True)
        ttk.Button(input_frame, text="Send", command=self.send_message).pack(side="right")

        # Start/Stop buttons
        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill="x", padx=5, pady=5)
        self.start_btn = ttk.Button(btn_frame, text="Start", command=self.start)
        self.start_btn.pack(side="left", expand=True, fill="x")
        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="right", expand=True, fill="x")

        self.audio_loop = None
        self.loop_thread = None
        self.running = False

    def display_output(self, text: str):
        """
        Input: text (str): Text to display.
        Process: Appends text to output box.
        Output: None
        """
        self.output.config(state="normal")
        self.output.insert("end", text + "\n")
        self.output.see("end")
        self.output.config(state="disabled")

    def start(self):
        """
        Input: None
        Process: Starts the AudioLoop in a background thread.
        Output: None
        """
        if self.running:
            return
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.audio_loop = AudioLoop(video_mode=self.mode_var.get(), text_callback=self.display_output)
        self.loop_thread = threading.Thread(target=self.run_asyncio_loop, daemon=True)
        self.loop_thread.start()

    def run_asyncio_loop(self):
        """
        Input: None
        Process: Runs the AudioLoop's run() method in an asyncio event loop.
        Output: None
        """
        asyncio.run(self.audio_loop.run())

    def send_message(self):
        """
        Input: None
        Process: Sends user input to the model.
        Output: None
        """
        if not self.audio_loop or not self.running:
            messagebox.showwarning("Not running", "Start the session first.")
            return
        text = self.input_var.get()
        self.input_var.set("")
        asyncio.run_coroutine_threadsafe(self.audio_loop.send_text(text), asyncio.get_event_loop())

    def stop(self):
        """
        Input: None
        Process: Stops the AudioLoop and cleans up.
        Output: None
        """
        if not self.running:
            return
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        if self.audio_loop:
            asyncio.run_coroutine_threadsafe(self.audio_loop.stop(), asyncio.get_event_loop())
        self.display_output("Session stopped.")

if __name__ == "__main__":
    root = tk.Tk()
    app = GeminiApp(root)
    root.mainloop() 