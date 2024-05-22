import subprocess
import sys
import tkinter as tk
from tkinter import scrolledtext, ttk
import os
import random
import threading
import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from gif_handler import GIFHandler  # Assuming GIFHandler is implemented in gif_handler.py
from gtts import gTTS

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Required packages
required_packages = [
    "tensorboard",
    "torch",
    "transformers",
    "googletrans==4.0.0-rc1",
    "gtts",
    "Pillow",
    "contextlib2"
]

# Install required packages
for package in required_packages:
    try:
        __import__(package.split('==')[0])  # Handle versioned packages
    except ImportError:
        install(package)

class ChatAppGUI:
    def __init__(self, master):
        self.master = master
        master.title("Jerry v5 - Chatbot by KolenMG")
        master.configure(bg='black')

        # Placeholder image setup (if needed)
        placeholder_path = os.path.join(os.path.dirname(__file__), 'placeholder.gif')
        if os.path.exists(placeholder_path):
            self.placeholder_image = tk.PhotoImage(file=placeholder_path)
        else:
            self.placeholder_image = tk.PhotoImage(width=100, height=100)  # Placeholder if image is missing

        self.image_frame = tk.Frame(master, bg='black')
        self.image_frame.grid(row=0, column=0, padx=10, pady=10)

        self.image_label = tk.Label(self.image_frame, image=self.placeholder_image, bg='black')
        self.image_label.pack(padx=10, pady=10)

        self.intro_label = tk.Label(master, text="Try saying 'Hello'", bg='black', fg='#cccccc', font=('Arial', 12, 'italic'))
        self.intro_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')

        self.chat_history = scrolledtext.ScrolledText(master, width=60, height=20, bg='#222222', fg='white',
                                                      font=('Arial', 11), wrap=tk.WORD, relief=tk.FLAT, bd=0)
        self.chat_history.grid(row=2, column=0, padx=10, pady=(20, 10), sticky='w')

        self.user_input = tk.Entry(master, width=60, bg='#222222', fg='white', font=('Arial', 11), relief=tk.FLAT, bd=0)
        self.user_input.grid(row=3, column=0, padx=10, pady=10, sticky='w')

        self.send_button = tk.Button(master, text="Send", command=self.send_message, bg='#0084ff', fg='white',
                                     font=('Arial', 11, 'bold'), relief=tk.FLAT, bd=0, cursor='hand2')
        self.send_button.grid(row=3, column=0, padx=(470, 10), pady=10, sticky='e')

        # GIF handler initialization
        self.gif_handler = GIFHandler(self.image_label)
        self.gif_handler.load_idle_sequence()

        # Load model
        self.load_model()

        # TTS Toggle
        self.tts_active = False
        self.tts_toggle_button = tk.Button(master, text="Activate TTS", command=self.toggle_tts, bg='#0084ff', fg='white',
                                           font=('Arial', 11, 'bold'), relief=tk.FLAT, bd=0, cursor='hand2')
        self.tts_toggle_button.grid(row=4, column=0, padx=10, pady=10, sticky='w')

    def load_model(self):
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'HuggingFaceBotModel400M')
            self.tokenizer = BlenderbotTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = BlenderbotForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def send_message(self):
        user_input = self.user_input.get()
        self.user_input.delete(0, 'end')
        self.update_chat_history(f"\nYou: {user_input}", "user_tag")
        self.gif_handler.load_search_sequence()  # Trigger search sequence
        threading.Thread(target=self.generate_response, args=(user_input,)).start()

    def generate_response(self, user_input):
        try:
            inputs = self.tokenizer([user_input], return_tensors='pt')
            reply_ids = self.model.generate(**inputs)
            response = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
            self.update_chat_history(f"\nJerry: {response}", "generated_tag")
            self.gif_handler.load_idle_sequence()  # Return to idle sequence
            if self.tts_active:
                self.speak_response(response)
        except Exception as e:
            print(f"Error generating response: {e}")

    def speak_response(self, response):
        try:
            tts = gTTS(text=response, lang='en')
            audio_file = os.path.join(os.path.dirname(__file__), 'generated_response.mp3')
            tts.save(audio_file)
            os.system(f"start {audio_file}")
        except Exception as e:
            print(f"Error speaking response: {e}")

    def toggle_tts(self):
        if self.tts_active:
            self.tts_active = False
            self.tts_toggle_button.configure(text="Activate TTS", bg='#0084ff', fg='white')
        else:
            self.tts_active = True
            self.tts_toggle_button.configure(text="Deactivate TTS", bg='#ff4444', fg='white')

    def update_chat_history(self, text, tag):
        self.chat_history.configure(state='normal')
        self.chat_history.insert(tk.END, f"{text}\n", tag)
        self.chat_history.configure(state='disabled')
        self.chat_history.yview(tk.END)

def main():
    try:
        root = tk.Tk()
        app = ChatAppGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"An error occurred: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
