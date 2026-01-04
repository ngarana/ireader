import fitz  # PyMuPDF
import ollama
import os
import sys
import soundfile as sf
from kokoro_onnx import Kokoro

# --- TUNING CONFIGURATION ---
PDF_PATH = "poem.pdf"
MODEL_NAME = "llama3.2"
MODEL_PATH = "./models"
KOKORO_MODEL = "kokoro-v1.0.onnx"
KOKORO_VOICES = "voices-v1.0.bin"
OUTPUT_AUDIO = "output.wav"

# Voice Tuning
VOICE = "af_sarah"  # See available voices at https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
LANG = "en-us"      # en-us, en-gb, fr-fr, it, ja, cmn
SPEAKING_SPEED = 1.0  # 1.0 is normal. Higher = faster, Lower = slower

def extract_text(pdf_path):
    print(f"[*] Reading {pdf_path}...")
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"[!] Error reading PDF: {e}")
        sys.exit(1)

def clean_text_with_ai(raw_text):
    print("[*] Thinking (Refining text for speech flow)...")
    
    chunk_size = 6000 
    cleaned_text = ""
    total_chunks = len(raw_text) // chunk_size + 1
    
    for i in range(0, len(raw_text), chunk_size):
        chunk = raw_text[i:i+chunk_size]
        
        # --- TUNED PROMPT FOR AUDIO ---
        prompt = f"""
        Rewrite the text below for a professional audiobook narrator.
        
        Rules:
        1. PUNCTUATION: Insert commas significantly more often than in written text to indicate breathing pauses.
        2. FLOW: Break long, complex sentences into two shorter sentences.
        3. NUMBERS: Convert "Fig 1.2" to "Figure one point two".
        4. TONE: Natural, engaging, and clear.
        5. CLEAN: Remove citations like [1], URLs, and page numbers.
        
        Text to process:
        {chunk}
        """
        
        try:
            response = ollama.chat(
                model=MODEL_NAME, 
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'num_ctx': 8192, 
                }
            )
            content = response['message']['content']
            cleaned_text += content + " "
            print(f"    - Processed chunk {i//chunk_size + 1}/{total_chunks}")
        except Exception as e:
            print(f"[!] Error connecting to Ollama: {e}")
            sys.exit(1)
        
    return cleaned_text

def generate_audio(text):
    print("[*] Speaking (Generating audio with Kokoro TTS)...")
    
    try:
        # Initialize Kokoro TTS
        kokoro_model_path = os.path.join(MODEL_PATH, KOKORO_MODEL)
        kokoro_voices_path = os.path.join(MODEL_PATH, KOKORO_VOICES)
        kokoro = Kokoro(kokoro_model_path, kokoro_voices_path)
        
        # Generate speech
        samples, sample_rate = kokoro.create(
            text,
            voice=VOICE,
            speed=SPEAKING_SPEED,
            lang=LANG
        )
        
        # Save to file
        sf.write(OUTPUT_AUDIO, samples, sample_rate)
        print(f"[*] Done! Audio saved to: {os.path.abspath(OUTPUT_AUDIO)}")
    except Exception as e:
        print("\n[!] CRITICAL ERROR: Kokoro TTS failed to run.")
        print(f"    Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"Please place a PDF file named '{PDF_PATH}' in this folder.")
        sys.exit(1)

    # 1. Extract
    raw = extract_text(PDF_PATH)
    
    # 2. Clean
    clean = clean_text_with_ai(raw)
    
    # 3. Generate
    generate_audio(clean)
    
    # 4. Play
    if os.path.exists("/usr/bin/mpv"):
        os.system(f"mpv {OUTPUT_AUDIO}")