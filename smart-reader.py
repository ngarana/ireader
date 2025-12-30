import fitz  # PyMuPDF
import ollama
import subprocess
import os
import sys

# --- TUNING CONFIGURATION ---
PDF_PATH = "poem.pdf"
MODEL_NAME = "llama3.2"
VOICE_MODEL = "./models/voice.onnx"
OUTPUT_AUDIO = "output.wav"

# Piper Command Name (Change if you installed differently)
PIPER_CMD = "piper-tts" # or "piper-tts" or "python -m piper"

# Voice Tuning
SPEAKING_SPEED = "1"      # 1.0 is normal. Higher = SLOWER. Lower = FASTER.
SENTENCE_PAUSE = "0.5"      # Seconds of silence between sentences.
NOISE_WIDTH    = "0.667"    # 0.3 to 1.0. Higher = more "expressive/unstable".

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
    print("[*] Speaking (Generating audio with Piper)...")
    
    # Constructing the command with tuning flags
    # Note: We pipe the text into piper using echo
    command = (
        f'echo "{text}" | {PIPER_CMD} '
        f'--model {VOICE_MODEL} '
        f'--output_file {OUTPUT_AUDIO} '
        f'--length_scale {SPEAKING_SPEED} '
        f'--sentence_silence {SENTENCE_PAUSE} '
        f'--noise_w {NOISE_WIDTH}'
    )
    
    try:
        # We run this in shell=True so the pipe (|) works
        subprocess.run(command, shell=True, check=True)
        print(f"[*] Done! Audio saved to: {os.path.abspath(OUTPUT_AUDIO)}")
    except subprocess.CalledProcessError:
        print(f"\n[!] CRITICAL ERROR: Piper failed to run.")
        print(f"    Command tried: {PIPER_CMD}")
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