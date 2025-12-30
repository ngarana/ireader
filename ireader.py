import fitz  # PyMuPDF
import ollama
import subprocess
import os
import sys

# --- CONFIGURATION ---
PDF_PATH = "pdf_file.pdf"      # Put your PDF file name here
MODEL_NAME = "llama3.2"     # Matches the model you pulled in Docker
VOICE_MODEL = "./models/voice.onnx"
OUTPUT_AUDIO = "output.wav"

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
    print("[*] Thinking (Cleaning text using Intel iGPU)...")
    
    # We process in chunks of 4000 characters to fit Llama 3.2 context
    chunk_size = 4000 
    cleaned_text = ""
    
    total_chunks = len(raw_text) // chunk_size + 1
    
    for i in range(0, len(raw_text), chunk_size):
        chunk = raw_text[i:i+chunk_size]
        prompt = f"""
        Refine the following text for a smooth audio listening experience.
        - Remove page numbers, headers, footers, and citations (like [12]).
        - Expand acronyms if they are obscure, but keep common ones.
        - Do not summarize; keep the full content but make it flow naturally.
        - Output ONLY the cleaned text. No intro/outro.
        
        TEXT:
        {chunk}
        """
        
        try:
            # This connects to your Docker container automatically via localhost:11434
            response = ollama.chat(model=MODEL_NAME, messages=[
                {'role': 'user', 'content': prompt},
            ])
            content = response['message']['content']
            cleaned_text += content + " "
            print(f"    - Processed chunk {i//chunk_size + 1}/{total_chunks}")
        except Exception as e:
            print(f"[!] Error connecting to Ollama: {e}")
            print("    (Make sure your Docker container is running!)")
            sys.exit(1)
        
    return cleaned_text

def generate_audio(text):
    print("[*] Speaking (Generating audio with Piper)...")
    # Using 'piper' directly since you installed it via AUR
    command = f'echo "{text}" | piper-tts --model {VOICE_MODEL} --output_file {OUTPUT_AUDIO}'
    subprocess.run(command, shell=True)
    print(f"[*] Done! Audio saved to: {os.path.abspath(OUTPUT_AUDIO)}")

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"Please place a PDF file named '{PDF_PATH}' in this folder.")
        sys.exit(1)

    # 1. Extract
    raw = extract_text(PDF_PATH)
    
    # 2. Clean (The AI part)
    clean = clean_text_with_ai(raw)
    
    # 3. Generate Audio
    generate_audio(clean)
    
    # 4. Play (Optional)
    os.system(f"mpv {OUTPUT_AUDIO}") # Assuming you have mpv or aplay
