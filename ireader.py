#!/usr/bin/env python3
"""
Smart Audiobook Reader
Optimized for Intel Core Ultra 7 (Meteor Lake)
Uses Ollama for text processing and Kokoro TTS for speech generation
"""

import os
import sys
import json
import time
import asyncio
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

import fitz  # PyMuPDF
import ollama
import pygame
from pygame import mixer
import soundfile as sf
from kokoro_onnx import Kokoro

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SmartAudiobookReader:
    """
    Smart Audiobook Reader that converts PDF files to audio using
    Ollama for text processing and Kokoro TTS for speech generation.
    Optimized for Intel Core Ultra 7.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Smart Audiobook Reader."""
        self.config = self._load_config(config_path)
        self.ollama_client = ollama.Client(host=self.config.get('ollama_host', 'http://localhost:11434'))
        self.model_path = self.config.get('model_path', './models')
        self.temp_dir = tempfile.mkdtemp(prefix='ireader_')
        
        # Initialize Kokoro TTS
        kokoro_model = os.path.join(self.model_path, self.config.get('kokoro_model', 'kokoro-v1.0.onnx'))
        kokoro_voices = os.path.join(self.model_path, self.config.get('kokoro_voices', 'voices-v1.0.bin'))
        self.kokoro = Kokoro(kokoro_model, kokoro_voices)
        
        # Initialize pygame mixer for audio playback
        pygame.init()
        mixer.init(frequency=24000, size=-16, channels=1, buffer=512)
        
        # Optimization settings for Intel Core Ultra 7
        self.optimization_settings = {
            'chunk_size': self.config.get('chunk_size', 1000),  # characters per chunk
            'overlap': self.config.get('overlap', 100),  # characters overlap between chunks
            'max_concurrent_tts': self.config.get('max_concurrent_tts', 2),  # for Meteor Lake
        }
        
        logger.info("Smart Audiobook Reader initialized with Kokoro TTS")
        logger.info(f"Optimization settings: {self.optimization_settings}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'ollama_host': 'http://localhost:11434',
            'ollama_model': 'llama3.1:8b', 
            'model_path': './models',
            'kokoro_model': 'kokoro-v1.0.onnx',
            'kokoro_voices': 'voices-v1.0.bin',
            'voice': 'af_sarah',
            'lang': 'en-us',
            'output_format': 'wav',
            'chunk_size': 1000,
            'overlap': 100,
            'max_concurrent_tts': 2,
            'sample_rate': 24000,
            'speed': 1.0
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def analyze_document_structure(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze PDF to detect document type, Title, Author, and Metadata using visual hierarchy.
        """
        logger.info(f"Analyzing document structure: {pdf_path}")
        structure = {
            'type': 'PROSE',
            'title': 'Unknown',
            'author': None,
            'metadata': [],
            'is_poetry': False
        }
        
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                return structure

            page = doc[0]
            blocks = page.get_text("dict")["blocks"]
            
            # 1. Parsing Blocks with Font Info
            text_blocks = [] # List of {'text': str, 'size': float, 'y': float}
            
            for b in blocks:
                if "lines" in b:
                    block_text = ""
                    max_size = 0.0
                    y_pos = b["bbox"][1] # Top Y coordinate
                    
                    for l in b["lines"]:
                        for s in l["spans"]:
                            if s["text"].strip():
                                block_text += s["text"] + " "
                                if s["size"] > max_size:
                                    max_size = s["size"]
                    
                    if block_text.strip():
                        text_blocks.append({
                            'text': block_text.strip(),
                            'size': max_size,
                            'y': y_pos
                        })
            
            # Sort by Y position (top to bottom)
            text_blocks.sort(key=lambda x: x['y'])
            
            if not text_blocks:
                return structure

            # 2. Identify Role based on Font Size
            # Find the largest font (Title)
            sizes = [b['size'] for b in text_blocks]
            max_font_size = max(sizes) if sizes else 0
            
            # Find the "body" font size (mode)
            # Round sizes to nearest 0.5 to group similar fonts
            from collections import Counter
            rounded_sizes = [round(s * 2) / 2 for s in sizes]
            common_size = Counter(rounded_sizes).most_common(1)[0][0]
            
            logger.info(f"Max Font: {max_font_size}, Body Font ~{common_size}")
            
            # Assign roles
            body_start_index = 0
            
            for i, block in enumerate(text_blocks):
                # Heuristic: If we hit the body font size, and we are past the top few blocks, 
                # assume body starts.
                # Optimization: For poetry, lines can be short, so relying on len > 50 is risky.
                # If the font size matches the common body size tightly, assume it is body.
                
                # Title is usually noticeably larger (e.g. +4pt)
                is_title_candidate = block['size'] > common_size + 2
                
                if not is_title_candidate and abs(block['size'] - common_size) < 1.0:
                    # If we found a block that matches body font size and isn't the main title
                    # It's likely the start of body, especially if we have already seen a title
                    if structure['title'] != 'Unknown':
                         body_start_index = i
                         break
                    # Fallback: if we haven't seen a title yet, but this is the common font,
                    # and it's not the very first block, it might be body.
                    elif i > 0:
                        body_start_index = i
                        break
            
            # Everything before body_start is potentially Title/Author/Metadata
            header_blocks = text_blocks[:body_start_index] if body_start_index > 0 else text_blocks[:1]
            
            # Extract specific fields from header blocks
            for block in header_blocks:
                # If it's the max size (or very close), it's the Title
                if block['size'] >= max_font_size - 1:
                    structure['title'] = block['text']
                # If it's smaller than title but larger than body (or distinct), likely Author/Meta
                elif block['size'] < max_font_size:
                    if not structure['author']:
                        # First non-title header is often Author
                        structure['author'] = block['text']
                    else:
                        structure['metadata'].append(block['text'])

            # 3. Detect Document Type (Visual Density Analysis on Body)
            # Re-scan typical body pages
            total_lines = 0
            short_lines = 0
            total_chars = 0
            
            for i in range(min(3, len(doc))):
                p = doc[i]
                text = p.get_text()
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                total_lines += len(lines)
                for line in lines:
                    total_chars += len(line)
                    if len(line) < 60:
                        short_lines += 1
            
            doc.close()
            
            avg_line_length = (total_chars / total_lines) if total_lines > 0 else 0
            short_line_ratio = (short_lines / total_lines) if total_lines > 0 else 0
            
            if short_line_ratio > 0.6 and avg_line_length < 60:
                structure['type'] = 'POETRY'
                structure['is_poetry'] = True
            
            structure['header_end_y'] = text_blocks[body_start_index-1]['y'] + 10 if body_start_index > 0 else 0
            
            logger.info(f"Detailed Structure: {structure}")
            return structure
            
        except Exception as e:
            logger.warning(f"Structure analysis failed: {e}")
            return structure

    def extract_text_from_pdf(self, pdf_path: str, structure: Dict[str, Any]) -> List[str]:
        """
        Extract text, separating Introduction (Metadata) from Body.
        """
        logger.info(f"Extracting text (Poetry Mode: {structure['is_poetry']})")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # 1. Construct Intro Chunk
        intro_text = f"Title: {structure['title']}. "
        if structure['author']:
            intro_text += f"By {structure['author']}. "
        if structure['metadata']:
            intro_text += " ".join(structure['metadata'])
        
        chunks = [intro_text]
        
        try:
            doc = fitz.open(pdf_path)
            full_body_text = ""
            
            # 2. Extract Body Text (Skipping Header on Page 1)
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                if page_num == 0 and structure.get('header_end_y', 0) > 0:
                    # Clip extraction to start BELOW the header
                    # rect = fitz.Rect(0, header_end_y, page.rect.width, page.rect.height)
                    # Simple approach: Get all text, then fuzzy match removal or just rely on blocks
                    # Better: Iterate blocks again and filter by Y
                    blocks = page.get_text("blocks")
                    for b in blocks:
                        # b is (x0, y0, x1, y1, text, block_no, block_type)
                        if b[1] > structure['header_end_y']:
                            full_body_text += b[4] + "\n"
                else:
                    text = page.get_text()
                    if text.strip():
                        full_body_text += text + "\n"
            
            doc.close()
            
            # 3. Chunk Body Text
            body_chunks = self._split_text_into_chunks(full_body_text, structure['is_poetry'])
            chunks.extend(body_chunks)
            
            logger.info(f"Extracted {len(chunks)} chunks (1 intro + {len(body_chunks)} body)")
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            raise
    
    def _split_text_into_chunks(self, text: str, is_poetry: bool = False) -> List[str]:
        """
        Split text into optimized chunks.
        Ensures strict semantic boundaries or sufficient overlap to prevent data loss.
        """
        chunks = []
        chunk_size = self.optimization_settings['chunk_size']
        # We need substantial overlap if we break mid-flow
        overlap_size = self.optimization_settings.get('overlap', 150)
        
        paragraphs = []
        
        if is_poetry:
            raw_paras = text.replace('\r\n', '\n').split('\n\n')
            for p in raw_paras:
                if p.strip():
                     paragraphs.append(p.strip())
        else:
            raw_paras = text.replace('\r\n', '\n').split('\n\n')
            for p in raw_paras:
                clean_p = p.replace('\n', ' ').strip()
                if clean_p:
                    paragraphs.append(clean_p)
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            sep = "\n\n"
            
            # If adding this paragraph fits, add it
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += sep + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Current chunk is full-ish. 
                if current_chunk:
                    chunks.append(current_chunk)
                    # Start new chunk. 
                    # If the PREVIOUS chunk ended cleanly on a paragraph, we ideally don't need context repetition
                    # for the AudioReader (because it causes double-reading).
                    # HOWEVER, for the LLM 'cleaning', it needs context.
                    # The Context-Aware Prompting handles the "understanding". 
                    # We should avoid putting overlapping text into the *output* unless we strip it later.
                    # Since we don't have intelligent deduplication post-LLM, we should try to break cleanly.
                    current_chunk = ""
                
                # If the paragraph itself is huge, we MUST split it with overlap
                if len(paragraph) > chunk_size:
                    # Split this specific paragraph
                    sub_start = 0
                    while sub_start < len(paragraph):
                        sub_end = min(sub_start + chunk_size, len(paragraph))
                        
                        # Find a sentence boundary to avoid cutting mid-word if possible
                        if sub_end < len(paragraph):
                            # Look for sentence end in the last 20% of the chunk
                            lookback = int(chunk_size * 0.2)
                            search_area = paragraph[sub_end-lookback:sub_end]
                            found_split = -1
                            for punct in ['.', '!', '?']:
                                idx = search_area.rfind(punct)
                                if idx != -1:
                                    found_split = (sub_end - lookback) + idx + 1
                                    break
                            
                            if found_split != -1:
                                sub_end = found_split
                        
                        sub_chunk = paragraph[sub_start:sub_end].strip()
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        
                        # Prepare next sub-chunk
                        if sub_end < len(paragraph):
                            # No overlap logic here for *audio* generation unless we want repetition.
                            # Standard TTS splitting usually just cuts.
                            # But to prevent LLM from deleting "broken" starts, we can just proceed.
                            sub_start = sub_end 
                        else:
                            break
                else:
                     current_chunk = paragraph
                    
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    async def process_text_runner(self, text_chunks: List[str], queue: asyncio.Queue, structure: Dict[str, Any]):
        """
        Producer task: Processes text with Ollama and puts results in a queue.
        Adapts prompt based on Document Structure (Poetry vs Prose).
        """
        logger.info("Starting text processing task...")
        previous_context = ""
        
        doc_type = structure.get('type', 'PROSE')
        title = structure.get('title', 'Unknown')
        
        # Base System Prompt
        base_prompt = (
            "You are a specialized text cleaner for specific Text-to-Speech processing. "
            "Your task is to fix OCR errors and normalize punctuation for natural selection. "
            f"DOCUMENT CONTEXT: Title='{title}', Type='{doc_type}'. "
        )
        
        # Adaptive Constraints
        if doc_type == 'POETRY':
            specific_rules = (
                "RULES for POETRY:\n"
                "1. PRESERVE LINE BREAKS. Do not merge lines into paragraphs.\n"
                "2. Keep the original rhythm and stanza structure.\n"
                "3. Fix typos but DO NOT change words to 'smooth' the meter.\n"
                "4. Output ONLY the cleaned text.\n"
            )
        else:
            specific_rules = (
                "RULES for PROSE:\n"
                "1. Merge broken lines into complete paragraphs (fix hard wraps).\n"
                "2. Ensure sentences end with proper punctuation (. ? !).\n"
                "3. Output ONLY the cleaned text.\n"
            )
            
        system_prompt = base_prompt + specific_rules + (
            "GENERAL RULES:\n"
            "- NO introductions, NO explanations, NO fillers.\n"
            "- If provided, use [PREVIOUS CONTEXT] for continuity but do not repeat it."
        )

        for i, chunk in enumerate(text_chunks):
            try:
                msg_content = f"Clean the following text:\n\n{chunk}"
                if previous_context:
                    msg_content = f"[PREVIOUS CONTEXT]: ...{previous_context}\n\n" + msg_content

                response = await asyncio.to_thread(
                    self.ollama_client.chat,
                    model=self.config['ollama_model'],
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': msg_content}
                    ]
                )
                
                processed_text = response['message']['content'].strip()
                
                # Aggressive Post-Processing: Strip common bot introductions
                # Even with strict prompts, smaller models sometimes can't help themselves.
                import re
                bot_intros = [
                    r"^Here is the cleaned text:?",
                    r"^Here is the text cleaned:?",
                    r"^Here is the text with .*?:?",
                    r"^Cleaned text:?",
                    r"^Output:?",
                    r"^Sure, here is.*?:",
                    r"^I have cleaned.*?:",
                    # Suffix patterns (multiline needed)
                    r"I applied the following rules.*",
                    r"Changes made:.*",
                    r"Note:.*",
                    r"Explanation:.*",
                    r"I have removed.*",
                ]
                
                for pattern in bot_intros:
                    processed_text = re.sub(pattern, "", processed_text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL).strip()

                # Heuristic validation
                if len(processed_text) < len(chunk) * 0.5 or len(processed_text) > len(chunk) * 1.5:
                    logger.warning(f"Chunk {i+1} processed text length suspicious. Using original.")
                    processed_text = chunk
                
                # Update context
                previous_context = processed_text[-300:] if len(processed_text) > 300 else processed_text
                previous_context = previous_context.replace('\n', ' ')

                logger.info(f"Processed chunk {i+1}/{len(text_chunks)}")
                await queue.put((i, processed_text))
                
            except Exception as e:
                logger.warning(f"Error processing chunk {i+1} with Ollama: {e}")
                await queue.put((i, chunk)) # Fallback
        
        # Signal completion
        await queue.put(None)

    def generate_speech_with_kokoro(self, text: str, output_path: str) -> bool:
        """
        Generate speech from text using Kokoro TTS.
        """
        try:
            samples, sample_rate = self.kokoro.create(
                text,
                voice=self.config.get('voice', 'af_sarah'),
                speed=self.config.get('speed', 1.0),
                lang=self.config.get('lang', 'en-us')
            )
            
            sf.write(output_path, samples, sample_rate)
            logger.debug(f"Generated speech: {output_path}")
            return True
                
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return False

    async def generate_audiobook(self, pdf_path: str, output_dir: str) -> List[str]:
        """
        Generate complete audiobook from PDF using a Producer-Consumer pipeline.
        Includes Document Structure Analysis.
        """
        logger.info(f"Starting audiobook generation for: {pdf_path}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Analyze Document Structure
        structure = self.analyze_document_structure(pdf_path)
        
        # 2. Extract text (aware of poetry mode)
        text_chunks = self.extract_text_from_pdf(pdf_path, structure=structure)
        
        # 3. Start Pipeline
        chunk_queue = asyncio.Queue()
        producer_task = asyncio.create_task(self.process_text_runner(text_chunks, chunk_queue, structure))
        
        audio_files_map = {}
        semaphore = asyncio.Semaphore(self.optimization_settings['max_concurrent_tts'])
        
        async def tts_worker():
            while True:
                item = await chunk_queue.get()
                if item is None:
                    chunk_queue.task_done()
                    return 
                
                index, text = item
                try:
                    output_path = os.path.join(output_dir, f"chunk_{index:04d}.{self.config['output_format']}")
                    async with semaphore:
                        success = await asyncio.to_thread(self.generate_speech_with_kokoro, text, output_path)
                        if success:
                            audio_files_map[index] = output_path
                except Exception as e:
                    logger.error(f"Error in TTS worker: {e}")
                finally:
                    chunk_queue.task_done()
        
        consumer_tasks = [asyncio.create_task(tts_worker()) for _ in range(self.optimization_settings['max_concurrent_tts'])]
        
        await producer_task
        for _ in range(len(consumer_tasks) - 1): 
            await chunk_queue.put(None)
            
        await asyncio.gather(*consumer_tasks)
        
        audio_files = []
        for i in range(len(text_chunks)):
            if i in audio_files_map:
                audio_files.append(audio_files_map[i])
        
        logger.info(f"Generated {len(audio_files)} audio files")
        return audio_files
    
    def play_audiobook(self, audio_files: List[str]) -> None:
        """
        Play the generated audiobook.
        
        Args:
            audio_files: List of audio file paths to play
        """
        logger.info("Starting audiobook playback...")
        
        for i, audio_file in enumerate(audio_files):
            try:
                logger.info(f"Playing: {os.path.basename(audio_file)} ({i+1}/{len(audio_files)})")
                
                sound = mixer.Sound(audio_file)
                sound.play()
                
                # Wait for audio to finish playing
                while mixer.get_busy():
                    time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error playing {audio_file}: {e}")
                continue
        
        logger.info("Audiobook playback completed")
    
    def cleanup(self):
        """Clean up temporary files and resources."""
        logger.info("Cleaning up resources...")
        
        # Stop any playing audio
        mixer.stop()
        
        # Remove temporary directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Quit pygame
        pygame.quit()

async def main():
    """Main function to run the Smart Audiobook Reader."""
    if len(sys.argv) < 2:
        print("Usage: python ireader.py <pdf_file> [output_directory]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./audiobook_output"
    
    try:
        reader = SmartAudiobookReader()
        
        # Generate audiobook
        audio_files = await reader.generate_audiobook(pdf_path, output_dir)
        
        if audio_files:
            print(f"\n✅ Audiobook generated successfully!")
            print(f"📁 Output directory: {output_dir}")
            print(f"🎵 Audio files: {len(audio_files)}")
            print("\n🎧 Starting playback in 3 seconds...")
            time.sleep(3)
            
            # Play the audiobook
            reader.play_audiobook(audio_files)
        else:
            print("❌ Failed to generate audiobook")
    
    except KeyboardInterrupt:
        print("\n⏹️  Audiobook generation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Main execution error")
    finally:
        if 'reader' in locals():
            reader.cleanup()

if __name__ == "__main__":
    asyncio.run(main())