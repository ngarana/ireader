#!/usr/bin/env python3
"""
Smart Audiobook Reader
Optimized for Intel Core Ultra 7 (Meteor Lake) with iGPU acceleration
Uses Ollama for text processing and Piper for TTS generation
"""

import os
import sys
import json
import time
import asyncio
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator
import logging

import fitz  # PyMuPDF
import ollama
import pygame
from pygame import mixer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SmartAudiobookReader:
    """
    Smart Audiobook Reader that converts PDF files to audio using
    Ollama for text processing and Piper for TTS generation.
    Optimized for Intel Core Ultra 7 with iGPU acceleration.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Smart Audiobook Reader."""
        self.config = self._load_config(config_path)
        self.ollama_client = ollama.Client(host=self.config.get('ollama_host', 'http://localhost:11434'))
        self.piper_path = self.config.get('piper_path', '/usr/bin/piper-tts')
        self.model_path = self.config.get('model_path', './models')
        self.temp_dir = tempfile.mkdtemp(prefix='ireader_')
        
        # Initialize pygame mixer for audio playback
        pygame.init()
        mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Optimization settings for Intel Core Ultra 7
        self.optimization_settings = {
            'chunk_size': self.config.get('chunk_size', 1000),  # characters per chunk
            'overlap': self.config.get('overlap', 100),  # characters overlap between chunks
            'max_concurrent_tts': self.config.get('max_concurrent_tts', 2),  # for Meteor Lake
            'gpu_acceleration': self.config.get('gpu_acceleration', True)
        }
        
        logger.info("Smart Audiobook Reader initialized")
        logger.info(f"Optimization settings: {self.optimization_settings}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'ollama_host': 'http://localhost:11434',
            'ollama_model': 'llama3.2:latest',
            'piper_path': 'piper',
            'model_path': './models',
            'voice_model': 'kristin/en_US/kristin-medium.onnx',
            'output_format': 'wav',
            'chunk_size': 1000,
            'overlap': 100,
            'max_concurrent_tts': 2,
            'gpu_acceleration': True,
            'sample_rate': 22050,
            'speed': 1.0,
            'pitch': 1.0
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def analyze_document_structure(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze PDF to detect document type (Poetry vs Prose), Title, and Structure.
        """
        logger.info(f"Analyzing document structure: {pdf_path}")
        structure = {
            'type': 'PROSE', # Default
            'title': 'Unknown',
            'is_poetry': False
        }
        
        try:
            doc = fitz.open(pdf_path)
            
            # Simple heuristic: Check text density of first few pages
            total_lines = 0
            short_lines = 0
            total_chars = 0
            
            # Check Title (largest font on Page 1)
            if len(doc) > 0:
                page = doc[0]
                blocks = page.get_text("dict")["blocks"]
                max_size = 0
                title_text = ""
                
                for b in blocks:
                    if "lines" in b:
                        for l in b["lines"]:
                            for s in l["spans"]:
                                if s["size"] > max_size:
                                    max_size = s["size"]
                                    title_text = s["text"]
                                    
                if title_text:
                    structure['title'] = title_text.strip()
            
            # Check Text Density to detect Poetry
            # Sample first 3 pages
            for i in range(min(3, len(doc))):
                page = doc[i]
                text = page.get_text()
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                
                total_lines += len(lines)
                for line in lines:
                    total_chars += len(line)
                    # Poetry strings tend to be shorter than full paragraph lines (e.g. < 60 chars)
                    if len(line) < 60:
                        short_lines += 1
            
            doc.close()
            
            avg_line_length = (total_chars / total_lines) if total_lines > 0 else 0
            short_line_ratio = (short_lines / total_lines) if total_lines > 0 else 0
            
            logger.info(f"Doc Analysis: Avg Line Len={avg_line_length:.1f}, Short Line Ratio={short_line_ratio:.2f}")
            
            # Heuristic for Poetry:
            # High ratio of "short" lines but not "tiny" (avoid TOC/Index)
            # And average line length is moderate
            if short_line_ratio > 0.6 and avg_line_length < 60:
                structure['type'] = 'POETRY'
                structure['is_poetry'] = True
                
            logger.info(f"Detected Document Structure: {structure}")
            return structure
            
        except Exception as e:
            logger.warning(f"Structure analysis failed: {e}")
            return structure

    def extract_text_from_pdf(self, pdf_path: str, is_poetry: bool = False) -> List[str]:
        """
        Extract text from PDF file and split into optimized chunks.
        Respects 'is_poetry' flag to preserve line breaks.
        """
        logger.info(f"Extracting text from PDF (Poetry Mode: {is_poetry})")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    full_text += text + "\n"
            
            doc.close()
            
            # Clean and split text into chunks
            chunks = self._split_text_into_chunks(full_text, is_poetry)
            logger.info(f"Extracted {len(chunks)} text chunks from PDF")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def _split_text_into_chunks(self, text: str, is_poetry: bool = False) -> List[str]:
        """
        Split text into optimized chunks.
        If is_poetry is True, preserves newlines as meaningful pauses.
        If is_poetry is False, normalizes newlines to spaces (except paragraphs).
        """
        chunks = []
        chunk_size = self.optimization_settings['chunk_size']
        
        paragraphs = []
        
        if is_poetry:
            # For poetry, treat blank lines as stanza breaks (paragraphs)
            # We want to keep internal newlines for the LLM to see the structure
            raw_paras = text.replace('\r\n', '\n').split('\n\n')
            for p in raw_paras:
                if p.strip():
                     paragraphs.append(p.strip())
        else:
            # For prose, normalize single newlines to spaces
            raw_paras = text.replace('\r\n', '\n').split('\n\n')
            for p in raw_paras:
                clean_p = p.replace('\n', ' ').strip()
                if clean_p:
                    paragraphs.append(clean_p)
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Separator: Double newline for prose (para break), 
            # or Double newline for poetry (stanza break)
            sep = "\n\n"
            
            if len(current_chunk) + len(paragraph) + 2 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Handle huge paragraphs
                if len(paragraph) > chunk_size:
                    # Logic similar to before but simpler
                    chunks.append(paragraph[:chunk_size]) # Naive split for rare huge case
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += sep + paragraph
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

    def generate_speech_with_piper(self, text: str, output_path: str) -> bool:
        """
        Generate speech from text using Piper TTS.
        """
        try:
            cmd = [
                self.piper_path,
                '--model', f"{self.model_path}/{self.config['voice_model']}",
                '--output_file', output_path,
                '--sample_rate', str(self.config['sample_rate']),
                '--speed', str(self.config['speed']),
                '--pitch', str(self.config['pitch'])
            ]
            
            if self.optimization_settings['gpu_acceleration']:
                cmd.extend(['--use_gpu'])
            
            process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode == 0:
                logger.debug(f"Generated speech: {output_path}")
                return True
            else:
                logger.error(f"Piper error: {stderr}")
                return False
                
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
        text_chunks = self.extract_text_from_pdf(pdf_path, is_poetry=structure['is_poetry'])
        
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
                        success = await asyncio.to_thread(self.generate_speech_with_piper, text, output_path)
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