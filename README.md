# Smart Audiobook Reader

A Python application that converts PDF files into high-quality audiobooks using local AI models. Optimized for Intel Core Ultra 7 (Meteor Lake) processors.

## Features

- 📚 **PDF Processing**: Extract text from PDF documents with intelligent chunking
- 🤖 **AI-Powered Text Processing**: Uses Ollama with local LLM models for text enhancement
- 🎤 **High-Quality TTS**: Kokoro neural text-to-speech for natural voice generation
- ⚡ **Intel Core Ultra 7 Optimization**: Multi-core processing optimized for Meteor Lake
- 🎵 **Audio Streaming**: Real-time playback with pygame
- 🌍 **Multi-Language**: Supports English (US/UK), French, Italian, Japanese, and Mandarin Chinese
- 🔧 **Configurable**: Customizable voice models, processing parameters, and output settings

## System Requirements

- **Python**: 3.10 - 3.12 (Python 3.13+ not currently supported by Kokoro)
- **Processor**: Intel Core Ultra 7 (Meteor Lake) recommended
- **RAM**: 8GB+ (16GB recommended for large documents)
- **Storage**: 500MB+ for models and temporary files
- **OS**: Linux (tested on Arch Linux)
- **Docker**: Required for Ollama container

## Installation

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Setup Ollama

```bash
# Start Ollama container (optimized for Intel iGPU)
./start-ollama.sh

# Pull a lightweight model (recommended for Meteor Lake)
docker exec -it ollama-intel ollama pull llama3.2:3b
```

### 3. Setup Kokoro TTS

```bash
# Run the setup script (downloads ~300MB model)
./setup_kokoro.sh
```

## Usage

### Basic Usage

```bash
# Convert PDF to audiobook and play immediately
python ireader.py path/to/your/document.pdf

# Convert and save to specific directory
python ireader.py path/to/your/document.pdf ./my_audiobook
```

### Simple Reader (smart-reader.py)

```bash
# Quick conversion with default settings
python smart-reader.py
```

### Advanced Configuration

Edit `config.json` to customize settings:

```json
{
  "ollama_host": "http://localhost:11434",
  "ollama_model": "llama3.1:8b",
  "model_path": "./models",
  "kokoro_model": "kokoro-v1.0.onnx",
  "kokoro_voices": "voices-v1.0.bin",
  "voice": "af_sarah",
  "lang": "en-us",
  "chunk_size": 1000,
  "max_concurrent_tts": 2,
  "speed": 1.0
}
```

## Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| `ollama_model` | Ollama model for text processing | `llama3.1:8b` |
| `voice` | Kokoro TTS voice | `af_sarah` |
| `lang` | Language code | `en-us` |
| `chunk_size` | Text chunk size (characters) | `1000` |
| `max_concurrent_tts` | Concurrent TTS processes | `2` |
| `speed` | Speech speed multiplier | `1.0` |

## Available Voices

### American English (en-us)
- **Female**: af_alloy, af_aoede, af_bella, af_heart, af_jessica, af_kore, af_nicole, af_nova, af_river, af_sarah, af_sky
- **Male**: am_adam, am_echo, am_eric, am_fenrir, am_liam, am_michael, am_onyx, am_puck

### British English (en-gb)
- **Female**: bf_alice, bf_emma, bf_isabella, bf_lily
- **Male**: bm_daniel, bm_fable, bm_george, bm_lewis

### Other Languages
- **French (fr-fr)**: ff_siwis
- **Italian (it)**: if_sara, im_nicola
- **Japanese (ja)**: jf_alpha, jf_gongitsune, jf_nezumi, jf_tebukuro, jm_kumo
- **Mandarin (cmn)**: zf_xiaobei, zf_xiaoni, zf_xiaoxiao, zf_xiaoyi, zm_yunjian, zm_yunxi, zm_yunxia, zm_yunyang

## Intel Core Ultra 7 Optimization

The application is specifically optimized for Intel Core Ultra 7 processors:

- **Multi-Core Processing**: Concurrent TTS generation optimized for P-cores and E-cores
- **Memory Management**: Intelligent chunking to balance RAM usage and performance
- **Thermal Optimization**: Balanced workload to prevent thermal throttling

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```bash
   # Check if Ollama is running
   docker ps | grep ollama
   
   # Restart if needed
   ./start-ollama.sh
   ```

2. **Kokoro Model Not Found**
   ```bash
   # Re-run setup script
   ./setup_kokoro.sh
   ```

3. **Audio Playback Issues**
   ```bash
   # Install audio system
   sudo pacman -S pulseaudio alsa-utils
   
   # Test audio
   speaker-test -c 2
   ```

4. **Python Version Issues**
   - Kokoro TTS requires Python 3.10-3.12
   - Use `uv` to manage Python versions: `uv python install 3.12`

### Performance Tips

- Use `llama3.2:3b` for best performance on Meteor Lake
- Adjust `chunk_size` based on document complexity
- Increase `max_concurrent_tts` for multi-core optimization
- Use SSD storage for temporary files

## Project Structure

```
ireader/
├── ireader.py          # Main application (full-featured)
├── smart-reader.py     # Simple reader script
├── config.json         # Configuration file
├── setup_kokoro.sh     # Kokoro TTS setup script
├── start-ollama.sh     # Ollama startup script
├── models/             # Voice model storage
│   ├── kokoro-v1.0.onnx
│   └── voices-v1.0.bin
└── pyproject.toml      # Project dependencies
```

## License

This project is open source. Please refer to the license file for details.

## Acknowledgments

- [Kokoro TTS](https://github.com/thewh1teagle/kokoro-onnx) - Neural TTS engine
- [Ollama](https://ollama.ai/) - Local LLM inference
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF processing
