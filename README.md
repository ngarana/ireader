# Smart Audiobook Reader

A Python application that converts PDF files into high-quality audiobooks using local AI models. Optimized for Intel Core Ultra 7 (Meteor Lake) processors with iGPU acceleration.

## Features

- 📚 **PDF Processing**: Extract text from PDF documents with intelligent chunking
- 🤖 **AI-Powered Text Processing**: Uses Ollama with local LLM models for text enhancement
- 🎤 **High-Quality TTS**: Piper neural text-to-speech for natural voice generation
- ⚡ **Intel Core Ultra 7 Optimization**: GPU acceleration and multi-core processing
- 🎵 **Audio Streaming**: Real-time playback with pygame
- 🔧 **Configurable**: Customizable voice models, processing parameters, and output settings

## System Requirements

- **Processor**: Intel Core Ultra 7 (Meteor Lake) recommended
- **RAM**: 8GB+ (16GB recommended for large documents)
- **Storage**: 2GB+ for models and temporary files
- **OS**: Linux (tested on Arch Linux)
- **Docker**: Required for Ollama container

## Installation

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Setup Ollama

```bash
# Start Ollama container (optimized for Intel iGPU)
./start-ollama.sh

# Pull a lightweight model (recommended for Meteor Lake)
docker exec -it ollama-intel ollama pull llama3.2:3b
```

### 3. Setup Piper TTS

```bash
# Run the setup script
./setup_piper.sh
```

## Usage

### Basic Usage

```bash
# Convert PDF to audiobook and play immediately
python ireader.py path/to/your/document.pdf

# Convert and save to specific directory
python ireader.py path/to/your/document.pdf ./my_audiobook
```

### Advanced Configuration

Edit `config.json` to customize settings:

```json
{
  "ollama_host": "http://localhost:11434",
  "ollama_model": "llama3.2:3b",
  "voice_model": "lessac/en_US/lessac-medium.onnx",
  "chunk_size": 1000,
  "max_concurrent_tts": 2,
  "gpu_acceleration": true,
  "speed": 1.0,
  "pitch": 1.0
}
```

## Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| `ollama_model` | Ollama model for text processing | `llama3.2:3b` |
| `voice_model` | Piper TTS voice model | `lessac-medium` |
| `chunk_size` | Text chunk size (characters) | `1000` |
| `max_concurrent_tts` | Concurrent TTS processes | `2` |
| `gpu_acceleration` | Use GPU acceleration | `true` |
| `speed` | Speech speed multiplier | `1.0` |
| `pitch` | Voice pitch adjustment | `1.0` |

## Intel Core Ultra 7 Optimization

The application is specifically optimized for Intel Core Ultra 7 processors:

- **iGPU Acceleration**: Utilizes Intel Arc Graphics for neural network processing
- **Multi-Core Processing**: Concurrent TTS generation optimized for P-cores and E-cores
- **Memory Management**: Intelligent chunking to balance RAM usage and performance
- **Thermal Optimization**: Balanced workload to prevent thermal throttling

## Voice Models

### Available Models

- **Lessac (Medium)**: Natural, clear voice - Default choice
- **Additional models**: Can be downloaded from [Piper Voices](https://huggingface.co/rhasspy/piper-voices)

### Adding New Voice Models

1. Download model files to `./models/` directory
2. Update `voice_model` in `config.json`
3. Restart the application

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```bash
   # Check if Ollama is running
   docker ps | grep ollama
   
   # Restart if needed
   ./start-ollama.sh
   ```

2. **Piper Not Found**
   ```bash
   # Reinstall Piper
   ./setup_piper.sh
   ```

3. **Audio Playback Issues**
   ```bash
   # Install audio system
   sudo pacman -S pulseaudio alsa-utils
   
   # Test audio
   speaker-test -c 2
   ```

4. **GPU Acceleration Not Working**
   - Ensure Intel GPU drivers are installed
   - Check `gpu_acceleration` is set to `true` in config
   - Verify Docker has GPU access

### Performance Tips

- Use `llama3.2:3b` for best performance on Meteor Lake
- Adjust `chunk_size` based on document complexity
- Increase `max_concurrent_tts` for multi-core optimization
- Use SSD storage for temporary files

## Development

### Project Structure

```
ireader/
├── ireader.py          # Main application
├── config.json         # Configuration file
├── setup_piper.sh      # Piper setup script
├── start-ollama.sh     # Ollama startup script
├── models/             # Voice model storage
└── pyproject.toml      # Project dependencies
```

### Adding Features

1. Modify `ireader.py` for core functionality
2. Update `config.json` for new settings
3. Test with various PDF documents
4. Ensure Intel Core Ultra 7 compatibility

## License

This project is open source. Please refer to the license file for details.

## Contributing

Contributions are welcome! Please ensure:
- Code follows Python best practices
- Intel Core Ultra 7 optimization is maintained
- Tests are added for new features
- Documentation is updated

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify system requirements
3. Test with minimal PDF files
4. Check logs for error details