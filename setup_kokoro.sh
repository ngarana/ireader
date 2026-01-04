#!/bin/bash

# Setup script for Kokoro TTS - Smart Audiobook Reader

echo "🎤 Setting up Kokoro TTS for Smart Audiobook Reader..."

# Create models directory
mkdir -p ./models
cd models

# Download Kokoro model files
echo "📥 Downloading Kokoro TTS model files..."

if [ ! -f "kokoro-v1.0.onnx" ]; then
    echo "Downloading kokoro-v1.0.onnx (~300MB)..."
    wget -q --show-progress https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
else
    echo "✅ kokoro-v1.0.onnx already exists"
fi

if [ ! -f "voices-v1.0.bin" ]; then
    echo "Downloading voices-v1.0.bin..."
    wget -q --show-progress https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
else
    echo "✅ voices-v1.0.bin already exists"
fi

cd ..

# Install Python dependencies
echo "📦 Installing Python dependencies..."
uv sync

# Test Kokoro installation
echo "🧪 Testing Kokoro TTS installation..."
python3 -c "
from kokoro_onnx import Kokoro
import soundfile as sf
import os

kokoro = Kokoro('./models/kokoro-v1.0.onnx', './models/voices-v1.0.bin')
samples, sample_rate = kokoro.create('Hello, this is a test of Kokoro text to speech.', voice='af_sarah', speed=1.0, lang='en-us')
sf.write('test_output.wav', samples, sample_rate)
print('Test audio generated successfully!')
"

if [ -f "test_output.wav" ]; then
    echo "✅ Kokoro TTS setup successful! Test audio file created."
    echo "🎵 You can play the test file with: mpv test_output.wav"
    rm test_output.wav
else
    echo "❌ Kokoro TTS setup failed. Please check the installation."
    exit 1
fi

echo ""
echo "🎉 Kokoro TTS setup completed!"
echo ""
echo "📋 Available voices (examples):"
echo "   🇺🇸 Female: af_sarah, af_bella, af_nicole, af_sky"
echo "   🇺🇸 Male: am_adam, am_michael, am_echo"
echo "   🇬🇧 Female: bf_emma, bf_isabella"
echo "   🇬🇧 Male: bm_george, bm_lewis"
echo "   🇫🇷 Female: ff_siwis"
echo "   🇯🇵 Female: jf_alpha"
echo ""
echo "📋 Supported languages:"
echo "   en-us (American English)"
echo "   en-gb (British English)"
echo "   fr-fr (French)"
echo "   it (Italian)"
echo "   ja (Japanese)"
echo "   cmn (Mandarin Chinese)"
echo ""
echo "🚀 You can now run the Smart Audiobook Reader:"
echo "   python ireader.py your_document.pdf [output_directory]"
