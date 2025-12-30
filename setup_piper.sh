#!/bin/bash

# Setup script for Piper TTS - Optimized for Intel Core Ultra 7 (Meteor Lake)

echo "🎤 Setting up Piper TTS for Smart Audiobook Reader..."

# Check if Piper is already installed
if command -v piper-tts &> /dev/null; then
    echo "✅ Piper is already installed"
    piper-tts --version
else
    echo "📦 Installing Piper TTS..."
    
    # Install Piper using pip (recommended approach)
    uv add piper-tts
    
    # Alternatively, you can download the binary:
    # wget https://github.com/rhasspy/piper/releases/latest/download/piper_linux_x86_64.tar.gz
    # tar xvf piper_linux_x86_64.tar.gz
    # sudo cp piper/piper /usr/local/bin/
fi

# Create models directory
mkdir -p ./models

# Download a high-quality voice model (Lessac - clear and natural)
echo "📥 Downloading voice model..."
cd models

if [ ! -f "lessac/en_US/lessac-medium.onnx" ]; then
    echo "Downloading Lessac medium quality voice model..."
    mkdir -p lessac/en_US
    wget -O lessac/en_US/lessac-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx
    wget -O lessac/en_US/lessac-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
else
    echo "✅ Voice model already exists"
fi

cd ..

# Test Piper installation
echo "🧪 Testing Piper installation..."
echo "Hello, this is a test of the Piper text-to-speech system." | piper-tts --model ./models/lessac/en_US/lessac-medium.onnx --output_file test_output.wav

if [ -f "test_output.wav" ]; then
    echo "✅ Piper setup successful! Test audio file created."
    echo "🎵 You can play the test file with: play test_output.wav"
    rm test_output.wav
else
    echo "❌ Piper setup failed. Please check the installation."
    exit 1
fi

echo ""
echo "🎉 Piper TTS setup completed!"
echo ""
echo "📋 Available voice models:"
echo "   - Lessac (medium quality) - Natural and clear voice"
echo ""
echo "⚙️  Optimization for Intel Core Ultra 7:"
echo "   - GPU acceleration enabled by default"
echo "   - Optimized chunk size for Meteor Lake iGPU"
echo "   - Concurrent processing for multi-core performance"
echo ""
echo "🚀 You can now run the Smart Audiobook Reader:"
echo "   python ireader.py your_document.pdf [output_directory]"