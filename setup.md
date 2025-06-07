# 🔧 Setup Guide - Offline Audio Translator

Complete installation and setup instructions for the Offline Audio Translator application.

## 📋 System Requirements

### Minimum Requirements
- **Operating System**: Windows 10, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 3GB free space (for models and dependencies)
- **Internet**: Required for initial setup and TTS (optional for core functionality)
- **Microphone**: Any working microphone or headset

### Recommended Specifications
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB+ for smooth performance
- **GPU**: CUDA-compatible GPU (optional, for faster processing)
- **Storage**: SSD for faster model loading
- **Audio**: External microphone for better audio quality

## 🐍 Python Installation

### Windows
1. Download Python from [python.org](https://www.python.org/downloads/)
2. **Important**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

### macOS
```bash
# Using Homebrew (recommended)
brew install python

# Or download from python.org
# Verify installation
python3 --version
pip3 --version
```

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Verify installation
python3 --version
pip3 --version
```

## 📦 Project Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/offline-audio-translator.git
cd offline-audio-translator
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

#### Core Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Manual Installation (if requirements.txt fails)
```bash
# Core packages
pip install streamlit
pip install openai-whisper
pip install torch torchvision torchaudio
pip install transformers
pip install sentencepiece
pip install protobuf

# Audio processing
pip install sounddevice
pip install scipy

# Text-to-speech
pip install gtts

# Additional utilities
pip install numpy
pip install pandas
```

## 🎵 Audio Setup

### Windows Audio Setup
1. **Check microphone permissions**:
   - Settings → Privacy → Microphone
   - Allow apps to access microphone
   
2. **Install audio drivers** (if needed):
   ```bash
   # May require Visual C++ redistributables
   pip install pyaudio
   ```

### macOS Audio Setup
1. **Grant microphone permissions**:
   - System Preferences → Security & Privacy → Microphone
   - Allow Terminal/Python access

2. **Install audio dependencies**:
   ```bash
   # Install PortAudio
   brew install portaudio
   pip install pyaudio
   ```

### Linux Audio Setup
```bash
# Install audio system dependencies
sudo apt install portaudio19-dev python3-pyaudio

# Install ALSA utilities (if needed)
sudo apt install alsa-utils

# Test microphone
arecord -l
```

## 🧠 Model Downloads

The application will automatically download required models on first run:

### Whisper Model (~500MB)
- Downloads automatically on first transcription
- Stored in `~/.cache/whisper/`
- Can take 2-5 minutes depending on internet speed

### NLLB Translation Model (~1.2GB)
- Downloads automatically on first translation
- Stored in `~/.cache/huggingface/`
- Can take 5-10 minutes depending on internet speed

### Manual Model Download (Optional)
```python
# Pre-download models to avoid wait time
import whisper
import transformers

# Download Whisper
whisper.load_model("base")

# Download NLLB
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
```

## 🚀 Running the Application

### 1. Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Start Application
```bash
streamlit run ai_audio_offline.py
```

### 3. Access Application
- Open browser to: `http://localhost:8501`
- Application should load automatically

## 🔧 Configuration Options

### Environment Variables
Create a `.env` file for custom configuration:

```bash
# Model settings
WHISPER_MODEL=base          # tiny, base, small, medium, large
NLLB_MODEL=facebook/nllb-200-distilled-600M

# Audio settings
SAMPLE_RATE=44100
AUDIO_FORMAT=int16

# UI settings
STREAMLIT_PORT=8501
STREAMLIT_HOST=localhost
```

### Custom Model Paths
```python
# In ai_audio_offline.py, modify model loading:
@st.cache_resource
def load_whisper():
    # Use different model size
    return whisper.load_model("small")  # faster but less accurate
    # return whisper.load_model("large")  # slower but more accurate
```

## 🐳 Docker Setup (Optional)

### Build Docker Image
```bash
# Create Dockerfile
cat > Dockerfile << EOF
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "ai_audio_offline.py", "--server.address", "0.0.0.0"]
EOF

# Build image
docker build -t audio-translator .
```

### Run with Docker
```bash
# Run container with microphone access
docker run -p 8501:8501 --device /dev/snd audio-translator
```

## 🛠️ Troubleshooting Setup

### Common Installation Issues

#### PyTorch Installation Problems
```bash
# For CPU-only installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA (GPU) support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Audio Device Issues
```bash
# Test audio recording
python -c "
import sounddevice as sd
import numpy as np
print('Recording test...')
audio = sd.rec(int(2 * 44100), samplerate=44100, channels=1)
sd.wait()
print('Recording successful!')
"
```

#### Model Download Failures
```bash
# Clear cache and retry
rm -rf ~/.cache/whisper/
rm -rf ~/.cache/huggingface/

# Set HuggingFace cache directory
export HF_HOME=/path/to/cache
```

#### Memory Issues
```bash
# Monitor memory usage
pip install psutil

# Reduce model size in code
# Change "base" to "tiny" for Whisper model
```

### Performance Optimization

#### For Faster Startup
```bash
# Pre-download all models
python -c "
import whisper
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

print('Downloading Whisper...')
whisper.load_model('base')

print('Downloading NLLB...')
NllbTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')

print('All models downloaded!')
"
```

#### For Better Audio Quality
```bash
# Install additional audio processing libraries
pip install librosa
pip install noisereduce
pip install webrtcvad
```

## 🔐 Security Considerations

### Network Security
- Application runs locally by default
- No data sent to external servers (except TTS)
- Models stored locally after download

### File Permissions
```bash
# Ensure proper permissions for audio devices
sudo usermod -a -G audio $USER  # Linux
```

### Privacy
- Audio files stored temporarily in system temp directory
- Files automatically cleaned up on exit
- No persistent storage of sensitive data

## 📱 Mobile/Remote Access

### Access from Mobile Device
```bash
# Run with external access
streamlit run ai_audio_offline.py --server.address 0.0.0.0 --server.port 8501

# Access from mobile browser
http://YOUR_COMPUTER_IP:8501
```

### Reverse Proxy Setup (Advanced)
```nginx
# Nginx configuration
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🆘 Getting Help

### Check Logs
```bash
# View Streamlit logs
streamlit run ai_audio_offline.py --logger.level debug

# Python import debugging
python -c "import whisper, transformers, streamlit; print('All imports successful')"
```

### Common Log Locations
- **Windows**: `%USERPROFILE%\.streamlit\logs\`
- **macOS/Linux**: `~/.streamlit/logs/`

### Support Channels
- **GitHub Issues**: Report bugs and feature requests
- **Discussions**: Community support and questions
- **Email**: Technical support for setup issues

---

✅ **Setup Complete!** You should now be able to run the audio translator successfully.

📖 **Next Step**: Read [HOW_TO_USE.md](HOW_TO_USE.md) for detailed usage instructions.
