# 🎙️ Offline Audio Translator

A real-time, multilingual audio translator that runs completely offline using state-of-the-art AI models. Record audio in any language and get instant translations with text-to-speech output.

![Demo](https://img.shields.io/badge/Demo-YouTube-red?style=for-the-badge&logo=youtube)  
**🎥 [Watch Demo Video](https://youtu.be/Xu-lkroIrDs)**

## ✨ Features

- 🎤 **Real-time audio recording** with customizable duration
- 🧠 **Automatic language detection** using OpenAI Whisper
- 🌍 **12+ language support** including Hindi, French, Spanish, German, Chinese, Arabic, and more
- 🔄 **Neural machine translation** powered by Meta's NLLB-200 model
- 🔊 **Text-to-speech synthesis** for translated output
- 💾 **Export functionality** for transcriptions and translations
- 🔒 **100% offline operation** (no API keys or internet required, except for TTS)
- 📱 **User-friendly web interface** built with Streamlit

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **UI Framework** | Streamlit | Interactive web interface |
| **Speech Recognition** | OpenAI Whisper | Audio transcription & language detection |
| **Translation** | Meta NLLB-200 | Neural machine translation |
| **Text-to-Speech** | Google TTS (gTTS) | Audio synthesis |
| **Audio Processing** | SoundDevice, SciPy | Recording and audio manipulation |
| **Deep Learning** | PyTorch, Transformers | Model inference |

## 🌐 Supported Languages

| Language | Transcription | Translation | TTS |
|----------|:-------------:|:-----------:|:---:|
| English | ✅ | ✅ | ✅ |
| Hindi | ✅ | ✅ | ✅ |
| Spanish | ✅ | ✅ | ✅ |
| French | ✅ | ✅ | ✅ |
| German | ✅ | ✅ | ✅ |
| Chinese | ✅ | ✅ | ✅ |
| Japanese | ✅ | ✅ | ✅ |
| Korean | ✅ | ✅ | ✅ |
| Arabic | ✅ | ✅ | ✅ |
| Portuguese | ✅ | ✅ | ✅ |
| Italian | ✅ | ✅ | ✅ |
| Russian | ✅ | ✅ | ✅ |

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Microphone access
- 4GB+ RAM (for model loading)
- 2GB+ free disk space (for models)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/offline-audio-translator.git
   cd offline-audio-translator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run ai_audio_offline.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 📖 How It Works

### 1. **Audio Recording**
- Choose recording duration (1-20 seconds)
- Click "Record Audio" and speak clearly
- Audio is saved locally as WAV file

### 2. **Transcription**
- Whisper automatically detects the source language
- Transcribes audio to text in the original language
- Displays confidence score and detected language

### 3. **Translation**
- NLLB-200 translates text to target language
- Preserves semantic meaning and context
- Supports 12+ language pairs

### 4. **Text-to-Speech**
- Generates audio in target language
- Playable directly in browser
- Uses Google TTS service

### 5. **Export**
- Download transcriptions as TXT files
- Save translations for later use
- Timestamped filenames for organization

## 🎯 Use Cases

- **Business Meetings**: Real-time translation for international calls
- **Travel**: Communicate in foreign countries
- **Education**: Language learning and content creation
- **Accessibility**: Audio content translation for diverse audiences
- **Content Creation**: Multilingual video/podcast production
- **Research**: Cross-language document analysis

## ⚠️ Known Limitations

### Whisper Transcription
- **Auto-translation issue**: Whisper sometimes translates to English instead of transcribing in the original language
- **Workaround**: Manual text input option available
- **Best for**: Clear speech in quiet environments

### TTS Requirements
- **Internet needed**: Google TTS requires internet connection
- **Alternative**: Use offline TTS libraries for fully offline operation

### Model Performance
- **Initial load time**: First run downloads ~2GB of models
- **RAM usage**: Requires 4GB+ RAM for optimal performance
- **Language accuracy**: Varies by language and audio quality

## 🔧 Troubleshooting

### Common Issues

**"No audio recorded" error:**
- Check microphone permissions
- Ensure microphone is not used by other applications
- Try adjusting recording duration

**Model loading failures:**
- Ensure stable internet for initial model download
- Check available disk space (2GB+ required)
- Restart application if models fail to load

**Translation quality issues:**
- Speak clearly and minimize background noise
- Try manual text input for better accuracy
- Check source language detection is correct

### Performance Optimization

**For better accuracy:**
- Use external microphone for clearer audio
- Record in quiet environment
- Speak slowly and clearly
- Use shorter recording durations (5-10 seconds)

**For faster processing:**
- Close other resource-intensive applications
- Use CPU with 4+ cores for better performance
- Consider GPU acceleration for PyTorch models

## 📁 Project Structure

```
offline-audio-translator/
├── ai_audio_offline.py      # Main application file
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── SETUP.md               # Detailed setup instructions
├── HOW_TO_USE.md          # User guide
├── .gitignore             # Git ignore rules
└── assets/                # Screenshots and demo files
    ├── demo_screenshot.png
    └── architecture.png
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Areas for Improvement
- Additional language support
- Offline TTS integration
- Better audio preprocessing
- Mobile app version
- API endpoint creation
- Docker containerization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for the incredible Whisper model
- **Meta** for the NLLB-200 translation model
- **Streamlit** for the amazing web framework
- **Hugging Face** for model hosting and transformers library

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/offline-audio-translator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/offline-audio-translator/discussions)
- **Email**: your.email@example.com

## 🔗 Links

- **Demo Video**: https://youtu.be/Xu-lkroIrDs
- **Documentation**: [Wiki](https://github.com/yourusername/offline-audio-translator/wiki)
- **LinkedIn Post**: [Project Announcement](https://linkedin.com/in/yourprofile)

---

⭐ **Star this repository** if you find it helpful!

📢 **Share it** with others who might benefit from multilingual audio translation!
