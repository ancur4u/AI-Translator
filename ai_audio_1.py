import os
os.environ["STREAMLIT_WATCHER_IGNORE"] = "torch"

import streamlit as st
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
from datetime import datetime
import whisper
from gtts import gTTS
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
import atexit
import contextlib
try:
    import torch
except ImportError:
    torch = None

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

# Streamlit config
st.set_page_config(page_title="🎙️ Multi-Modal Audio Translator", layout="wide")

# Session state for audio recording tab
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "detected_language" not in st.session_state:
    st.session_state.detected_language = None

# Session state for YouTube tab
if "youtube_transcription" not in st.session_state:
    st.session_state.youtube_transcription = ""
if "youtube_translated_text" not in st.session_state:
    st.session_state.youtube_translated_text = ""
if "youtube_detected_language" not in st.session_state:
    st.session_state.youtube_detected_language = None
if "youtube_audio_path" not in st.session_state:
    st.session_state.youtube_audio_path = None

st.title("🎙️ Multi-Modal Audio Translator")
st.markdown("Record audio or transcribe YouTube videos, then translate with AI — 100% offline (except YouTube download).")

# Load Whisper with error handling and specific model configuration
@st.cache_resource
def load_whisper():
    try:
        # Use base model which is good for transcription without translation
        model = whisper.load_model("base")
        return model
    except Exception as e:
        st.error(f"Failed to load Whisper model: {e}")
        return None

model_whisper = load_whisper()

# Load NLLB with proper tokenizer and error handling
@st.cache_resource
def load_nllb():
    try:
        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = NllbTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load NLLB model: {e}")
        return None, None

tokenizer, nllb_model = load_nllb()

# Language map for NLLB
lang_map = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "German": "deu_Latn",
    "Chinese": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Arabic": "arb_Arab",
    "Portuguese": "por_Latn",
    "Italian": "ita_Latn",
    "Russian": "rus_Cyrl"
}

# TTS language mapping (corrected for gTTS)
tts_lang_map = {
    "English": "en",
    "Hindi": "hi", 
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Italian": "it",
    "Russian": "ru"
}

# Whisper to NLLB language mapping
whisper_to_nllb = {
    "en": "English",
    "hi": "Hindi", 
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian"
}

# Function to download YouTube audio
def download_youtube_audio(url):
    """Download audio from YouTube URL"""
    if yt_dlp is None:
        st.error("yt-dlp not installed. Please install it with: pip install yt-dlp")
        return None, None
    
    try:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"youtube_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path + '.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'Unknown')
            
        audio_file = output_path + '.wav'
        return audio_file, title
        
    except Exception as e:
        st.error(f"Failed to download YouTube audio: {e}")
        return None, None

# Function to perform translation (shared between tabs)
def perform_translation(transcription, detected_language, target_lang, session_key_prefix=""):
    """Perform translation with proper error handling"""
    if tokenizer is None or nllb_model is None:
        st.error("❌ NLLB model not loaded. Please restart the app.")
        return None
    
    try:
        source_language = detected_language
        st.info(f"🎯 Translating from: **{source_language}** → **{target_lang}**")
        
        # Check if translation is needed
        if source_language == target_lang:
            st.warning("⚠️ Source and target languages are the same. No translation needed.")
            return transcription
        
        # Show original text for reference
        with st.expander(f"📖 Original Text ({source_language})"):
            st.write(transcription)
        
        # Get language codes for NLLB
        source_lang_code = lang_map.get(source_language, "eng_Latn")
        target_lang_code = lang_map.get(target_lang, "eng_Latn")
        
        # Prepare text for translation
        input_text = transcription.strip()
        
        # For very long texts, split into chunks
        max_chunk_length = 400  # Conservative chunk size
        chunks = []
        words = input_text.split()
        
        if len(words) > max_chunk_length:
            st.info(f"📄 Long text detected ({len(words)} words). Processing in chunks for complete translation...")
            
            # Split into chunks at sentence boundaries when possible
            sentences = input_text.replace('!', '.').replace('?', '.').split('.')
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Check if adding this sentence would exceed chunk size
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                if len(test_chunk.split()) <= max_chunk_length:
                    current_chunk = test_chunk
                else:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
            
            # Add the last chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            chunks = [input_text]
        
        st.info(f"🔄 Processing {len(chunks)} chunk(s)...")
        
        # Set the source language for the tokenizer
        tokenizer.src_lang = source_lang_code
        
        translated_chunks = []
        
        for i, chunk in enumerate(chunks):
            st.write(f"Processing chunk {i+1}/{len(chunks)}...")
            
            # Tokenize the chunk
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Get the target language token ID
            forced_bos_token_id = None
            
            if hasattr(tokenizer, 'lang_code_to_id') and target_lang_code in tokenizer.lang_code_to_id:
                forced_bos_token_id = tokenizer.lang_code_to_id[target_lang_code]
            elif hasattr(tokenizer, 'convert_tokens_to_ids'):
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang_code)
                if forced_bos_token_id == tokenizer.unk_token_id:
                    forced_bos_token_id = None
            elif hasattr(tokenizer, 'get_vocab'):
                vocab = tokenizer.get_vocab()
                forced_bos_token_id = vocab.get(target_lang_code)
            
            # Translation parameters optimized for completeness
            translation_params = {
                "max_length": 1024,  # Increased for longer outputs
                "min_length": max(5, len(chunk.split()) // 3),  # Ensure minimum output
                "num_beams": 4,  # Reduced for speed while maintaining quality
                "early_stopping": False,  # Don't stop early
                "length_penalty": 0.8,  # Slightly favor longer translations
                "repetition_penalty": 1.1,
                "do_sample": False,
                "no_repeat_ngram_size": 3,
            }
            
            if forced_bos_token_id is not None:
                translation_params["forced_bos_token_id"] = forced_bos_token_id
            
            # Generate translation for this chunk
            with torch.no_grad() if torch is not None else contextlib.nullcontext():
                generated_tokens = nllb_model.generate(
                    **inputs,
                    **translation_params
                )
            
            # Decode the translation
            chunk_translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            
            # Clean up the chunk translation
            chunk_translation = chunk_translation.strip()
            if chunk.lower() in chunk_translation.lower():
                chunk_translation = chunk_translation.replace(chunk, "").strip()
            
            if len(chunk_translation.strip()) >= 3:
                translated_chunks.append(chunk_translation)
            else:
                st.warning(f"⚠️ Chunk {i+1} produced short output, keeping original")
                translated_chunks.append(chunk)
        
        # Combine all translated chunks
        final_translation = " ".join(translated_chunks).strip()
        
        # Final cleanup
        if len(final_translation.strip()) < 10:
            st.error("❌ Translation too short. There may be an issue with the model.")
            return None
        
        # Display the translation with statistics
        col1, col2 = st.columns([2, 1])
        with col1:
            st.text_area(
                f"🌍 Complete Translation ({target_lang})",
                final_translation,
                height=200,  # Increased height for longer translations
                help=f"Text translated from {source_language} to {target_lang}",
                key=f"{session_key_prefix}_translation_display"
            )
        with col2:
            st.success("✅ Translation Complete!")
            st.write(f"**From:** {source_language}")
            st.write(f"**To:** {target_lang}")
            
            # Detailed word count comparison
            source_words = len(input_text.split())
            target_words = len(final_translation.split())
            compression_ratio = (source_words - target_words) / source_words * 100 if source_words > 0 else 0
            
            st.metric("Words", f"{source_words} → {target_words}")
            st.metric("Compression", f"{compression_ratio:.1f}%")
            st.metric("Chunks Processed", len(chunks))
            
            # Show completion percentage
            if target_words > 0:
                completion_estimate = min(100, (target_words / max(source_words * 0.5, 1)) * 100)
                st.metric("Estimated Completeness", f"{completion_estimate:.1f}%")
            
            # Show model info
            st.caption("🤖 NLLB-200 Model")
            
            # Warning if translation seems too short
            if target_words < source_words * 0.3:
                st.warning("⚠️ Translation seems shorter than expected. Check for completeness.")
        
        return final_translation
            
    except Exception as e:
        st.error(f"❌ Translation failed: {e}")
        st.error("Please check your text and language selection.")
        st.error(f"Error details: {str(e)}")
        return None

# Create tabs
tab1, tab2 = st.tabs(["🎤 Audio Recording", "📺 YouTube Transcription"])

# TAB 1: Audio Recording (Original functionality)
with tab1:
    st.header("🎤 Record and Translate Audio")
    
    # Language selection with automatic detection option
    col1, col2 = st.columns(2)
    with col1:
        auto_detect = st.checkbox("🤖 Auto-detect source language", value=True, key="audio_auto_detect")
        if not auto_detect:
            source_lang = st.selectbox("🗣️ Source Language", list(lang_map.keys()), index=0, key="audio_source_lang")
        else:
            st.info("Source language will be detected automatically from audio")
            
    with col2:
        target_lang = st.selectbox("🌍 Target Language", list(lang_map.keys()), index=1, key="audio_target_lang")

    # Progress indicators with detected language
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Audio", "✅" if st.session_state.audio_path else "⏳")
    with col2:
        st.metric("Transcription", "✅" if st.session_state.transcription else "⏳")
    with col3:
        detected_lang_display = st.session_state.detected_language if st.session_state.detected_language else "⏳"
        st.metric("Detected Lang", detected_lang_display)
    with col4:
        st.metric("Translation", "✅" if st.session_state.translated_text else "⏳")

    # Record audio with improved error handling
    record_seconds = st.slider("🎤 Recording duration (seconds)", 1, 20, 5, key="audio_record_seconds")
    if st.button("🔴 Record Audio", key="audio_record_btn"):
        try:
            fs = 44100
            st.info("Recording... Speak now.")
            
            # Record audio
            audio = sd.rec(int(record_seconds * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()
            
            # Check if audio was actually recorded
            if audio is None or len(audio) == 0:
                st.error("❌ No audio recorded. Check your microphone.")
            else:
                # Save audio file
                wav_path = os.path.join(tempfile.gettempdir(), f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
                wav.write(wav_path, fs, audio)
                st.session_state.audio_path = wav_path
                st.success("✅ Recording saved!")
                
        except Exception as e:
            st.error(f"❌ Recording failed: {e}")

    # Transcription with automatic language detection and source language preservation
    if st.session_state.audio_path and st.button("📝 Transcribe Audio", key="audio_transcribe_btn"):
        if model_whisper is None:
            st.error("❌ Whisper model not loaded. Please restart the app.")
        else:
            try:
                with st.spinner("Transcribing in original language..."):
                    # CRITICAL: Use task="transcribe" to keep original language, NOT "translate"
                    result = model_whisper.transcribe(
                        st.session_state.audio_path,
                        task="transcribe",     # This ensures NO translation, only transcription
                        language=None,         # Auto-detect but don't translate
                        fp16=False,
                        verbose=False
                    )
                    
                    # Get the transcription in the ORIGINAL language (not translated)
                    original_transcription = result["text"]
                    detected_lang_code = result.get("language", "en")
                    
                    # Map Whisper language code to our language names
                    detected_language = whisper_to_nllb.get(detected_lang_code, "English")
                    
                    # Store the original language transcription
                    st.session_state.transcription = original_transcription
                    st.session_state.detected_language = detected_language
                    
                    # Display results - transcription should be in ORIGINAL language
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text_area(
                            f"📝 Transcription (Original: {detected_language})", 
                            original_transcription, 
                            height=100,
                            help=f"Audio transcribed in original {detected_language} language - NOT translated",
                            key="audio_transcription_display"
                        )
                        
                        # Verification message
                        if detected_language != "English":
                            st.info(f"✅ Text is in {detected_language} (original language)")
                        else:
                            st.info("✅ Text is in English (original language)")
                            
                    with col2:
                        st.success(f"🎯 Detected Language: **{detected_language}**")
                        st.write(f"**Language Code:** {detected_lang_code}")
                        
                        # Show confidence if available
                        if result.get("segments"):
                            avg_logprob = sum(seg.get("avg_logprob", -1) for seg in result["segments"]) / len(result["segments"])
                            confidence_percent = min(100, max(0, (avg_logprob + 1) * 100))
                            st.metric("Confidence", f"{confidence_percent:.1f}%")
                        
                        # Important note
                        st.warning("⚠️ This is the ORIGINAL language text")
                        st.info("💡 Use 'Translate Text' button to convert to target language")
                            
            except Exception as e:
                st.error(f"❌ Transcription failed: {e}")
                st.error("Make sure your audio file is valid and try again.")

    # Translation
    if st.session_state.transcription and st.button("🌐 Translate Text", key="audio_translate_btn"):
        if auto_detect and st.session_state.detected_language:
            result = perform_translation(st.session_state.transcription, st.session_state.detected_language, target_lang, "audio")
            if result:
                st.session_state.translated_text = result

    # Text-to-Speech playback with improved language mapping
    if st.session_state.translated_text and st.button("🔊 Play Translated Audio", key="audio_tts_btn"):
        try:
            # Use the corrected TTS language mapping
            tts_lang = tts_lang_map.get(target_lang, "en")  # Default to English if not found
            
            with st.spinner("Generating audio..."):
                tts = gTTS(st.session_state.translated_text, lang=tts_lang)
                tts_path = os.path.join(tempfile.gettempdir(), f"translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
                tts.save(tts_path)
                
                with open(tts_path, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")
                    
        except Exception as e:
            st.error(f"❌ TTS failed: {e}")
            st.info("Note: TTS requires internet connection for gTTS service")

    # Export functionality
    st.markdown("---")
    st.subheader("📥 Export Results")

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.transcription:
            st.download_button(
                "📄 Download Transcription", 
                st.session_state.transcription, 
                file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                key="audio_download_transcription"
            )

    with col2:
        if st.session_state.translated_text:
            st.download_button(
                "🌐 Download Translation", 
                st.session_state.translated_text, 
                file_name=f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                key="audio_download_translation"
            )

# TAB 2: YouTube Transcription
with tab2:
    st.header("📺 YouTube Video Transcription")
    
    # Check if yt-dlp is available
    if yt_dlp is None:
        st.error("⚠️ YouTube functionality requires yt-dlp. Install it with: `pip install yt-dlp`")
    else:
        st.success("✅ YouTube functionality available")
    
    # Language selection for YouTube
    col1, col2 = st.columns(2)
    with col1:
        st.info("🤖 Source language will be auto-detected from video")
    with col2:
        youtube_target_lang = st.selectbox("🌍 Target Language", list(lang_map.keys()), index=1, key="youtube_target_lang")

    # Progress indicators for YouTube
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Video", "✅" if st.session_state.youtube_audio_path else "⏳")
    with col2:
        st.metric("Transcription", "✅" if st.session_state.youtube_transcription else "⏳")
    with col3:
        youtube_detected_lang_display = st.session_state.youtube_detected_language if st.session_state.youtube_detected_language else "⏳"
        st.metric("Detected Lang", youtube_detected_lang_display)
    with col4:
        st.metric("Translation", "✅" if st.session_state.youtube_translated_text else "⏳")

    # YouTube URL input
    youtube_url = st.text_input(
        "🔗 Enter YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Enter a valid YouTube video URL",
        key="youtube_url_input"
    )

    # Download YouTube audio
    if youtube_url and st.button("📥 Download Audio from YouTube", key="youtube_download_btn"):
        if yt_dlp is None:
            st.error("yt-dlp not installed. Please install it with: pip install yt-dlp")
        else:
            with st.spinner("Downloading audio from YouTube..."):
                audio_path, title = download_youtube_audio(youtube_url)
                
                if audio_path and os.path.exists(audio_path):
                    st.session_state.youtube_audio_path = audio_path
                    st.success(f"✅ Downloaded: {title}")
                    st.info(f"📁 Audio saved to: {os.path.basename(audio_path)}")
                else:
                    st.error("❌ Failed to download audio. Check the URL and try again.")

    # Transcribe YouTube audio
    if st.session_state.youtube_audio_path and st.button("📝 Transcribe YouTube Audio", key="youtube_transcribe_btn"):
        if model_whisper is None:
            st.error("❌ Whisper model not loaded. Please restart the app.")
        else:
            try:
                with st.spinner("Transcribing YouTube audio..."):
                    result = model_whisper.transcribe(
                        st.session_state.youtube_audio_path,
                        task="transcribe",
                        language=None,
                        fp16=False,
                        verbose=False
                    )
                    
                    # Get transcription and detected language
                    youtube_transcription = result["text"]
                    detected_lang_code = result.get("language", "en")
                    detected_language = whisper_to_nllb.get(detected_lang_code, "English")
                    
                    # Store results
                    st.session_state.youtube_transcription = youtube_transcription
                    st.session_state.youtube_detected_language = detected_language
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text_area(
                            f"📝 YouTube Transcription ({detected_language})",
                            youtube_transcription,
                            height=150,
                            help=f"Video transcribed in original {detected_language} language",
                            key="youtube_transcription_display"
                        )
                        
                        if detected_language != "English":
                            st.info(f"✅ Text is in {detected_language} (original language)")
                        else:
                            st.info("✅ Text is in English (original language)")
                    
                    with col2:
                        st.success(f"🎯 Detected Language: **{detected_language}**")
                        st.write(f"**Language Code:** {detected_lang_code}")
                        
                        # Show confidence and length info
                        if result.get("segments"):
                            avg_logprob = sum(seg.get("avg_logprob", -1) for seg in result["segments"]) / len(result["segments"])
                            confidence_percent = min(100, max(0, (avg_logprob + 1) * 100))
                            st.metric("Confidence", f"{confidence_percent:.1f}%")
                        
                        # Show transcription stats
                        word_count = len(youtube_transcription.split())
                        char_count = len(youtube_transcription)
                        st.metric("Words", word_count)
                        st.metric("Characters", char_count)
                        
                        st.info("💡 Use 'Translate Text' button to convert to target language")
                        
            except Exception as e:
                st.error(f"❌ YouTube transcription failed: {e}")

    # Translate YouTube transcription
    if st.session_state.youtube_transcription and st.button("🌐 Translate YouTube Text", key="youtube_translate_btn"):
        result = perform_translation(
            st.session_state.youtube_transcription, 
            st.session_state.youtube_detected_language, 
            youtube_target_lang, 
            "youtube"
        )
        if result:
            st.session_state.youtube_translated_text = result

    # Text-to-Speech for YouTube translation
    if st.session_state.youtube_translated_text and st.button("🔊 Play YouTube Translation", key="youtube_tts_btn"):
        try:
            tts_lang = tts_lang_map.get(youtube_target_lang, "en")
            
            with st.spinner("Generating audio..."):
                tts = gTTS(st.session_state.youtube_translated_text, lang=tts_lang)
                tts_path = os.path.join(tempfile.gettempdir(), f"youtube_translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
                tts.save(tts_path)
                
                with open(tts_path, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")
                    
        except Exception as e:
            st.error(f"❌ TTS failed: {e}")
            st.info("Note: TTS requires internet connection for gTTS service")

    # Export functionality for YouTube
    st.markdown("---")
    st.subheader("📥 Export YouTube Results")

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.youtube_transcription:
            st.download_button(
                "📄 Download YouTube Transcription", 
                st.session_state.youtube_transcription, 
                file_name=f"youtube_transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                key="youtube_download_transcription"
            )

    with col2:
        if st.session_state.youtube_translated_text:
            st.download_button(
                "🌐 Download YouTube Translation", 
                st.session_state.youtube_translated_text, 
                file_name=f"youtube_translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                key="youtube_download_translation"
            )

# Clear session buttons
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🗑️ Clear Audio Tab", key="clear_audio"):
        st.session_state.audio_path = None
        st.session_state.transcription = ""
        st.session_state.translated_text = ""
        st.session_state.detected_language = None
        st.rerun()

with col2:
    if st.button("🗑️ Clear YouTube Tab", key="clear_youtube"):
        st.session_state.youtube_transcription = ""
        st.session_state.youtube_translated_text = ""
        st.session_state.youtube_detected_language = None
        st.session_state.youtube_audio_path = None
        st.rerun()

with col3:
    if st.button("🗑️ Clear All", key="clear_all"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Cleanup function for temporary files
def cleanup_temp_files():
    try:
        temp_dir = tempfile.gettempdir()
        for filename in os.listdir(temp_dir):
            if filename.startswith(('recording_', 'translated_', 'youtube_audio_')) and (filename.endswith('.wav') or filename.endswith('.mp3')):
                try:
                    os.remove(os.path.join(temp_dir, filename))
                except:
                    pass
    except:
        pass

# Register cleanup function
atexit.register(cleanup_temp_files)