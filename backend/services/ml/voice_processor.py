import asyncio
import io
import base64
import tempfile
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import torchaudio
import librosa
import whisper
from transformers import (
    Wav2Vec2ForCTC, Wav2Vec2Tokenizer,
    pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
)
import speech_recognition as sr
import pyaudio
import wave
import structlog
from ...utils.config import settings

logger = structlog.get_logger(__name__)

@dataclass
class VoiceAnalysisResult:
    """Result of voice analysis"""
    transcript: str
    language: str
    confidence: float
    audio_quality: float
    speaker_info: Optional[Dict[str, Any]] = None
    emotions: Optional[List[Dict[str, Any]]] = None
    keywords: List[str] = None
    processing_time: float = 0.0

@dataclass
class AudioMetadata:
    """Audio file metadata"""
    duration: float
    sample_rate: int
    channels: int
    format: str
    bit_depth: Optional[int] = None
    file_size: int = 0

class VoiceProcessor:
    """
    Advanced voice processing service for speech-to-text and audio analysis
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Speech recognition models
        self.whisper_model = None
        self.wav2vec2_model = None
        self.wav2vec2_tokenizer = None
        self.speech_pipeline = None
        
        # Audio processing settings
        self.target_sample_rate = 16000
        self.chunk_duration = 30.0  # seconds
        
        # Language detection
        self.language_detector = None
        
        # Supported languages and their models
        self.supported_languages = {
            'en': 'english',
            'es': 'spanish',
            'fr': 'french', 
            'de': 'german',
            'it': 'italian',
            'pt': 'portuguese',
            'ru': 'russian',
            'ja': 'japanese',
            'ko': 'korean',
            'zh': 'chinese',
            'ar': 'arabic',
            'hi': 'hindi'
        }
        
        # Audio quality thresholds
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'fair': 0.5,
            'poor': 0.3
        }
    
    async def initialize(self):
        """Initialize all speech recognition models and processors"""
        logger.info("Initializing voice processor...")
        
        try:
            # Initialize Whisper model (OpenAI's robust speech recognition)
            whisper_model_size = getattr(settings, 'WHISPER_MODEL_SIZE', 'base')
            self.whisper_model = whisper.load_model(whisper_model_size, device=self.device)
            
            # Initialize Wav2Vec2 model for additional accuracy
            model_name = "facebook/wav2vec2-large-960h-lv60-self"
            self.wav2vec2_tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
            self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
            
            # Initialize Hugging Face speech recognition pipeline
            self.speech_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize language detection
            self.language_detector = pipeline(
                "text-classification",
                model="papluca/xlm-roberta-base-language-detection",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Voice processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice processor: {e}")
            raise
    
    async def speech_to_text(
        self,
        audio_data: bytes,
        language: str = None,
        enhance_audio: bool = True,
        return_metadata: bool = False
    ) -> VoiceAnalysisResult:
        """
        Convert speech to text with multiple models for best accuracy
        """
        logger.info("Processing speech-to-text conversion")
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Extract audio metadata
                metadata = await self._extract_audio_metadata(temp_file_path)
                
                # Enhance audio quality if requested
                if enhance_audio:
                    enhanced_path = await self._enhance_audio(temp_file_path)
                    audio_path = enhanced_path
                else:
                    audio_path = temp_file_path
                
                # Load and preprocess audio
                audio_array, sample_rate = await self._load_audio(audio_path)
                
                # Assess audio quality
                audio_quality = await self._assess_audio_quality(audio_array, sample_rate)
                
                # Detect language if not provided
                if not language:
                    language = await self._detect_language(audio_array, sample_rate)
                
                # Perform speech recognition with multiple models
                transcripts = await asyncio.gather(
                    self._transcribe_with_whisper(audio_array, language),
                    self._transcribe_with_wav2vec2(audio_array),
                    self._transcribe_with_pipeline(audio_path, language),
                    return_exceptions=True
                )
                
                # Select best transcript based on confidence
                best_transcript, confidence = self._select_best_transcript(transcripts)
                
                # Extract keywords from transcript
                keywords = await self._extract_keywords(best_transcript, language)
                
                # Calculate processing time
                processing_time = asyncio.get_event_loop().time() - start_time
                
                result = VoiceAnalysisResult(
                    transcript=best_transcript,
                    language=language,
                    confidence=confidence,
                    audio_quality=audio_quality,
                    keywords=keywords,
                    processing_time=processing_time
                )
                
                if return_metadata:
                    result.speaker_info = await self._analyze_speaker(audio_array, sample_rate)
                    result.emotions = await self._analyze_emotions(audio_array, sample_rate)
                
                return result
                
            finally:
                # Clean up temporary files
                for path in [temp_file_path, enhanced_path if enhance_audio else None]:
                    if path and os.path.exists(path):
                        os.unlink(path)
                        
        except Exception as e:
            logger.error(f"Error in speech-to-text conversion: {e}")
            return VoiceAnalysisResult(
                transcript="",
                language="en",
                confidence=0.0,
                audio_quality=0.0,
                processing_time=asyncio.get_event_loop().time() - start_time
            )
    
    async def _extract_audio_metadata(self, file_path: str) -> AudioMetadata:
        """Extract metadata from audio file"""
        try:
            # Load audio file
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=audio_data, sr=sample_rate)
            
            # Get file info
            file_size = os.path.getsize(file_path)
            
            return AudioMetadata(
                duration=duration,
                sample_rate=sample_rate,
                channels=1 if len(audio_data.shape) == 1 else audio_data.shape[0],
                format="wav",
                file_size=file_size
            )
            
        except Exception as e:
            logger.error(f"Error extracting audio metadata: {e}")
            return AudioMetadata(0.0, 16000, 1, "unknown")
    
    async def _enhance_audio(self, input_path: str) -> str:
        """Enhance audio quality using signal processing"""
        try:
            # Load audio
            audio_data, sample_rate = librosa.load(input_path, sr=self.target_sample_rate)
            
            # Noise reduction
            audio_data = self._reduce_noise(audio_data)
            
            # Normalize volume
            audio_data = self._normalize_audio(audio_data)
            
            # Apply pre-emphasis filter
            audio_data = self._apply_preemphasis(audio_data)
            
            # Save enhanced audio
            enhanced_path = input_path.replace('.wav', '_enhanced.wav')
            librosa.output.write_wav(enhanced_path, audio_data, sample_rate)
            
            return enhanced_path
            
        except Exception as e:
            logger.error(f"Error enhancing audio: {e}")
            return input_path
    
    def _reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """Reduce background noise using spectral subtraction"""
        try:
            # Simple noise reduction using spectral subtraction
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first 0.5 seconds
            noise_frame_count = int(0.5 * len(magnitude[0]) / len(audio_data) * 16000)
            noise_magnitude = np.mean(magnitude[:, :noise_frame_count], axis=1, keepdims=True)
            
            # Spectral subtraction
            alpha = 2.0  # Over-subtraction factor
            clean_magnitude = magnitude - alpha * noise_magnitude
            clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
            
            # Reconstruct signal
            clean_stft = clean_magnitude * np.exp(1j * phase)
            clean_audio = librosa.istft(clean_stft)
            
            return clean_audio
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio_data
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio volume"""
        try:
            # Peak normalization
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                normalized = audio_data / max_val * 0.95
            else:
                normalized = audio_data
            
            return normalized.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Audio normalization failed: {e}")
            return audio_data
    
    def _apply_preemphasis(self, audio_data: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        """Apply pre-emphasis filter to enhance high frequencies"""
        try:
            return np.append(audio_data[0], audio_data[1:] - alpha * audio_data[:-1])
        except Exception as e:
            logger.warning(f"Pre-emphasis failed: {e}")
            return audio_data
    
    async def _load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file"""
        try:
            # Load audio with librosa
            audio_data, sample_rate = librosa.load(
                file_path, 
                sr=self.target_sample_rate,
                mono=True
            )
            
            return audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return np.array([]), self.target_sample_rate
    
    async def _assess_audio_quality(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Assess audio quality using multiple metrics"""
        try:
            if len(audio_data) == 0:
                return 0.0
            
            # Signal-to-noise ratio estimation
            snr_score = self._estimate_snr(audio_data)
            
            # Dynamic range assessment
            dynamic_range_score = self._assess_dynamic_range(audio_data)
            
            # Spectral characteristics
            spectral_score = self._assess_spectral_quality(audio_data, sample_rate)
            
            # Combine scores
            quality_score = (snr_score + dynamic_range_score + spectral_score) / 3.0
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Error assessing audio quality: {e}")
            return 0.5
    
    def _estimate_snr(self, audio_data: np.ndarray) -> float:
        """Estimate signal-to-noise ratio"""
        try:
            # Simple SNR estimation using energy-based method
            frame_length = int(0.025 * self.target_sample_rate)  # 25ms frames
            hop_length = int(0.01 * self.target_sample_rate)      # 10ms hop
            
            # Calculate frame energies
            frames = librosa.util.frame(audio_data, frame_length=frame_length, 
                                      hop_length=hop_length, axis=0)
            energies = np.sum(frames ** 2, axis=0)
            
            # Assume lowest 10% of energies represent noise
            noise_threshold = np.percentile(energies, 10)
            signal_threshold = np.percentile(energies, 90)
            
            if noise_threshold > 0:
                snr = 10 * np.log10(signal_threshold / noise_threshold)
                # Normalize to 0-1 scale (assume 20dB is excellent)
                return min(1.0, max(0.0, snr / 20.0))
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"SNR estimation failed: {e}")
            return 0.5
    
    def _assess_dynamic_range(self, audio_data: np.ndarray) -> float:
        """Assess dynamic range of audio"""
        try:
            if len(audio_data) == 0:
                return 0.0
            
            # Calculate RMS values in overlapping windows
            window_size = int(0.1 * self.target_sample_rate)  # 100ms windows
            rms_values = []
            
            for i in range(0, len(audio_data) - window_size, window_size // 2):
                window = audio_data[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                if rms > 0:
                    rms_values.append(rms)
            
            if len(rms_values) > 0:
                dynamic_range = np.max(rms_values) / (np.min(rms_values) + 1e-8)
                # Normalize (assume 100:1 ratio is excellent)
                return min(1.0, np.log10(dynamic_range) / 2.0)
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Dynamic range assessment failed: {e}")
            return 0.5
    
    def _assess_spectral_quality(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Assess spectral characteristics"""
        try:
            # Calculate spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
            
            # Score based on presence of high-frequency content
            mean_centroid = np.mean(spectral_centroids)
            mean_bandwidth = np.mean(spectral_bandwidth)
            
            # Good speech should have centroid around 1-3kHz
            centroid_score = 1.0 - abs(mean_centroid - 2000) / 2000
            bandwidth_score = min(1.0, mean_bandwidth / 2000)
            
            return (centroid_score + bandwidth_score) / 2.0
            
        except Exception as e:
            logger.warning(f"Spectral quality assessment failed: {e}")
            return 0.5
    
    async def _detect_language(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Detect language from audio"""
        try:
            # Use Whisper's built-in language detection
            if self.whisper_model:
                # Get a sample for language detection (first 30 seconds)
                sample_length = min(len(audio_data), 30 * sample_rate)
                audio_sample = audio_data[:sample_length]
                
                # Whisper language detection
                mel = whisper.log_mel_spectrogram(audio_sample).to(self.device)
                _, probs = self.whisper_model.detect_language(mel)
                detected_lang = max(probs, key=probs.get)
                
                # Map to our supported languages
                if detected_lang in self.supported_languages:
                    return detected_lang
                else:
                    return 'en'  # Default to English
            
            return 'en'  # Default fallback
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return 'en'
    
    async def _transcribe_with_whisper(self, audio_data: np.ndarray, language: str = None) -> Tuple[str, float]:
        """Transcribe using OpenAI Whisper"""
        try:
            if self.whisper_model is None:
                return "", 0.0
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                audio_data,
                language=language if language in self.supported_languages else None,
                task='transcribe'
            )
            
            transcript = result['text'].strip()
            
            # Estimate confidence from Whisper segments
            if 'segments' in result:
                confidences = [seg.get('no_speech_prob', 0.5) for seg in result['segments']]
                avg_confidence = 1.0 - np.mean(confidences) if confidences else 0.8
            else:
                avg_confidence = 0.8  # Default confidence
            
            return transcript, avg_confidence
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return "", 0.0
    
    async def _transcribe_with_wav2vec2(self, audio_data: np.ndarray) -> Tuple[str, float]:
        """Transcribe using Wav2Vec2"""
        try:
            if self.wav2vec2_model is None or self.wav2vec2_tokenizer is None:
                return "", 0.0
            
            # Prepare input
            input_values = self.wav2vec2_tokenizer(
                audio_data,
                return_tensors="pt",
                sampling_rate=self.target_sample_rate
            ).input_values.to(self.device)
            
            # Get logits
            with torch.no_grad():
                logits = self.wav2vec2_model(input_values).logits
            
            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcript = self.wav2vec2_tokenizer.batch_decode(predicted_ids)[0].strip()
            
            # Estimate confidence from logits
            probs = torch.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]
            confidence = torch.mean(max_probs).item()
            
            return transcript, confidence
            
        except Exception as e:
            logger.error(f"Wav2Vec2 transcription failed: {e}")
            return "", 0.0
    
    async def _transcribe_with_pipeline(self, audio_path: str, language: str = None) -> Tuple[str, float]:
        """Transcribe using Hugging Face pipeline"""
        try:
            if self.speech_pipeline is None:
                return "", 0.0
            
            # Transcribe
            result = self.speech_pipeline(audio_path)
            
            if isinstance(result, dict):
                transcript = result.get('text', '').strip()
                confidence = 0.7  # Default confidence for pipeline
            else:
                transcript = str(result).strip()
                confidence = 0.7
            
            return transcript, confidence
            
        except Exception as e:
            logger.error(f"Pipeline transcription failed: {e}")
            return "", 0.0
    
    def _select_best_transcript(self, transcripts: List[Tuple[str, float]]) -> Tuple[str, float]:
        """Select the best transcript from multiple models"""
        try:
            valid_transcripts = [
                (text, conf) for text, conf in transcripts 
                if isinstance((text, conf), tuple) and text.strip()
            ]
            
            if not valid_transcripts:
                return "", 0.0
            
            # Select transcript with highest confidence
            best_transcript, best_confidence = max(valid_transcripts, key=lambda x: x[1])
            
            # If multiple transcripts are similar, boost confidence
            if len(valid_transcripts) > 1:
                # Simple similarity check
                similarities = []
                for other_text, _ in valid_transcripts:
                    if other_text != best_transcript:
                        similarity = self._calculate_text_similarity(best_transcript, other_text)
                        similarities.append(similarity)
                
                if similarities and np.mean(similarities) > 0.7:
                    best_confidence = min(1.0, best_confidence + 0.1)
            
            return best_transcript, best_confidence
            
        except Exception as e:
            logger.error(f"Error selecting best transcript: {e}")
            return "", 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        try:
            # Simple word-level Jaccard similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    async def _extract_keywords(self, text: str, language: str) -> List[str]:
        """Extract keywords from transcript"""
        try:
            if not text.strip():
                return []
            
            # Simple keyword extraction using word frequency
            words = text.lower().split()
            
            # Remove common stop words (basic list)
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that',
                'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me',
                'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                'must', 'can'
            }
            
            # Filter and count words
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top keywords
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return [word for word, freq in keywords]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    async def _analyze_speaker(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze speaker characteristics"""
        try:
            # Extract basic acoustic features
            pitch = librosa.piptrack(y=audio_data, sr=sample_rate)
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            
            # Estimate speaker characteristics
            mean_pitch = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 100
            
            # Rough gender estimation based on pitch (very basic)
            gender_estimate = "female" if mean_pitch > 165 else "male"
            
            return {
                "estimated_gender": gender_estimate,
                "mean_pitch": float(mean_pitch),
                "tempo": float(tempo),
                "speech_rate": "normal"  # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Error analyzing speaker: {e}")
            return {}
    
    async def _analyze_emotions(self, audio_data: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Analyze emotional content in speech"""
        try:
            # Extract prosodic features
            pitch, _ = librosa.piptrack(y=audio_data, sr=sample_rate)
            energy = librosa.feature.rms(y=audio_data)[0]
            
            # Simple emotion estimation based on acoustic features
            mean_pitch = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 100
            pitch_variance = np.var(pitch[pitch > 0]) if np.any(pitch > 0) else 0
            mean_energy = np.mean(energy)
            
            emotions = []
            
            # Basic emotion rules (very simplified)
            if mean_energy > 0.5 and pitch_variance > 1000:
                emotions.append({"emotion": "excited", "confidence": 0.7})
            elif mean_energy < 0.2:
                emotions.append({"emotion": "calm", "confidence": 0.6})
            elif mean_pitch > 200:
                emotions.append({"emotion": "surprised", "confidence": 0.5})
            else:
                emotions.append({"emotion": "neutral", "confidence": 0.8})
            
            return emotions
            
        except Exception as e:
            logger.error(f"Error analyzing emotions: {e}")
            return [{"emotion": "neutral", "confidence": 0.5}]
    
    async def convert_audio_format(
        self,
        audio_data: bytes,
        target_format: str = 'wav',
        target_sample_rate: int = None
    ) -> bytes:
        """Convert audio to target format"""
        try:
            target_sample_rate = target_sample_rate or self.target_sample_rate
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_input:
                temp_input.write(audio_data)
                temp_input_path = temp_input.name
            
            try:
                # Load and convert
                audio_array, sr = librosa.load(temp_input_path, sr=target_sample_rate)
                
                # Save in target format
                output_path = temp_input_path.replace('.wav', f'.{target_format}')
                
                if target_format == 'wav':
                    librosa.output.write_wav(output_path, audio_array, target_sample_rate)
                else:
                    # For other formats, use librosa with soundfile backend
                    import soundfile as sf
                    sf.write(output_path, audio_array, target_sample_rate)
                
                # Read converted file
                with open(output_path, 'rb') as f:
                    converted_data = f.read()
                
                return converted_data
                
            finally:
                # Clean up
                for path in [temp_input_path, output_path]:
                    if os.path.exists(path):
                        os.unlink(path)
                        
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            return audio_data
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get voice processor statistics"""
        return {
            "models_loaded": {
                "whisper": self.whisper_model is not None,
                "wav2vec2": self.wav2vec2_model is not None,
                "speech_pipeline": self.speech_pipeline is not None,
                "language_detector": self.language_detector is not None
            },
            "supported_languages": list(self.supported_languages.keys()),
            "target_sample_rate": self.target_sample_rate,
            "chunk_duration": self.chunk_duration
        }
    
    async def health_check(self) -> bool:
        """Check if voice processor is healthy"""
        try:
            # Test with silent audio
            test_audio = np.zeros(self.target_sample_rate)  # 1 second of silence
            result = await self.speech_to_text(
                test_audio.tobytes(),
                language='en'
            )
            return result is not None
        except Exception as e:
            logger.error(f"Voice processor health check failed: {e}")
            return False