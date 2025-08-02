"""
Voice Processor for AstraFind
Handles voice/speech processing and recognition
"""

import asyncio
import base64
import io
import wave
from typing import Optional, Dict, Any
import tempfile
import os

import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
import structlog

from utils.config import settings

logger = structlog.get_logger(__name__)

class VoiceProcessor:
    """Handles voice processing and speech recognition"""
    
    def __init__(self):
        self.initialized = False
        self.recognizer = None
    
    async def initialize(self) -> None:
        """Initialize the voice processor"""
        if self.initialized:
            return
        
        try:
            # Initialize speech recognizer
            self.recognizer = sr.Recognizer()
            
            # Configure recognizer settings
            self.recognizer.energy_threshold = 4000
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            
            self.initialized = True
            logger.info("Voice Processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Voice Processor: {e}")
            raise
    
    async def process_audio(
        self,
        audio_data: str,
        format: str = "wav",
        language: str = "en-US"
    ) -> Dict[str, Any]:
        """
        Process audio data and convert to text
        
        Args:
            audio_data: Base64 encoded audio data
            format: Audio format (wav, mp3, etc.)
            language: Speech recognition language
            
        Returns:
            Dict with transcription results
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
            
            # Convert audio to WAV format if needed
            if format.lower() != "wav":
                audio_bytes = await self._convert_audio_format(audio_bytes, format, "wav")
            
            # Create audio file object
            audio_file = io.BytesIO(audio_bytes)
            
            # Load audio with speech recognition
            with sr.AudioFile(audio_file) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Record audio
                audio = self.recognizer.record(source)
            
            # Perform speech recognition
            transcription = await self._recognize_speech(audio, language)
            
            return {
                "success": True,
                "transcription": transcription.get("text", ""),
                "confidence": transcription.get("confidence", 0.0),
                "language": language,
                "duration": transcription.get("duration", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Failed to process audio: {e}")
            return {
                "success": False,
                "error": str(e),
                "transcription": "",
                "confidence": 0.0
            }
    
    async def _recognize_speech(self, audio: sr.AudioData, language: str) -> Dict[str, Any]:
        """Perform speech recognition on audio data"""
        try:
            # Try Google Speech Recognition first
            try:
                text = self.recognizer.recognize_google(audio, language=language)
                confidence = 0.8  # Google doesn't provide confidence scores
                return {
                    "text": text,
                    "confidence": confidence,
                    "provider": "google"
                }
            except sr.UnknownValueError:
                logger.warning("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                logger.warning(f"Google Speech Recognition service error: {e}")
            
            # Fallback to other recognition services
            # You can add more recognition services here
            
            # For now, return empty result
            return {
                "text": "",
                "confidence": 0.0,
                "provider": "none"
            }
            
        except Exception as e:
            logger.error(f"Speech recognition failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "provider": "error"
            }
    
    async def _convert_audio_format(
        self,
        audio_bytes: bytes,
        input_format: str,
        output_format: str
    ) -> bytes:
        """Convert audio between different formats"""
        try:
            # Create temporary file for input
            with tempfile.NamedTemporaryFile(suffix=f".{input_format}", delete=False) as temp_input:
                temp_input.write(audio_bytes)
                temp_input_path = temp_input.name
            
            # Create temporary file for output
            with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as temp_output:
                temp_output_path = temp_output.name
            
            try:
                # Load audio with pydub
                audio = AudioSegment.from_file(temp_input_path, format=input_format)
                
                # Export to output format
                audio.export(temp_output_path, format=output_format)
                
                # Read converted audio
                with open(temp_output_path, 'rb') as f:
                    converted_bytes = f.read()
                
                return converted_bytes
                
            finally:
                # Clean up temporary files
                if os.path.exists(temp_input_path):
                    os.unlink(temp_input_path)
                if os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)
                    
        except Exception as e:
            logger.error(f"Failed to convert audio format: {e}")
            raise
    
    async def analyze_audio_quality(self, audio_data: str, format: str = "wav") -> Dict[str, Any]:
        """Analyze audio quality and characteristics"""
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
            
            # Convert to WAV if needed
            if format.lower() != "wav":
                audio_bytes = await self._convert_audio_format(audio_bytes, format, "wav")
            
            # Load audio with pydub
            audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
            
            # Analyze audio characteristics
            duration_ms = len(audio)
            sample_rate = audio.frame_rate
            channels = audio.channels
            sample_width = audio.sample_width
            
            # Calculate audio levels
            db_fs = audio.dBFS
            max_db = audio.max_possible_amplitude
            
            # Analyze frequency content (basic)
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
            
            # Calculate RMS (Root Mean Square) for volume analysis
            rms = np.sqrt(np.mean(samples**2))
            
            # Detect silence
            silence_threshold = -50  # dB
            silent_parts = audio.detect_silence(
                min_silence_len=1000,  # 1 second
                silence_thresh=silence_threshold
            )
            
            silence_duration = sum(end - start for start, end in silent_parts)
            silence_percentage = (silence_duration / duration_ms) * 100 if duration_ms > 0 else 0
            
            return {
                "duration_ms": duration_ms,
                "duration_seconds": duration_ms / 1000.0,
                "sample_rate": sample_rate,
                "channels": channels,
                "sample_width": sample_width,
                "db_fs": db_fs,
                "max_db": max_db,
                "rms": float(rms),
                "silence_percentage": silence_percentage,
                "silent_parts_count": len(silent_parts),
                "quality_score": self._calculate_audio_quality_score(
                    db_fs, silence_percentage, sample_rate, duration_ms
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze audio quality: {e}")
            return {
                "error": str(e),
                "quality_score": 0.0
            }
    
    def _calculate_audio_quality_score(
        self,
        db_fs: float,
        silence_percentage: float,
        sample_rate: int,
        duration_ms: int
    ) -> float:
        """Calculate overall audio quality score"""
        score = 0.5  # Base score
        
        # Volume level scoring (optimal range: -20 to -10 dB)
        if -20 <= db_fs <= -10:
            score += 0.2
        elif -30 <= db_fs <= -5:
            score += 0.1
        elif db_fs < -40:
            score -= 0.2  # Too quiet
        elif db_fs > 0:
            score -= 0.2  # Too loud
        
        # Silence percentage scoring (optimal: 10-30%)
        if 10 <= silence_percentage <= 30:
            score += 0.2
        elif silence_percentage < 5:
            score -= 0.1  # Too much continuous speech
        elif silence_percentage > 50:
            score -= 0.2  # Too much silence
        
        # Sample rate scoring (higher is better)
        if sample_rate >= 44100:
            score += 0.1
        elif sample_rate >= 22050:
            score += 0.05
        elif sample_rate < 16000:
            score -= 0.1
        
        # Duration scoring (optimal: 1-30 seconds)
        duration_seconds = duration_ms / 1000.0
        if 1 <= duration_seconds <= 30:
            score += 0.1
        elif duration_seconds < 0.5:
            score -= 0.2  # Too short
        elif duration_seconds > 60:
            score -= 0.1  # Too long
        
        return max(0.0, min(1.0, score))
    
    async def extract_audio_features(self, audio_data: str, format: str = "wav") -> Dict[str, Any]:
        """Extract advanced audio features for analysis"""
        try:
            # Decode and convert audio
            audio_bytes = base64.b64decode(audio_data)
            if format.lower() != "wav":
                audio_bytes = await self._convert_audio_format(audio_bytes, format, "wav")
            
            # Load audio
            audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
            samples = np.array(audio.get_array_of_samples())
            
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                # Use left channel for analysis
                samples = samples[:, 0]
            
            # Basic statistical features
            mean_amplitude = float(np.mean(samples))
            std_amplitude = float(np.std(samples))
            max_amplitude = float(np.max(samples))
            min_amplitude = float(np.min(samples))
            
            # Zero crossing rate (measure of frequency content)
            zero_crossings = np.sum(np.diff(np.sign(samples)) != 0)
            zero_crossing_rate = zero_crossings / len(samples) if len(samples) > 0 else 0
            
            # Spectral features (basic)
            fft = np.fft.fft(samples)
            magnitude_spectrum = np.abs(fft)
            
            # Dominant frequency
            freqs = np.fft.fftfreq(len(samples), 1/audio.frame_rate)
            dominant_freq_idx = np.argmax(magnitude_spectrum[:len(magnitude_spectrum)//2])
            dominant_frequency = float(freqs[dominant_freq_idx])
            
            # Spectral centroid (center of mass of spectrum)
            spectral_centroid = float(np.sum(freqs[:len(freqs)//2] * magnitude_spectrum[:len(magnitude_spectrum)//2]) / 
                                    np.sum(magnitude_spectrum[:len(magnitude_spectrum)//2]))
            
            return {
                "mean_amplitude": mean_amplitude,
                "std_amplitude": std_amplitude,
                "max_amplitude": max_amplitude,
                "min_amplitude": min_amplitude,
                "zero_crossing_rate": zero_crossing_rate,
                "dominant_frequency": dominant_frequency,
                "spectral_centroid": spectral_centroid,
                "dynamic_range": max_amplitude - min_amplitude,
                "crest_factor": max_amplitude / (np.sqrt(np.mean(samples**2)) + 1e-10)
            }
            
        except Exception as e:
            logger.error(f"Failed to extract audio features: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> bool:
        """Check if voice processor is healthy"""
        return self.initialized and self.recognizer is not None