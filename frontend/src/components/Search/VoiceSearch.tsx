import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MicrophoneIcon,
  XMarkIcon,
  SpeakerWaveIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';

// Types
interface VoiceSearchProps {
  isOpen: boolean;
  onClose: () => void;
  onResult: (transcript: string) => void;
  language?: string;
}

interface SpeechRecognitionEvent {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

interface SpeechRecognitionResult {
  transcript: string;
  confidence: number;
}

interface SpeechRecognitionResultList {
  [index: number]: SpeechRecognitionResult & { isFinal: boolean };
  length: number;
}

// Extend Window interface for Speech Recognition
declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}

const VoiceSearch: React.FC<VoiceSearchProps> = ({
  isOpen,
  onClose,
  onResult,
  language = 'en-US',
}) => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [isSupported, setIsSupported] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0);
  const [volume, setVolume] = useState(0);

  const recognitionRef = useRef<any>(null);
  const animationRef = useRef<number>();
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const microphoneRef = useRef<MediaStreamAudioSourceNode | null>(null);

  useEffect(() => {
    // Check if speech recognition is supported
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      setIsSupported(false);
      setError('Speech recognition is not supported in this browser. Please try Chrome or Edge.');
      return;
    }

    // Initialize speech recognition
    recognitionRef.current = new SpeechRecognition();
    recognitionRef.current.continuous = true;
    recognitionRef.current.interimResults = true;
    recognitionRef.current.lang = language;

    // Event handlers
    recognitionRef.current.onstart = () => {
      setIsListening(true);
      setError(null);
      setupAudioAnalysis();
    };

    recognitionRef.current.onresult = (event: any) => {
      let finalTranscript = '';
      let interimTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          finalTranscript += result[0].transcript;
          setConfidence(result[0].confidence);
        } else {
          interimTranscript += result[0].transcript;
        }
      }

      setTranscript(finalTranscript);
      setInterimTranscript(interimTranscript);
    };

    recognitionRef.current.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error);
      setError(`Speech recognition error: ${event.error}`);
      setIsListening(false);
      stopAudioAnalysis();
    };

    recognitionRef.current.onend = () => {
      setIsListening(false);
      stopAudioAnalysis();
    };

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      stopAudioAnalysis();
    };
  }, [language]);

  useEffect(() => {
    if (isOpen && isSupported) {
      startListening();
    } else {
      stopListening();
    }
  }, [isOpen, isSupported]);

  const setupAudioAnalysis = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new AudioContext();
      analyserRef.current = audioContextRef.current.createAnalyser();
      microphoneRef.current = audioContextRef.current.createMediaStreamSource(stream);
      
      microphoneRef.current.connect(analyserRef.current);
      analyserRef.current.fftSize = 256;

      const bufferLength = analyserRef.current.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const updateVolume = () => {
        if (analyserRef.current) {
          analyserRef.current.getByteFrequencyData(dataArray);
          const avg = dataArray.reduce((sum, value) => sum + value, 0) / bufferLength;
          setVolume(avg / 255);
          
          if (isListening) {
            animationRef.current = requestAnimationFrame(updateVolume);
          }
        }
      };

      updateVolume();
    } catch (error) {
      console.error('Error setting up audio analysis:', error);
    }
  };

  const stopAudioAnalysis = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    
    if (microphoneRef.current) {
      microphoneRef.current.disconnect();
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
  };

  const startListening = () => {
    if (recognitionRef.current && isSupported) {
      setTranscript('');
      setInterimTranscript('');
      setError(null);
      recognitionRef.current.start();
    }
  };

  const stopListening = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
  };

  const handleSubmit = () => {
    const finalTranscript = transcript.trim();
    if (finalTranscript) {
      onResult(finalTranscript);
    }
    onClose();
  };

  const handleClose = () => {
    stopListening();
    onClose();
  };

  if (!isOpen) return null;

  const displayText = transcript || interimTranscript || '';
  const isRecording = isListening && !error;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black bg-opacity-50 backdrop-blur-sm"
        onClick={handleClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          className="relative w-full max-w-md p-8 bg-white dark:bg-gray-800 rounded-2xl shadow-2xl"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Close Button */}
          <button
            onClick={handleClose}
            className="absolute top-4 right-4 p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
          >
            <XMarkIcon className="w-6 h-6" />
          </button>

          {/* Header */}
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              Voice Search
            </h2>
            <p className="text-gray-600 dark:text-gray-300">
              {isRecording ? 'Listening...' : 'Click the microphone to start'}
            </p>
          </div>

          {/* Error Display */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg"
            >
              <div className="flex items-center">
                <ExclamationTriangleIcon className="w-5 h-5 text-red-500 mr-2" />
                <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
              </div>
            </motion.div>
          )}

          {/* Microphone Animation */}
          <div className="flex justify-center mb-8">
            <motion.div
              className="relative"
              animate={{
                scale: isRecording ? 1 + volume * 0.3 : 1,
              }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            >
              {/* Pulse Rings */}
              {isRecording && (
                <>
                  <motion.div
                    className="absolute inset-0 bg-blue-500 rounded-full opacity-20"
                    animate={{
                      scale: [1, 1.5, 2],
                      opacity: [0.3, 0.1, 0],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: 'easeOut',
                    }}
                  />
                  <motion.div
                    className="absolute inset-0 bg-blue-500 rounded-full opacity-20"
                    animate={{
                      scale: [1, 1.5, 2],
                      opacity: [0.3, 0.1, 0],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: 'easeOut',
                      delay: 0.5,
                    }}
                  />
                </>
              )}

              {/* Microphone Button */}
              <motion.button
                onClick={isRecording ? stopListening : startListening}
                disabled={!isSupported}
                className={`
                  relative w-20 h-20 rounded-full flex items-center justify-center
                  transition-all duration-200 shadow-lg
                  ${isRecording
                    ? 'bg-red-500 hover:bg-red-600 text-white'
                    : 'bg-blue-500 hover:bg-blue-600 text-white'
                  }
                  ${!isSupported ? 'opacity-50 cursor-not-allowed' : 'hover:shadow-xl'}
                `}
                whileTap={{ scale: 0.95 }}
              >
                <MicrophoneIcon className="w-8 h-8" />
              </motion.button>
            </motion.div>
          </div>

          {/* Transcript Display */}
          <div className="mb-6">
            <div className="min-h-[100px] p-4 bg-gray-50 dark:bg-gray-700 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-600">
              {displayText ? (
                <div className="space-y-2">
                  {transcript && (
                    <p className="text-gray-900 dark:text-white font-medium">
                      {transcript}
                    </p>
                  )}
                  {interimTranscript && (
                    <p className="text-gray-500 dark:text-gray-400 italic">
                      {interimTranscript}
                    </p>
                  )}
                  {confidence > 0 && (
                    <div className="flex items-center mt-2">
                      <span className="text-xs text-gray-500 dark:text-gray-400 mr-2">
                        Confidence:
                      </span>
                      <div className="flex-1 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                        <motion.div
                          className="bg-green-500 h-2 rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${confidence * 100}%` }}
                          transition={{ duration: 0.3 }}
                        />
                      </div>
                      <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">
                        {Math.round(confidence * 100)}%
                      </span>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex items-center justify-center h-full text-gray-400 dark:text-gray-500">
                  <SpeakerWaveIcon className="w-6 h-6 mr-2" />
                  <span>Your speech will appear here...</span>
                </div>
              )}
            </div>
          </div>

          {/* Volume Indicator */}
          {isRecording && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mb-6"
            >
              <div className="flex items-center justify-center space-x-1">
                {Array.from({ length: 20 }, (_, i) => (
                  <motion.div
                    key={i}
                    className={`w-1 bg-blue-500 rounded-full ${
                      volume * 20 > i ? 'opacity-100' : 'opacity-30'
                    }`}
                    animate={{
                      height: volume * 20 > i ? Math.random() * 20 + 10 : 4,
                    }}
                    transition={{ duration: 0.1 }}
                  />
                ))}
              </div>
            </motion.div>
          )}

          {/* Action Buttons */}
          <div className="flex space-x-4">
            <button
              onClick={handleClose}
              className="flex-1 px-4 py-2 text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSubmit}
              disabled={!transcript.trim()}
              className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Search
            </button>
          </div>

          {/* Tips */}
          <div className="mt-6 text-center">
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Tip: Speak clearly and wait for the transcript to appear
            </p>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default VoiceSearch;