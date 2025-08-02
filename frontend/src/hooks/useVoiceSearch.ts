import { useState, useCallback, useEffect } from 'react';

interface VoiceSearchState {
  isListening: boolean;
  transcript: string;
  isSupported: boolean;
  error: string | null;
}

export const useVoiceSearch = () => {
  const [state, setState] = useState<VoiceSearchState>({
    isListening: false,
    transcript: '',
    isSupported: false,
    error: null,
  });

  const [recognition, setRecognition] = useState<SpeechRecognition | null>(null);

  useEffect(() => {
    // Check if browser supports Speech Recognition
    const SpeechRecognition = window.SpeechRecognition || (window as any).webkitSpeechRecognition;
    
    if (SpeechRecognition) {
      const recognitionInstance = new SpeechRecognition();
      
      // Configure recognition settings
      recognitionInstance.continuous = true;
      recognitionInstance.interimResults = true;
      recognitionInstance.lang = 'en-US';
      
      // Set up event handlers
      recognitionInstance.onstart = () => {
        setState(prev => ({
          ...prev,
          isListening: true,
          error: null,
        }));
      };

      recognitionInstance.onresult = (event: SpeechRecognitionEvent) => {
        let finalTranscript = '';
        let interimTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
          } else {
            interimTranscript += transcript;
          }
        }

        setState(prev => ({
          ...prev,
          transcript: finalTranscript || interimTranscript,
        }));
      };

      recognitionInstance.onerror = (event: SpeechRecognitionErrorEvent) => {
        let errorMessage = 'Voice recognition error occurred';
        
        switch (event.error) {
          case 'no-speech':
            errorMessage = 'No speech was detected. Please try again.';
            break;
          case 'audio-capture':
            errorMessage = 'Audio capture failed. Please check your microphone.';
            break;
          case 'not-allowed':
            errorMessage = 'Microphone access was denied. Please allow microphone access.';
            break;
          case 'network':
            errorMessage = 'Network error occurred. Please check your connection.';
            break;
          case 'service-not-allowed':
            errorMessage = 'Speech recognition service is not allowed.';
            break;
          case 'bad-grammar':
            errorMessage = 'Speech recognition grammar error.';
            break;
          case 'language-not-supported':
            errorMessage = 'Language not supported.';
            break;
          default:
            errorMessage = `Voice recognition error: ${event.error}`;
        }

        setState(prev => ({
          ...prev,
          isListening: false,
          error: errorMessage,
        }));
      };

      recognitionInstance.onend = () => {
        setState(prev => ({
          ...prev,
          isListening: false,
        }));
      };

      setRecognition(recognitionInstance);
      setState(prev => ({ ...prev, isSupported: true }));
    } else {
      setState(prev => ({
        ...prev,
        isSupported: false,
        error: 'Speech recognition is not supported in this browser.',
      }));
    }
  }, []);

  const startListening = useCallback(() => {
    if (recognition && state.isSupported) {
      try {
        setState(prev => ({ ...prev, transcript: '', error: null }));
        recognition.start();
      } catch (error) {
        setState(prev => ({
          ...prev,
          error: 'Failed to start voice recognition.',
        }));
      }
    }
  }, [recognition, state.isSupported]);

  const stopListening = useCallback(() => {
    if (recognition && state.isListening) {
      try {
        recognition.stop();
      } catch (error) {
        setState(prev => ({
          ...prev,
          error: 'Failed to stop voice recognition.',
        }));
      }
    }
  }, [recognition, state.isListening]);

  const clearTranscript = useCallback(() => {
    setState(prev => ({ ...prev, transcript: '' }));
  }, []);

  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  return {
    // State
    isListening: state.isListening,
    transcript: state.transcript,
    isSupported: state.isSupported,
    error: state.error,
    
    // Actions
    startListening,
    stopListening,
    clearTranscript,
    clearError,
  };
};