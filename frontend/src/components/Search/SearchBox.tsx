import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MagnifyingGlassIcon,
  MicrophoneIcon,
  PhotoIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';

// Components
import VoiceSearch from './VoiceSearch';
import ImageSearch from './ImageSearch';

interface SearchBoxProps {
  initialQuery?: string;
  onSearch: (query: string) => void;
  placeholder?: string;
  showFilters?: boolean;
  className?: string;
}

const SearchBox: React.FC<SearchBoxProps> = ({
  initialQuery = '',
  onSearch,
  placeholder = 'Search anything...',
  showFilters = true,
  className = '',
}) => {
  const [query, setQuery] = useState(initialQuery);
  const [isFocused, setIsFocused] = useState(false);
  const [showVoiceSearch, setShowVoiceSearch] = useState(false);
  const [showImageSearch, setShowImageSearch] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  
  const inputRef = useRef<HTMLInputElement>(null);
  const suggestionsRef = useRef<HTMLDivElement>(null);

  // Mock suggestions - in a real app, these would come from an API
  const mockSuggestions = [
    'artificial intelligence',
    'machine learning',
    'deep learning',
    'neural networks',
    'data science',
    'python programming',
    'web development',
    'mobile apps',
  ];

  useEffect(() => {
    if (query.trim()) {
      const filtered = mockSuggestions.filter(suggestion =>
        suggestion.toLowerCase().includes(query.toLowerCase())
      );
      setSuggestions(filtered.slice(0, 5));
    } else {
      setSuggestions([]);
    }
  }, [query]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (suggestionsRef.current && !suggestionsRef.current.contains(event.target as Node)) {
        setSuggestions([]);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
      setSuggestions([]);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
    onSearch(suggestion);
    setSuggestions([]);
    inputRef.current?.blur();
  };

  const handleVoiceResult = (transcript: string) => {
    setQuery(transcript);
    setShowVoiceSearch(false);
    if (transcript.trim()) {
      onSearch(transcript);
    }
  };

  const handleImageUpload = (file: File) => {
    setShowImageSearch(false);
    // TODO: Implement image search
    console.log('Image uploaded:', file.name);
  };

  const clearQuery = () => {
    setQuery('');
    setSuggestions([]);
    inputRef.current?.focus();
  };

  return (
    <div className={`relative ${className}`}>
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative">
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder={placeholder}
            className="w-full pl-12 pr-20 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
          />
          
          {/* Search Icon */}
          <MagnifyingGlassIcon className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          
          {/* Clear Button */}
          {query && (
            <button
              type="button"
              onClick={clearQuery}
              className="absolute right-16 top-1/2 transform -translate-y-1/2 p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <XMarkIcon className="h-4 w-4" />
            </button>
          )}
          
          {/* Action Buttons */}
          <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex items-center space-x-1">
            {/* Voice Search Button */}
            <button
              type="button"
              onClick={() => setShowVoiceSearch(true)}
              className="p-2 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-200"
              title="Voice Search"
            >
              <MicrophoneIcon className="h-5 w-5" />
            </button>
            
            {/* Image Search Button */}
            <button
              type="button"
              onClick={() => setShowImageSearch(true)}
              className="p-2 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-200"
              title="Image Search"
            >
              <PhotoIcon className="h-5 w-5" />
            </button>
          </div>
        </div>
      </form>

      {/* Search Suggestions */}
      <AnimatePresence>
        {suggestions.length > 0 && isFocused && (
          <motion.div
            ref={suggestionsRef}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute top-full left-0 right-0 mt-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg z-50"
          >
            {suggestions.map((suggestion, index) => (
              <button
                key={index}
                onClick={() => handleSuggestionClick(suggestion)}
                className="w-full px-4 py-3 text-left text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors duration-200 first:rounded-t-lg last:rounded-b-lg"
              >
                <div className="flex items-center space-x-3">
                  <MagnifyingGlassIcon className="h-4 w-4 text-gray-400" />
                  <span>{suggestion}</span>
                </div>
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Voice Search Modal */}
      <AnimatePresence>
        {showVoiceSearch && (
          <VoiceSearch
            onResult={handleVoiceResult}
            onClose={() => setShowVoiceSearch(false)}
          />
        )}
      </AnimatePresence>

      {/* Image Search Modal */}
      <AnimatePresence>
        {showImageSearch && (
          <ImageSearch
            onUpload={handleImageUpload}
            onClose={() => setShowImageSearch(false)}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default SearchBox;