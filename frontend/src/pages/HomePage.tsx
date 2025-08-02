import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MagnifyingGlassIcon,
  MicrophoneIcon,
  PhotoIcon,
  SparklesIcon,
  GlobeAltIcon,
  LightBulbIcon,
  ShieldCheckIcon,
} from '@heroicons/react/24/outline';

// Components
import SearchBox from '../components/Search/SearchBox';
import VoiceSearch from '../components/Search/VoiceSearch';
import ImageSearch from '../components/Search/ImageSearch';
import QuickSearchSuggestions from '../components/Search/QuickSearchSuggestions';
import TrendingTopics from '../components/Search/TrendingTopics';
import SearchStats from '../components/Search/SearchStats';

// Hooks
import { useSearch } from '../hooks/useSearch';
import { useVoiceSearch } from '../hooks/useVoiceSearch';

const HomePage: React.FC = () => {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  const [showVoiceSearch, setShowVoiceSearch] = useState(false);
  const [showImageSearch, setShowImageSearch] = useState(false);
  const [isSearchFocused, setIsSearchFocused] = useState(false);
  
  const { performSearch, isLoading } = useSearch();
  const { isListening, startListening, stopListening } = useVoiceSearch();
  
  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Focus search input on page load
    searchInputRef.current?.focus();
  }, []);

  const handleSearch = async (query: string) => {
    if (!query.trim()) return;
    
    // Navigate to search results page
    navigate(`/search?q=${encodeURIComponent(query)}`);
  };

  const handleVoiceResult = (transcript: string) => {
    setSearchQuery(transcript);
    setShowVoiceSearch(false);
    if (transcript.trim()) {
      handleSearch(transcript);
    }
  };

  const handleImageUpload = (file: File) => {
    // Handle image search
    setShowImageSearch(false);
    // TODO: Implement image search
    console.log('Image uploaded:', file.name);
  };

  const features = [
    {
      icon: SparklesIcon,
      title: 'AI-Powered Search',
      description: 'Advanced machine learning algorithms understand your intent and deliver precise results.',
    },
    {
      icon: GlobeAltIcon,
      title: 'Multilingual Support',
      description: 'Search in any language with automatic translation and language detection.',
    },
    {
      icon: LightBulbIcon,
      title: 'Smart Suggestions',
      description: 'Intelligent autocomplete and query expansion help you find what you need faster.',
    },
    {
      icon: ShieldCheckIcon,
      title: 'Privacy First',
      description: 'Your search data is protected with end-to-end encryption and minimal tracking.',
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-purple-900">
      {/* Main Search Section */}
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-12">
          {/* Logo and Brand */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="mb-8"
          >
            <h1 className="text-6xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
              AstraFind
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
              The next-generation AI-powered search engine that understands your needs
            </p>
          </motion.div>

          {/* Search Interface */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="max-w-4xl mx-auto mb-8"
          >
            <div className="relative">
              <SearchBox
                ref={searchInputRef}
                value={searchQuery}
                onChange={setSearchQuery}
                onSubmit={handleSearch}
                onFocus={() => setIsSearchFocused(true)}
                onBlur={() => setIsSearchFocused(false)}
                placeholder="Search anything..."
                className="w-full px-6 py-4 text-lg rounded-full border-2 border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-800 shadow-lg focus:border-blue-500 focus:shadow-xl transition-all duration-200"
              />
              
              {/* Search Actions */}
              <div className="absolute right-4 top-1/2 transform -translate-y-1/2 flex items-center space-x-2">
                {/* Voice Search */}
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => setShowVoiceSearch(true)}
                  className={`p-2 rounded-full transition-colors ${
                    isListening
                      ? 'bg-red-100 text-red-600 dark:bg-red-900 dark:text-red-400'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600'
                  }`}
                  title="Voice Search"
                >
                  <MicrophoneIcon className="w-5 h-5" />
                </motion.button>

                {/* Image Search */}
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => setShowImageSearch(true)}
                  className="p-2 rounded-full bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600 transition-colors"
                  title="Image Search"
                >
                  <PhotoIcon className="w-5 h-5" />
                </motion.button>

                {/* Search Button */}
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => handleSearch(searchQuery)}
                  disabled={!searchQuery.trim() || isLoading}
                  className="p-2 rounded-full bg-blue-600 text-white hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                  title="Search"
                >
                  <MagnifyingGlassIcon className="w-5 h-5" />
                </motion.button>
              </div>
            </div>

            {/* Quick Suggestions */}
            <AnimatePresence>
              {isSearchFocused && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="mt-4"
                >
                  <QuickSearchSuggestions
                    query={searchQuery}
                    onSelect={(suggestion) => {
                      setSearchQuery(suggestion);
                      handleSearch(suggestion);
                    }}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Search Stats */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <SearchStats />
          </motion.div>
        </div>

        {/* Trending Topics */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="mb-16"
        >
          <TrendingTopics onTopicSelect={handleSearch} />
        </motion.div>

        {/* Features Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16"
        >
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.8 + index * 0.1 }}
              className="text-center p-6 rounded-xl bg-white dark:bg-gray-800 shadow-lg hover:shadow-xl transition-shadow"
            >
              <div className="inline-flex items-center justify-center w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-lg mb-4">
                <feature.icon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                {feature.title}
              </h3>
              <p className="text-gray-600 dark:text-gray-300 text-sm">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </motion.div>

        {/* Call to Action */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.2 }}
          className="text-center"
        >
          <div className="inline-flex items-center space-x-4 bg-white dark:bg-gray-800 rounded-full px-8 py-4 shadow-lg">
            <span className="text-gray-600 dark:text-gray-300">
              Ready to explore the future of search?
            </span>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => searchInputRef.current?.focus()}
              className="px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-full font-medium hover:shadow-lg transition-shadow"
            >
              Start Searching
            </motion.button>
          </div>
        </motion.div>
      </div>

      {/* Voice Search Modal */}
      <AnimatePresence>
        {showVoiceSearch && (
          <VoiceSearch
            isOpen={showVoiceSearch}
            onClose={() => setShowVoiceSearch(false)}
            onResult={handleVoiceResult}
          />
        )}
      </AnimatePresence>

      {/* Image Search Modal */}
      <AnimatePresence>
        {showImageSearch && (
          <ImageSearch
            isOpen={showImageSearch}
            onClose={() => setShowImageSearch(false)}
            onImageUpload={handleImageUpload}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default HomePage;