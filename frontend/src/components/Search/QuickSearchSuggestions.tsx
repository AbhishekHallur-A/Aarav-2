import React from 'react';
import { motion } from 'framer-motion';
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';

interface QuickSearchSuggestionsProps {
  onSuggestionClick: (suggestion: string) => void;
}

const QuickSearchSuggestions: React.FC<QuickSearchSuggestionsProps> = ({
  onSuggestionClick,
}) => {
  const suggestions = [
    'artificial intelligence',
    'machine learning',
    'web development',
    'data science',
    'python programming',
    'react tutorial',
    'blockchain technology',
    'cloud computing',
    'cybersecurity',
    'mobile app development',
  ];

  return (
    <div className="mt-8">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        Popular Searches
      </h3>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
        {suggestions.map((suggestion, index) => (
          <motion.button
            key={suggestion}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            onClick={() => onSuggestionClick(suggestion)}
            className="flex items-center space-x-2 px-4 py-3 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-md hover:border-blue-300 dark:hover:border-blue-600 transition-all duration-200 group"
          >
            <MagnifyingGlassIcon className="h-4 w-4 text-gray-400 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors duration-200" />
            <span className="text-sm text-gray-700 dark:text-gray-300 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors duration-200 truncate">
              {suggestion}
            </span>
          </motion.button>
        ))}
      </div>
    </div>
  );
};

export default QuickSearchSuggestions;