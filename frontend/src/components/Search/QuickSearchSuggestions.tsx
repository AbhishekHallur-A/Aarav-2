import React from 'react';

interface QuickSearchSuggestionsProps {
  query: string;
  onSelect: (suggestion: string) => void;
  suggestions?: string[];
}

const QuickSearchSuggestions: React.FC<QuickSearchSuggestionsProps> = ({
  query,
  onSelect,
  suggestions = [
    "artificial intelligence",
    "machine learning",
    "web development",
    "data science",
    "cybersecurity"
  ]
}) => {
  return (
    <div className="mt-6">
      <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
        Quick Search Suggestions
      </h3>
      <div className="flex flex-wrap gap-2">
        {suggestions.map((suggestion, index) => (
          <button
            key={index}
            onClick={() => onSelect(suggestion)}
            className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 rounded-full transition-colors"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
};

export default QuickSearchSuggestions;