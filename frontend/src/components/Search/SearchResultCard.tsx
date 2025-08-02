import React from 'react';
import { motion } from 'framer-motion';
import {
  GlobeAltIcon,
  DocumentTextIcon,
  PhotoIcon,
  VideoCameraIcon,
  CalendarIcon,
  TagIcon,
} from '@heroicons/react/24/outline';

interface SearchResult {
  id: string;
  title: string;
  url: string;
  description: string;
  type: 'web' | 'image' | 'video' | 'document';
  timestamp: string;
  relevance: number;
  tags: string[];
}

interface SearchResultCardProps {
  result: SearchResult;
}

const SearchResultCard: React.FC<SearchResultCardProps> = ({ result }) => {
  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'image':
        return PhotoIcon;
      case 'video':
        return VideoCameraIcon;
      case 'document':
        return DocumentTextIcon;
      default:
        return GlobeAltIcon;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'image':
        return 'text-green-600';
      case 'video':
        return 'text-purple-600';
      case 'document':
        return 'text-orange-600';
      default:
        return 'text-blue-600';
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - date.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 1) return '1 day ago';
    if (diffDays < 7) return `${diffDays} days ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
    if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
    return `${Math.floor(diffDays / 365)} years ago`;
  };

  const TypeIcon = getTypeIcon(result.type);
  const typeColor = getTypeColor(result.type);

  return (
    <motion.div
      whileHover={{ y: -2 }}
      className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 hover:shadow-md transition-shadow duration-200"
    >
      <div className="flex items-start space-x-4">
        {/* Type Icon */}
        <div className={`flex-shrink-0 ${typeColor}`}>
          <TypeIcon className="h-6 w-6" />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Title and URL */}
          <div className="mb-2">
            <a
              href={result.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-lg font-semibold text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 transition-colors duration-200 line-clamp-2"
            >
              {result.title}
            </a>
            <p className="text-sm text-green-700 dark:text-green-400 mt-1 truncate">
              {result.url}
            </p>
          </div>

          {/* Description */}
          <p className="text-gray-600 dark:text-gray-300 text-sm leading-relaxed mb-3 line-clamp-3">
            {result.description}
          </p>

          {/* Metadata */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-400">
              {/* Date */}
              <div className="flex items-center space-x-1">
                <CalendarIcon className="h-3 w-3" />
                <span>{formatDate(result.timestamp)}</span>
              </div>

              {/* Relevance Score */}
              <div className="flex items-center space-x-1">
                <span>Relevance: {Math.round(result.relevance * 100)}%</span>
              </div>

              {/* Type Badge */}
              <span className={`px-2 py-1 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-700 ${typeColor}`}>
                {result.type.charAt(0).toUpperCase() + result.type.slice(1)}
              </span>
            </div>
          </div>

          {/* Tags */}
          {result.tags.length > 0 && (
            <div className="mt-3 flex items-center space-x-2">
              <TagIcon className="h-3 w-3 text-gray-400" />
              <div className="flex flex-wrap gap-1">
                {result.tags.slice(0, 3).map((tag, index) => (
                  <span
                    key={index}
                    className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 text-xs rounded-full"
                  >
                    {tag}
                  </span>
                ))}
                {result.tags.length > 3 && (
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    +{result.tags.length - 3} more
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default SearchResultCard;