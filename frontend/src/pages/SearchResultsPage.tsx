import React, { useState, useEffect } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  MagnifyingGlassIcon,
  FunnelIcon,
  AdjustmentsHorizontalIcon,
  ClockIcon,
  GlobeAltIcon,
  DocumentTextIcon,
  PhotoIcon,
  VideoCameraIcon,
} from '@heroicons/react/24/outline';

// Components
import SearchBox from '../components/Search/SearchBox';
import SearchResultCard from '../components/Search/SearchResultCard';
import FilterPanel from '../components/Search/FilterPanel';
import LoadingSpinner from '../components/UI/LoadingSpinner';

// Hooks
import { useSearch } from '../hooks/useSearch';

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

const SearchResultsPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const query = searchParams.get('q') || '';
  
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({
    type: 'all',
    dateRange: 'all',
    language: 'all',
  });

  const { performSearch } = useSearch();

  useEffect(() => {
    if (query) {
      handleSearch(query);
    }
  }, [query]);

  const handleSearch = async (searchQuery: string) => {
    setIsLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock results
      const mockResults: SearchResult[] = [
        {
          id: '1',
          title: 'AstraFind - AI-Powered Search Engine',
          url: 'https://astrafind.com',
          description: 'AstraFind is an advanced AI-powered search engine that uses machine learning to deliver precise and relevant search results. Our intelligent algorithms understand user intent and provide personalized search experiences.',
          type: 'web',
          timestamp: '2024-01-15',
          relevance: 0.95,
          tags: ['AI', 'search', 'machine learning'],
        },
        {
          id: '2',
          title: 'Understanding AI Search Algorithms',
          url: 'https://example.com/ai-search',
          description: 'Learn how artificial intelligence is revolutionizing search technology. This comprehensive guide covers the latest developments in AI-powered search algorithms and their applications.',
          type: 'document',
          timestamp: '2024-01-10',
          relevance: 0.88,
          tags: ['AI', 'algorithms', 'technology'],
        },
        {
          id: '3',
          title: 'The Future of Search Technology',
          url: 'https://example.com/future-search',
          description: 'Explore the emerging trends in search technology, from voice search to visual search, and how AI is shaping the future of information discovery.',
          type: 'web',
          timestamp: '2024-01-08',
          relevance: 0.82,
          tags: ['future', 'technology', 'trends'],
        },
      ];
      
      setResults(mockResults);
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFilterChange = (newFilters: typeof filters) => {
    setFilters(newFilters);
    // Apply filters to results
    // This would typically involve refetching with new parameters
  };

  const resultTypes = [
    { id: 'all', name: 'All', icon: GlobeAltIcon, count: results.length },
    { id: 'web', name: 'Web', icon: GlobeAltIcon, count: results.filter(r => r.type === 'web').length },
    { id: 'image', name: 'Images', icon: PhotoIcon, count: results.filter(r => r.type === 'image').length },
    { id: 'video', name: 'Videos', icon: VideoCameraIcon, count: results.filter(r => r.type === 'video').length },
    { id: 'document', name: 'Documents', icon: DocumentTextIcon, count: results.filter(r => r.type === 'document').length },
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Search Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 sticky top-16 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center space-x-4">
            <div className="flex-1">
              <SearchBox
                initialQuery={query}
                onSearch={handleSearch}
                placeholder="Search anything..."
                showFilters={false}
              />
            </div>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors duration-200"
            >
              <FunnelIcon className="h-5 w-5 text-gray-700 dark:text-gray-300" />
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Filters Sidebar */}
          {showFilters && (
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="lg:w-64"
            >
              <FilterPanel
                filters={filters}
                onFilterChange={handleFilterChange}
                resultTypes={resultTypes}
              />
            </motion.div>
          )}

          {/* Main Content */}
          <div className="flex-1">
            {/* Results Header */}
            <div className="mb-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h1 className="text-2xl font-semibold text-gray-900 dark:text-white">
                    Search Results
                  </h1>
                  <p className="text-gray-600 dark:text-gray-400">
                    {isLoading ? 'Searching...' : `${results.length} results for "${query}"`}
                  </p>
                </div>
                <div className="flex items-center space-x-2 text-sm text-gray-500 dark:text-gray-400">
                  <ClockIcon className="h-4 w-4" />
                  <span>0.45 seconds</span>
                </div>
              </div>

              {/* Result Type Tabs */}
              <div className="flex space-x-1 bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
                {resultTypes.map((type) => (
                  <button
                    key={type.id}
                    className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200 ${
                      filters.type === type.id
                        ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                        : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                    }`}
                    onClick={() => handleFilterChange({ ...filters, type: type.id })}
                  >
                    <type.icon className="h-4 w-4" />
                    <span>{type.name}</span>
                    <span className="text-xs bg-gray-200 dark:bg-gray-500 px-1.5 py-0.5 rounded-full">
                      {type.count}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Results */}
            {isLoading ? (
              <div className="flex justify-center py-12">
                <LoadingSpinner size="lg" />
              </div>
            ) : results.length > 0 ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-6"
              >
                {results.map((result, index) => (
                  <motion.div
                    key={result.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <SearchResultCard result={result} />
                  </motion.div>
                ))}
              </motion.div>
            ) : (
              <div className="text-center py-12">
                <MagnifyingGlassIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                  No results found
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  Try adjusting your search terms or filters to find what you're looking for.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SearchResultsPage;