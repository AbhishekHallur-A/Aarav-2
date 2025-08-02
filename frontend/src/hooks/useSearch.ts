import { useState, useCallback } from 'react';
import axios from 'axios';

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

interface SearchParams {
  query: string;
  type?: string;
  dateRange?: string;
  language?: string;
  region?: string;
  page?: number;
  limit?: number;
}

interface SearchResponse {
  results: SearchResult[];
  total: number;
  page: number;
  limit: number;
  query: string;
}

export const useSearch = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastQuery, setLastQuery] = useState<string>('');

  const performSearch = useCallback(async (params: SearchParams): Promise<SearchResponse | null> => {
    setIsLoading(true);
    setError(null);
    setLastQuery(params.query);

    try {
      // In a real application, this would be an actual API call
      // const response = await axios.get('/api/search', { params });
      
      // For now, we'll simulate an API call with mock data
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate network delay
      
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
        {
          id: '4',
          title: 'Machine Learning in Search Engines',
          url: 'https://example.com/ml-search',
          description: 'Discover how machine learning algorithms are transforming search engines and improving the accuracy of search results.',
          type: 'web',
          timestamp: '2024-01-05',
          relevance: 0.78,
          tags: ['machine learning', 'search', 'algorithms'],
        },
        {
          id: '5',
          title: 'Natural Language Processing for Search',
          url: 'https://example.com/nlp-search',
          description: 'Understanding how natural language processing enhances search capabilities and improves user experience.',
          type: 'document',
          timestamp: '2024-01-03',
          relevance: 0.75,
          tags: ['NLP', 'search', 'AI'],
        },
      ];

      // Filter results based on search parameters
      let filteredResults = mockResults;
      
      if (params.type && params.type !== 'all') {
        filteredResults = filteredResults.filter(result => result.type === params.type);
      }

      // Apply pagination
      const page = params.page || 1;
      const limit = params.limit || 10;
      const startIndex = (page - 1) * limit;
      const endIndex = startIndex + limit;
      const paginatedResults = filteredResults.slice(startIndex, endIndex);

      const response: SearchResponse = {
        results: paginatedResults,
        total: filteredResults.length,
        page,
        limit,
        query: params.query,
      };

      return response;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred during search';
      setError(errorMessage);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    performSearch,
    isLoading,
    error,
    lastQuery,
    clearError,
  };
};