import { useState, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import axios from 'axios';
import toast from 'react-hot-toast';

// Types
interface SearchRequest {
  query: string;
  language?: string;
  contentTypeFilter?: string;
  timeFilter?: string;
  safeSearch?: boolean;
  personalized?: boolean;
  maxResults?: number;
  includeSimilar?: boolean;
}

interface SearchResult {
  documentId: string;
  url: string;
  title: string;
  description: string;
  contentSnippet: string;
  contentType: string;
  language: string;
  qualityScore: number;
  finalScore: number;
  crawlTime?: string;
  similarDocuments?: string[];
}

interface SearchResponse {
  query: string;
  totalResults: number;
  searchTimeMs: number;
  results: SearchResult[];
  suggestions?: string[];
  trendingTopics?: string[];
}

interface VoiceSearchRequest {
  audioData: string;
  format?: string;
  language?: string;
}

interface ImageSearchResponse extends SearchResponse {
  analysisResult?: {
    objects: Array<{ name: string; confidence: number }>;
    text?: string;
    tags: string[];
  };
}

// API functions
const searchAPI = {
  textSearch: async (request: SearchRequest): Promise<SearchResponse> => {
    const response = await axios.post('/api/v1/search/text', request);
    return response.data;
  },

  voiceSearch: async (request: VoiceSearchRequest): Promise<SearchResponse> => {
    const response = await axios.post('/api/v1/search/voice', request);
    return response.data;
  },

  imageSearch: async (formData: FormData): Promise<ImageSearchResponse> => {
    const response = await axios.post('/api/v1/search/image', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  getSuggestions: async (query: string): Promise<string[]> => {
    const response = await axios.get(`/api/v1/search/suggestions?query=${encodeURIComponent(query)}`);
    return response.data.suggestions || [];
  },

  getTrendingTopics: async (): Promise<string[]> => {
    const response = await axios.get('/api/v1/search/trending');
    return response.data.trending_topics || [];
  },

  getSearchStats: async () => {
    const response = await axios.get('/api/v1/search/stats');
    return response.data.stats;
  },
};

// Custom hook for search functionality
export const useSearch = () => {
  const [searchHistory, setSearchHistory] = useState<string[]>([]);
  const [recentSearches, setRecentSearches] = useState<SearchResult[]>([]);
  const queryClient = useQueryClient();

  // Text search mutation
  const textSearchMutation = useMutation(searchAPI.textSearch, {
    onSuccess: (data, variables) => {
      // Update search history
      setSearchHistory(prev => [
        variables.query,
        ...prev.filter(q => q !== variables.query)
      ].slice(0, 10));

      // Update recent searches
      setRecentSearches(prev => [
        ...data.results.slice(0, 3),
        ...prev
      ].slice(0, 20));

      // Cache results
      queryClient.setQueryData(['search', variables.query], data);
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Search failed');
    },
  });

  // Voice search mutation
  const voiceSearchMutation = useMutation(searchAPI.voiceSearch, {
    onSuccess: (data) => {
      setRecentSearches(prev => [
        ...data.results.slice(0, 3),
        ...prev
      ].slice(0, 20));
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Voice search failed');
    },
  });

  // Image search mutation
  const imageSearchMutation = useMutation(searchAPI.imageSearch, {
    onSuccess: (data) => {
      setRecentSearches(prev => [
        ...data.results.slice(0, 3),
        ...prev
      ].slice(0, 20));
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Image search failed');
    },
  });

  // Search suggestions query
  const useSuggestions = (query: string, enabled: boolean = true) => {
    return useQuery(
      ['suggestions', query],
      () => searchAPI.getSuggestions(query),
      {
        enabled: enabled && query.length > 1,
        staleTime: 5 * 60 * 1000, // 5 minutes
        cacheTime: 10 * 60 * 1000, // 10 minutes
      }
    );
  };

  // Trending topics query
  const useTrendingTopics = () => {
    return useQuery(
      'trending-topics',
      searchAPI.getTrendingTopics,
      {
        staleTime: 30 * 60 * 1000, // 30 minutes
        cacheTime: 60 * 60 * 1000, // 1 hour
      }
    );
  };

  // Search stats query
  const useSearchStats = () => {
    return useQuery(
      'search-stats',
      searchAPI.getSearchStats,
      {
        staleTime: 60 * 60 * 1000, // 1 hour
        cacheTime: 2 * 60 * 60 * 1000, // 2 hours
      }
    );
  };

  // Perform text search
  const performSearch = useCallback(async (request: SearchRequest) => {
    return textSearchMutation.mutateAsync(request);
  }, [textSearchMutation]);

  // Perform voice search
  const performVoiceSearch = useCallback(async (request: VoiceSearchRequest) => {
    return voiceSearchMutation.mutateAsync(request);
  }, [voiceSearchMutation]);

  // Perform image search
  const performImageSearch = useCallback(async (file: File, options?: any) => {
    const formData = new FormData();
    formData.append('image', file);
    
    if (options?.searchType) {
      formData.append('search_type', options.searchType);
    }
    if (options?.includeObjects !== undefined) {
      formData.append('include_objects', options.includeObjects.toString());
    }
    if (options?.includeText !== undefined) {
      formData.append('include_text', options.includeText.toString());
    }

    return imageSearchMutation.mutateAsync(formData);
  }, [imageSearchMutation]);

  // Clear search history
  const clearSearchHistory = useCallback(() => {
    setSearchHistory([]);
    setRecentSearches([]);
    localStorage.removeItem('search-history');
    localStorage.removeItem('recent-searches');
  }, []);

  // Get cached search result
  const getCachedSearch = useCallback((query: string) => {
    return queryClient.getQueryData(['search', query]) as SearchResponse | undefined;
  }, [queryClient]);

  // Save search to favorites (localStorage)
  const saveToFavorites = useCallback((result: SearchResult) => {
    const favorites = JSON.parse(localStorage.getItem('search-favorites') || '[]');
    const updated = [result, ...favorites.filter((f: SearchResult) => f.url !== result.url)];
    localStorage.setItem('search-favorites', JSON.stringify(updated.slice(0, 50)));
    toast.success('Saved to favorites');
  }, []);

  // Get favorites from localStorage
  const getFavorites = useCallback((): SearchResult[] => {
    return JSON.parse(localStorage.getItem('search-favorites') || '[]');
  }, []);

  // Remove from favorites
  const removeFromFavorites = useCallback((url: string) => {
    const favorites = JSON.parse(localStorage.getItem('search-favorites') || '[]');
    const updated = favorites.filter((f: SearchResult) => f.url !== url);
    localStorage.setItem('search-favorites', JSON.stringify(updated));
    toast.success('Removed from favorites');
  }, []);

  return {
    // Search functions
    performSearch,
    performVoiceSearch,
    performImageSearch,
    
    // Search state
    isLoading: textSearchMutation.isLoading || voiceSearchMutation.isLoading || imageSearchMutation.isLoading,
    isError: textSearchMutation.isError || voiceSearchMutation.isError || imageSearchMutation.isError,
    error: textSearchMutation.error || voiceSearchMutation.error || imageSearchMutation.error,
    
    // Search data
    searchResults: textSearchMutation.data || voiceSearchMutation.data || imageSearchMutation.data,
    searchHistory,
    recentSearches,
    
    // Utility functions
    clearSearchHistory,
    getCachedSearch,
    saveToFavorites,
    removeFromFavorites,
    getFavorites,
    
    // Hook factories
    useSuggestions,
    useTrendingTopics,
    useSearchStats,
  };
};