import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  MagnifyingGlassIcon,
  FunnelIcon,
  CalendarIcon,
  GlobeAltIcon,
  DocumentTextIcon,
  PhotoIcon,
  VideoCameraIcon,
  MicrophoneIcon,
} from '@heroicons/react/24/outline';

const AdvancedSearchPage: React.FC = () => {
  const navigate = useNavigate();
  const [searchForm, setSearchForm] = useState({
    query: '',
    exactPhrase: '',
    excludeWords: '',
    site: '',
    fileType: '',
    dateRange: '',
    language: '',
    region: '',
  });

  const handleInputChange = (field: string, value: string) => {
    setSearchForm(prev => ({ ...prev, [field]: value }));
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const params = new URLSearchParams();
    
    if (searchForm.query) params.append('q', searchForm.query);
    if (searchForm.exactPhrase) params.append('exact', searchForm.exactPhrase);
    if (searchForm.excludeWords) params.append('exclude', searchForm.excludeWords);
    if (searchForm.site) params.append('site', searchForm.site);
    if (searchForm.fileType) params.append('type', searchForm.fileType);
    if (searchForm.dateRange) params.append('date', searchForm.dateRange);
    if (searchForm.language) params.append('lang', searchForm.language);
    if (searchForm.region) params.append('region', searchForm.region);
    
    navigate(`/search?${params.toString()}`);
  };

  const fileTypes = [
    { value: '', label: 'Any file type' },
    { value: 'pdf', label: 'PDF documents' },
    { value: 'doc', label: 'Word documents' },
    { value: 'xls', label: 'Excel spreadsheets' },
    { value: 'ppt', label: 'PowerPoint presentations' },
    { value: 'txt', label: 'Text files' },
  ];

  const dateRanges = [
    { value: '', label: 'Any time' },
    { value: 'day', label: 'Past 24 hours' },
    { value: 'week', label: 'Past week' },
    { value: 'month', label: 'Past month' },
    { value: 'year', label: 'Past year' },
  ];

  const languages = [
    { value: '', label: 'Any language' },
    { value: 'en', label: 'English' },
    { value: 'es', label: 'Spanish' },
    { value: 'fr', label: 'French' },
    { value: 'de', label: 'German' },
    { value: 'it', label: 'Italian' },
    { value: 'pt', label: 'Portuguese' },
    { value: 'ru', label: 'Russian' },
    { value: 'ja', label: 'Japanese' },
    { value: 'ko', label: 'Korean' },
    { value: 'zh', label: 'Chinese' },
  ];

  const regions = [
    { value: '', label: 'Any region' },
    { value: 'us', label: 'United States' },
    { value: 'uk', label: 'United Kingdom' },
    { value: 'ca', label: 'Canada' },
    { value: 'au', label: 'Australia' },
    { value: 'de', label: 'Germany' },
    { value: 'fr', label: 'France' },
    { value: 'jp', label: 'Japan' },
    { value: 'in', label: 'India' },
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6"
        >
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
              Advanced Search
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Use advanced search operators and filters to find exactly what you're looking for
            </p>
          </div>

          <form onSubmit={handleSearch} className="space-y-6">
            {/* Main Search Query */}
            <div>
              <label htmlFor="query" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Find pages with...
              </label>
              <div className="relative">
                <input
                  type="text"
                  id="query"
                  value={searchForm.query}
                  onChange={(e) => handleInputChange('query', e.target.value)}
                  className="w-full pl-10 pr-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white"
                  placeholder="Enter your search terms"
                />
                <MagnifyingGlassIcon className="absolute left-3 top-3.5 h-5 w-5 text-gray-400" />
              </div>
            </div>

            {/* Exact Phrase */}
            <div>
              <label htmlFor="exactPhrase" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                With the exact phrase
              </label>
              <input
                type="text"
                id="exactPhrase"
                value={searchForm.exactPhrase}
                onChange={(e) => handleInputChange('exactPhrase', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white"
                placeholder="Enter exact phrase (optional)"
              />
            </div>

            {/* Exclude Words */}
            <div>
              <label htmlFor="excludeWords" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                With none of the words
              </label>
              <input
                type="text"
                id="excludeWords"
                value={searchForm.excludeWords}
                onChange={(e) => handleInputChange('excludeWords', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white"
                placeholder="Enter words to exclude (optional)"
              />
            </div>

            {/* Site Restriction */}
            <div>
              <label htmlFor="site" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Site or domain
              </label>
              <input
                type="text"
                id="site"
                value={searchForm.site}
                onChange={(e) => handleInputChange('site', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white"
                placeholder="e.g., example.com or .edu"
              />
            </div>

            {/* File Type */}
            <div>
              <label htmlFor="fileType" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                File type
              </label>
              <select
                id="fileType"
                value={searchForm.fileType}
                onChange={(e) => handleInputChange('fileType', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                {fileTypes.map((type) => (
                  <option key={type.value} value={type.value}>
                    {type.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Date Range */}
            <div>
              <label htmlFor="dateRange" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Date range
              </label>
              <select
                id="dateRange"
                value={searchForm.dateRange}
                onChange={(e) => handleInputChange('dateRange', e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                {dateRanges.map((range) => (
                  <option key={range.value} value={range.value}>
                    {range.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Language and Region */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="language" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Language
                </label>
                <select
                  id="language"
                  value={searchForm.language}
                  onChange={(e) => handleInputChange('language', e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  {languages.map((lang) => (
                    <option key={lang.value} value={lang.value}>
                      {lang.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="region" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Region
                </label>
                <select
                  id="region"
                  value={searchForm.region}
                  onChange={(e) => handleInputChange('region', e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  {regions.map((region) => (
                    <option key={region.value} value={region.value}>
                      {region.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Search Button */}
            <div className="flex justify-center pt-6">
              <button
                type="submit"
                className="btn-primary px-8 py-3 text-lg"
              >
                <MagnifyingGlassIcon className="h-5 w-5 mr-2" />
                Advanced Search
              </button>
            </div>
          </form>

          {/* Search Tips */}
          <div className="mt-8 p-6 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-3">
              Search Tips
            </h3>
            <ul className="space-y-2 text-sm text-blue-800 dark:text-blue-200">
              <li>• Use quotes for exact phrase matching: "artificial intelligence"</li>
              <li>• Use - to exclude words: AI -chatbot</li>
              <li>• Use site: to search specific domains: site:example.com</li>
              <li>• Use filetype: to find specific file types: filetype:pdf</li>
              <li>• Use date ranges to find recent content</li>
            </ul>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default AdvancedSearchPage;