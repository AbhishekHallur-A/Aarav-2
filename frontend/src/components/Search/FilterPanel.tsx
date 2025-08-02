import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  FunnelIcon,
  CalendarIcon,
  GlobeAltIcon,
  DocumentTextIcon,
  PhotoIcon,
  VideoCameraIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';

interface FilterPanelProps {
  filters: {
    type: string;
    dateRange: string;
    language: string;
  };
  onFilterChange: (filters: any) => void;
  resultTypes: Array<{
    id: string;
    name: string;
    icon: any;
    count: number;
  }>;
}

const FilterPanel: React.FC<FilterPanelProps> = ({
  filters,
  onFilterChange,
  resultTypes,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const dateRanges = [
    { value: 'all', label: 'Any time' },
    { value: 'day', label: 'Past 24 hours' },
    { value: 'week', label: 'Past week' },
    { value: 'month', label: 'Past month' },
    { value: 'year', label: 'Past year' },
  ];

  const languages = [
    { value: 'all', label: 'Any language' },
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

  const handleFilterChange = (key: string, value: string) => {
    onFilterChange({ ...filters, [key]: value });
  };

  const clearFilters = () => {
    onFilterChange({
      type: 'all',
      dateRange: 'all',
      language: 'all',
    });
  };

  const hasActiveFilters = filters.type !== 'all' || filters.dateRange !== 'all' || filters.language !== 'all';

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <FunnelIcon className="h-5 w-5 text-gray-600 dark:text-gray-400" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Filters
            </h3>
          </div>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          >
            <XMarkIcon className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="p-4 space-y-6">
        {/* Result Type Filter */}
        <div>
          <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
            Result Type
          </h4>
          <div className="space-y-2">
            {resultTypes.map((type) => (
              <button
                key={type.id}
                onClick={() => handleFilterChange('type', type.id)}
                className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-sm transition-colors duration-200 ${
                  filters.type === type.id
                    ? 'bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                <div className="flex items-center space-x-2">
                  <type.icon className="h-4 w-4" />
                  <span>{type.name}</span>
                </div>
                <span className="text-xs bg-gray-200 dark:bg-gray-600 px-2 py-1 rounded-full">
                  {type.count}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Date Range Filter */}
        <div>
          <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
            Date Range
          </h4>
          <select
            value={filters.dateRange}
            onChange={(e) => handleFilterChange('dateRange', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
          >
            {dateRanges.map((range) => (
              <option key={range.value} value={range.value}>
                {range.label}
              </option>
            ))}
          </select>
        </div>

        {/* Language Filter */}
        <div>
          <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
            Language
          </h4>
          <select
            value={filters.language}
            onChange={(e) => handleFilterChange('language', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
          >
            {languages.map((lang) => (
              <option key={lang.value} value={lang.value}>
                {lang.label}
              </option>
            ))}
          </select>
        </div>

        {/* Clear Filters */}
        {hasActiveFilters && (
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            onClick={clearFilters}
            className="w-full px-4 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors duration-200"
          >
            Clear All Filters
          </motion.button>
        )}
      </div>
    </div>
  );
};

export default FilterPanel;