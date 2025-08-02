import React from 'react';
import { ChartBarIcon, GlobeAltIcon, ClockIcon } from '@heroicons/react/24/outline';

interface SearchStatsProps {
  stats?: {
    totalSearches: number;
    activeUsers: number;
    averageResponseTime: number;
  };
}

const SearchStats: React.FC<SearchStatsProps> = ({
  stats = {
    totalSearches: 1234567,
    activeUsers: 89012,
    averageResponseTime: 0.15
  }
}) => {
  return (
    <div className="mt-8 p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        Search Statistics
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="flex items-center">
          <ChartBarIcon className="h-8 w-8 text-blue-500 mr-3" />
          <div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {stats.totalSearches.toLocaleString()}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Total Searches
            </p>
          </div>
        </div>
        <div className="flex items-center">
          <GlobeAltIcon className="h-8 w-8 text-green-500 mr-3" />
          <div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {stats.activeUsers.toLocaleString()}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Active Users
            </p>
          </div>
        </div>
        <div className="flex items-center">
          <ClockIcon className="h-8 w-8 text-purple-500 mr-3" />
          <div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {stats.averageResponseTime}s
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Avg Response Time
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SearchStats;