import React from 'react';
import { motion } from 'framer-motion';
import {
  GlobeAltIcon,
  DocumentTextIcon,
  PhotoIcon,
  VideoCameraIcon,
  ClockIcon,
  UsersIcon,
} from '@heroicons/react/24/outline';

const SearchStats: React.FC = () => {
  const stats = [
    {
      icon: GlobeAltIcon,
      label: 'Web Pages Indexed',
      value: '50B+',
      color: 'text-blue-600',
      bgColor: 'bg-blue-100 dark:bg-blue-900/20',
    },
    {
      icon: DocumentTextIcon,
      label: 'Documents Processed',
      value: '100M+',
      color: 'text-green-600',
      bgColor: 'bg-green-100 dark:bg-green-900/20',
    },
    {
      icon: PhotoIcon,
      label: 'Images Analyzed',
      value: '10B+',
      color: 'text-purple-600',
      bgColor: 'bg-purple-100 dark:bg-purple-900/20',
    },
    {
      icon: VideoCameraIcon,
      label: 'Videos Indexed',
      value: '1B+',
      color: 'text-orange-600',
      bgColor: 'bg-orange-100 dark:bg-orange-900/20',
    },
    {
      icon: ClockIcon,
      label: 'Avg. Response Time',
      value: '<100ms',
      color: 'text-red-600',
      bgColor: 'bg-red-100 dark:bg-red-900/20',
    },
    {
      icon: UsersIcon,
      label: 'Daily Searches',
      value: '10M+',
      color: 'text-indigo-600',
      bgColor: 'bg-indigo-100 dark:bg-indigo-900/20',
    },
  ];

  return (
    <div className="mt-12">
      <div className="text-center mb-8">
        <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
          Search Engine Statistics
        </h3>
        <p className="text-gray-600 dark:text-gray-400">
          Real-time data about our search engine performance
        </p>
      </div>
      
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="text-center p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow duration-200"
          >
            <div className={`inline-flex items-center justify-center w-12 h-12 rounded-full ${stat.bgColor} mb-3`}>
              <stat.icon className={`h-6 w-6 ${stat.color}`} />
            </div>
            <div className={`text-2xl font-bold ${stat.color} mb-1`}>
              {stat.value}
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400">
              {stat.label}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default SearchStats;