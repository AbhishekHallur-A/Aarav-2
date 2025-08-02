import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUpIcon, FireIcon } from '@heroicons/react/24/outline';

interface TrendingTopicsProps {
  onTopicClick: (topic: string) => void;
}

const TrendingTopics: React.FC<TrendingTopicsProps> = ({
  onTopicClick,
}) => {
  const trendingTopics = [
    {
      topic: 'AI Chatbots',
      trend: '+45%',
      category: 'Technology',
    },
    {
      topic: 'Web3 Development',
      trend: '+32%',
      category: 'Programming',
    },
    {
      topic: 'Data Privacy',
      trend: '+28%',
      category: 'Security',
    },
    {
      topic: 'Cloud Computing',
      trend: '+25%',
      category: 'Technology',
    },
    {
      topic: 'Machine Learning',
      trend: '+22%',
      category: 'AI',
    },
  ];

  return (
    <div className="mt-8">
      <div className="flex items-center space-x-2 mb-4">
        <FireIcon className="h-5 w-5 text-orange-500" />
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Trending Topics
        </h3>
      </div>
      <div className="space-y-3">
        {trendingTopics.map((item, index) => (
          <motion.button
            key={item.topic}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            onClick={() => onTopicClick(item.topic)}
            className="w-full flex items-center justify-between p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-md hover:border-orange-300 dark:hover:border-orange-600 transition-all duration-200 group"
          >
            <div className="flex items-center space-x-3">
              <div className="flex items-center justify-center w-8 h-8 bg-orange-100 dark:bg-orange-900/20 rounded-full">
                <span className="text-sm font-semibold text-orange-600 dark:text-orange-400">
                  {index + 1}
                </span>
              </div>
              <div className="text-left">
                <p className="text-sm font-medium text-gray-900 dark:text-white group-hover:text-orange-600 dark:group-hover:text-orange-400 transition-colors duration-200">
                  {item.topic}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {item.category}
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <TrendingUpIcon className="h-4 w-4 text-green-500" />
              <span className="text-sm font-medium text-green-600 dark:text-green-400">
                {item.trend}
              </span>
            </div>
          </motion.button>
        ))}
      </div>
    </div>
  );
};

export default TrendingTopics;