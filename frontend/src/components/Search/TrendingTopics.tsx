import React from 'react';
import { ArrowTrendingUpIcon } from '@heroicons/react/24/outline';

interface TrendingTopicsProps {
  onTopicSelect: (topic: string) => void;
  topics?: Array<{ name: string; count: number }>;
}

const TrendingTopics: React.FC<TrendingTopicsProps> = ({
  onTopicSelect,
  topics = [
    { name: "AI Ethics", count: 1234 },
    { name: "Quantum Computing", count: 987 },
    { name: "Climate Tech", count: 756 },
    { name: "Web3", count: 654 },
    { name: "Cybersecurity", count: 543 }
  ]
}) => {
  return (
    <div className="mt-8">
      <div className="flex items-center mb-4">
        <ArrowTrendingUpIcon className="h-5 w-5 text-red-500 mr-2" />
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Trending Topics
        </h3>
      </div>
      <div className="space-y-2">
        {topics.map((topic, index) => (
          <button
            key={index}
            onClick={() => onTopicSelect(topic.name)}
            className="w-full flex items-center justify-between p-3 text-left hover:bg-gray-50 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            <span className="text-gray-700 dark:text-gray-300 font-medium">
              {topic.name}
            </span>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {topic.count.toLocaleString()} searches
            </span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default TrendingTopics;