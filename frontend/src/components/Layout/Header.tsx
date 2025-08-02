import React from 'react';
import { Link } from 'react-router-dom';

const Header: React.FC = () => {
  return (
    <header className="bg-white dark:bg-gray-800 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4">
          <Link to="/" className="text-2xl font-bold text-gray-900 dark:text-white">
            AstraFind
          </Link>
          <nav className="hidden md:flex space-x-8">
            <Link to="/" className="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white">
              Home
            </Link>
            <Link to="/advanced" className="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white">
              Advanced Search
            </Link>
            <Link to="/settings" className="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white">
              Settings
            </Link>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header;