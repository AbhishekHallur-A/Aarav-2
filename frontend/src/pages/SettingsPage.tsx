import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Cog6ToothIcon,
  SunIcon,
  MoonIcon,
  GlobeAltIcon,
  ShieldCheckIcon,
  BellIcon,
  MagnifyingGlassIcon,
  EyeIcon,
  EyeSlashIcon,
} from '@heroicons/react/24/outline';
import { useTheme } from '../hooks/useTheme';

const SettingsPage: React.FC = () => {
  const { theme, toggleTheme } = useTheme();
  const [settings, setSettings] = useState({
    language: 'en',
    region: 'us',
    safeSearch: true,
    notifications: false,
    searchHistory: true,
    personalizedResults: true,
    autoComplete: true,
    voiceSearch: false,
    imageSearch: true,
  });

  const handleSettingChange = (key: string, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const languages = [
    { code: 'en', name: 'English' },
    { code: 'es', name: 'Spanish' },
    { code: 'fr', name: 'French' },
    { code: 'de', name: 'German' },
    { code: 'it', name: 'Italian' },
    { code: 'pt', name: 'Portuguese' },
    { code: 'ru', name: 'Russian' },
    { code: 'ja', name: 'Japanese' },
    { code: 'ko', name: 'Korean' },
    { code: 'zh', name: 'Chinese' },
  ];

  const regions = [
    { code: 'us', name: 'United States' },
    { code: 'uk', name: 'United Kingdom' },
    { code: 'ca', name: 'Canada' },
    { code: 'au', name: 'Australia' },
    { code: 'de', name: 'Germany' },
    { code: 'fr', name: 'France' },
    { code: 'jp', name: 'Japan' },
    { code: 'in', name: 'India' },
  ];

  const settingSections = [
    {
      title: 'Appearance',
      icon: Cog6ToothIcon,
      settings: [
        {
          key: 'theme',
          label: 'Theme',
          type: 'select',
          options: [
            { value: 'light', label: 'Light' },
            { value: 'dark', label: 'Dark' },
            { value: 'auto', label: 'System' },
          ],
          currentValue: theme,
          onChange: toggleTheme,
        },
      ],
    },
    {
      title: 'Language & Region',
      icon: GlobeAltIcon,
      settings: [
        {
          key: 'language',
          label: 'Language',
          type: 'select',
          options: languages.map(lang => ({ value: lang.code, label: lang.name })),
          currentValue: settings.language,
          onChange: (value: string) => handleSettingChange('language', value),
        },
        {
          key: 'region',
          label: 'Region',
          type: 'select',
          options: regions.map(region => ({ value: region.code, label: region.name })),
          currentValue: settings.region,
          onChange: (value: string) => handleSettingChange('region', value),
        },
      ],
    },
    {
      title: 'Search Preferences',
      icon: MagnifyingGlassIcon,
      settings: [
        {
          key: 'safeSearch',
          label: 'Safe Search',
          description: 'Filter out explicit content from search results',
          type: 'toggle',
          currentValue: settings.safeSearch,
          onChange: (value: boolean) => handleSettingChange('safeSearch', value),
        },
        {
          key: 'autoComplete',
          label: 'Auto-complete',
          description: 'Show search suggestions as you type',
          type: 'toggle',
          currentValue: settings.autoComplete,
          onChange: (value: boolean) => handleSettingChange('autoComplete', value),
        },
        {
          key: 'voiceSearch',
          label: 'Voice Search',
          description: 'Enable voice input for search queries',
          type: 'toggle',
          currentValue: settings.voiceSearch,
          onChange: (value: boolean) => handleSettingChange('voiceSearch', value),
        },
        {
          key: 'imageSearch',
          label: 'Image Search',
          description: 'Allow searching by uploading images',
          type: 'toggle',
          currentValue: settings.imageSearch,
          onChange: (value: boolean) => handleSettingChange('imageSearch', value),
        },
      ],
    },
    {
      title: 'Privacy & Security',
      icon: ShieldCheckIcon,
      settings: [
        {
          key: 'searchHistory',
          label: 'Search History',
          description: 'Save your search queries for personalized results',
          type: 'toggle',
          currentValue: settings.searchHistory,
          onChange: (value: boolean) => handleSettingChange('searchHistory', value),
        },
        {
          key: 'personalizedResults',
          label: 'Personalized Results',
          description: 'Use your search history to improve results',
          type: 'toggle',
          currentValue: settings.personalizedResults,
          onChange: (value: boolean) => handleSettingChange('personalizedResults', value),
        },
      ],
    },
    {
      title: 'Notifications',
      icon: BellIcon,
      settings: [
        {
          key: 'notifications',
          label: 'Push Notifications',
          description: 'Receive notifications about new features and updates',
          type: 'toggle',
          currentValue: settings.notifications,
          onChange: (value: boolean) => handleSettingChange('notifications', value),
        },
      ],
    },
  ];

  const renderSettingControl = (setting: any) => {
    switch (setting.type) {
      case 'select':
        return (
          <select
            value={setting.currentValue}
            onChange={(e) => setting.onChange(e.target.value)}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white"
          >
            {setting.options.map((option: any) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        );
      case 'toggle':
        return (
          <button
            onClick={() => setting.onChange(!setting.currentValue)}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
              setting.currentValue
                ? 'bg-blue-600'
                : 'bg-gray-200 dark:bg-gray-700'
            }`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-200 ${
                setting.currentValue ? 'translate-x-6' : 'translate-x-1'
              }`}
            />
          </button>
        );
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
              Settings
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Customize your search experience and preferences
            </p>
          </div>

          {/* Settings Sections */}
          {settingSections.map((section, sectionIndex) => (
            <motion.div
              key={section.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: sectionIndex * 0.1 }}
              className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden"
            >
              <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center space-x-3">
                  <section.icon className="h-5 w-5 text-blue-600" />
                  <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                    {section.title}
                  </h2>
                </div>
              </div>

              <div className="divide-y divide-gray-200 dark:divide-gray-700">
                {section.settings.map((setting) => (
                  <div key={setting.key} className="px-6 py-4">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <label className="text-sm font-medium text-gray-900 dark:text-white">
                          {setting.label}
                        </label>
                        {'description' in setting && setting.description && (
                          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                            {setting.description}
                          </p>
                        )}
                      </div>
                      <div className="ml-4">
                        {renderSettingControl(setting)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          ))}

          {/* Actions */}
          <div className="flex justify-center space-x-4 pt-6">
            <button className="btn-secondary px-6 py-2">
              Reset to Defaults
            </button>
            <button className="btn-primary px-6 py-2">
              Save Changes
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default SettingsPage;