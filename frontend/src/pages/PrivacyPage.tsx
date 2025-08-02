import React from 'react';
import { motion } from 'framer-motion';
import {
  ShieldCheckIcon,
  EyeIcon,
  LockClosedIcon,
  DocumentTextIcon,
  UserIcon,
  GlobeAltIcon,
} from '@heroicons/react/24/outline';

const PrivacyPage: React.FC = () => {
  const privacySections = [
    {
      title: 'Information We Collect',
      icon: DocumentTextIcon,
      content: [
        {
          subtitle: 'Search Queries',
          text: 'We collect the search terms you enter to provide you with relevant results and improve our search algorithms.',
        },
        {
          subtitle: 'Usage Data',
          text: 'We collect information about how you interact with our search engine, including the results you click on and the time spent on pages.',
        },
        {
          subtitle: 'Device Information',
          text: 'We collect basic device information such as browser type, operating system, and IP address to ensure optimal performance.',
        },
      ],
    },
    {
      title: 'How We Use Your Information',
      icon: UserIcon,
      content: [
        {
          subtitle: 'Providing Search Results',
          text: 'Your search queries are used to deliver relevant and accurate search results tailored to your needs.',
        },
        {
          subtitle: 'Improving Our Service',
          text: 'We analyze usage patterns to improve our search algorithms and enhance the overall user experience.',
        },
        {
          subtitle: 'Personalization',
          text: 'With your consent, we use your search history to provide personalized results and suggestions.',
        },
      ],
    },
    {
      title: 'Data Protection',
      icon: LockClosedIcon,
      content: [
        {
          subtitle: 'Encryption',
          text: 'All data transmitted between your device and our servers is encrypted using industry-standard SSL/TLS protocols.',
        },
        {
          subtitle: 'Secure Storage',
          text: 'Your data is stored in secure, encrypted databases with strict access controls and monitoring.',
        },
        {
          subtitle: 'Access Controls',
          text: 'We implement strict access controls to ensure only authorized personnel can access your personal information.',
        },
      ],
    },
    {
      title: 'Data Sharing',
      icon: GlobeAltIcon,
      content: [
        {
          subtitle: 'Third-Party Services',
          text: 'We may share data with trusted third-party services that help us provide our search functionality, such as content providers.',
        },
        {
          subtitle: 'Legal Requirements',
          text: 'We may disclose your information if required by law or to protect our rights and the safety of our users.',
        },
        {
          subtitle: 'No Sale of Personal Data',
          text: 'We do not sell, rent, or trade your personal information to third parties for marketing purposes.',
        },
      ],
    },
    {
      title: 'Your Rights',
      icon: ShieldCheckIcon,
      content: [
        {
          subtitle: 'Access and Control',
          text: 'You have the right to access, correct, or delete your personal information at any time through your account settings.',
        },
        {
          subtitle: 'Data Portability',
          text: 'You can request a copy of your personal data in a machine-readable format.',
        },
        {
          subtitle: 'Opt-Out',
          text: 'You can opt out of personalized search results and data collection at any time.',
        },
      ],
    },
    {
      title: 'Data Retention',
      icon: EyeIcon,
      content: [
        {
          subtitle: 'Search History',
          text: 'Your search history is retained for up to 18 months unless you choose to delete it earlier.',
        },
        {
          subtitle: 'Usage Analytics',
          text: 'Aggregated usage data is retained for up to 2 years for service improvement purposes.',
        },
        {
          subtitle: 'Account Data',
          text: 'Account information is retained until you delete your account or request data deletion.',
        },
      ],
    },
  ];

  const contactInfo = {
    email: 'privacy@astrafind.com',
    address: '123 Privacy Street, Security City, SC 12345',
    phone: '+1 (555) 123-4567',
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-8"
        >
          {/* Header */}
          <div className="text-center">
            <div className="flex justify-center mb-4">
              <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/20 rounded-full flex items-center justify-center">
                <ShieldCheckIcon className="h-8 w-8 text-blue-600" />
              </div>
            </div>
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Privacy Policy
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              Your privacy is important to us. This policy explains how we collect, use, and protect your information when you use AstraFind.
            </p>
            <div className="mt-6 text-sm text-gray-500 dark:text-gray-400">
              Last updated: January 15, 2024
            </div>
          </div>

          {/* Privacy Sections */}
          {privacySections.map((section, sectionIndex) => (
            <motion.div
              key={section.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: sectionIndex * 0.1 }}
              className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden"
            >
              <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center space-x-3">
                  <section.icon className="h-6 w-6 text-blue-600" />
                  <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                    {section.title}
                  </h2>
                </div>
              </div>

              <div className="px-6 py-6 space-y-6">
                {section.content.map((item, itemIndex) => (
                  <div key={itemIndex}>
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                      {item.subtitle}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
                      {item.text}
                    </p>
                  </div>
                ))}
              </div>
            </motion.div>
          ))}

          {/* Contact Information */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6"
          >
            <h2 className="text-xl font-semibold text-blue-900 dark:text-blue-100 mb-4">
              Contact Us
            </h2>
            <p className="text-blue-800 dark:text-blue-200 mb-4">
              If you have any questions about this Privacy Policy or our data practices, please contact us:
            </p>
            <div className="space-y-2 text-blue-800 dark:text-blue-200">
              <p><strong>Email:</strong> {contactInfo.email}</p>
              <p><strong>Phone:</strong> {contactInfo.phone}</p>
              <p><strong>Address:</strong> {contactInfo.address}</p>
            </div>
          </motion.div>

          {/* Additional Information */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="bg-gray-100 dark:bg-gray-700 rounded-lg p-6"
          >
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
              Additional Information
            </h2>
            <div className="space-y-4 text-gray-600 dark:text-gray-400">
              <p>
                <strong>Cookies:</strong> We use cookies to enhance your search experience and provide personalized results. You can control cookie settings through your browser preferences.
              </p>
              <p>
                <strong>Children's Privacy:</strong> Our service is not intended for children under 13. We do not knowingly collect personal information from children under 13.
              </p>
              <p>
                <strong>International Users:</strong> If you are accessing our service from outside the United States, please note that your information may be transferred to and processed in the United States.
              </p>
              <p>
                <strong>Changes to This Policy:</strong> We may update this Privacy Policy from time to time. We will notify you of any changes by posting the new Privacy Policy on this page.
              </p>
            </div>
          </motion.div>

          {/* Footer */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
            className="text-center text-sm text-gray-500 dark:text-gray-400 pt-8 border-t border-gray-200 dark:border-gray-700"
          >
            <p>
              By using AstraFind, you agree to the collection and use of information in accordance with this Privacy Policy.
            </p>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};

export default PrivacyPage;