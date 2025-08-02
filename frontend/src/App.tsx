import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';

// Pages
import HomePage from './pages/HomePage';

// Components
import Header from './components/Layout/Header';
import Footer from './components/Layout/Footer';

// Hooks
import { useTheme } from './hooks/useTheme';

// Styles
import './App.css';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

function App() {
  const { theme } = useTheme();

  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className={`App ${theme} min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200`}>
          <Header />
          
          <main className="flex-1">
            <Routes>
              <Route path="/" element={<HomePage />} />
            </Routes>
          </main>
          
          <Footer />
          
          {/* Global notifications */}
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              className: 'dark:bg-gray-800 dark:text-white',
            }}
          />
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;