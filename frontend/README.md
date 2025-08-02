# AstraFind Frontend

A modern, AI-powered search engine frontend built with React, TypeScript, and Tailwind CSS.

## Features

- 🔍 **Advanced Search**: AI-powered search with intelligent algorithms
- 🎤 **Voice Search**: Speech-to-text search capabilities
- 📷 **Image Search**: Visual search by uploading images
- 🌙 **Dark Mode**: Beautiful dark and light theme support
- 📱 **Responsive Design**: Optimized for all devices
- ⚡ **Fast Performance**: Built with modern React patterns
- 🔒 **Privacy Focused**: User privacy and data protection

## Tech Stack

- **React 18** - Modern React with hooks and functional components
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Smooth animations and transitions
- **React Router** - Client-side routing
- **React Query** - Data fetching and caching
- **Heroicons** - Beautiful SVG icons
- **Axios** - HTTP client for API calls

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/astrafind.git
cd astrafind/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The application will be available at `http://localhost:3000`.

### Building for Production

```bash
npm run build
```

The build files will be created in the `build` directory.

### Running Tests

```bash
npm test
```

### Code Quality

```bash
# Lint code
npm run lint

# Fix linting issues
npm run lint:fix

# Format code
npm run format

# Type checking
npm run type-check
```

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── Layout/         # Layout components (Header, Footer)
│   ├── Search/         # Search-related components
│   └── UI/             # Basic UI components
├── hooks/              # Custom React hooks
├── pages/              # Page components
├── App.tsx             # Main application component
├── App.css             # Global styles
└── index.tsx           # Application entry point
```

## Key Components

### Search Components
- `SearchBox` - Main search input with voice and image search
- `SearchResultCard` - Individual search result display
- `FilterPanel` - Search filters and options
- `VoiceSearch` - Voice recognition modal
- `ImageSearch` - Image upload and search modal

### Pages
- `HomePage` - Landing page with search interface
- `SearchResultsPage` - Search results display
- `AdvancedSearchPage` - Advanced search options
- `SettingsPage` - User preferences and settings
- `PrivacyPage` - Privacy policy and information

### Hooks
- `useSearch` - Search functionality and API integration
- `useVoiceSearch` - Voice recognition capabilities
- `useTheme` - Dark/light theme management

## Configuration

### Environment Variables

Create a `.env` file in the frontend directory:

```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_GA_TRACKING_ID=your-ga-tracking-id
```

### Tailwind Configuration

The project uses Tailwind CSS with custom configuration in `tailwind.config.js`. Key features:

- Dark mode support
- Custom color palette
- Responsive breakpoints
- Custom animations

## Deployment

### GitHub Pages

The project includes GitHub Actions workflow for automatic deployment to GitHub Pages. The workflow:

1. Builds the application
2. Runs tests
3. Deploys to GitHub Pages (on main branch)

### Manual Deployment

1. Build the application:
```bash
npm run build
```

2. Deploy the `build` directory to your hosting provider.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `npm test`
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## Development Guidelines

### Code Style
- Use TypeScript for all new code
- Follow ESLint and Prettier configurations
- Use functional components with hooks
- Implement proper error handling
- Add JSDoc comments for complex functions

### Component Guidelines
- Keep components small and focused
- Use proper TypeScript interfaces
- Implement proper prop validation
- Use React.memo for performance optimization when needed

### State Management
- Use React hooks for local state
- Use React Query for server state
- Keep state as close to where it's used as possible

## Performance Optimization

- Code splitting with React.lazy()
- Image optimization
- Bundle analysis with webpack-bundle-analyzer
- Memoization of expensive calculations
- Optimized re-renders with React.memo

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Email: support@astrafind.com
- Documentation: https://docs.astrafind.com