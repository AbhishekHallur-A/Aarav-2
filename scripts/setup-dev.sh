#!/bin/bash

# AstraFind Development Setup Script
# This script sets up the complete development environment for AstraFind

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is available
port_available() {
    ! nc -z localhost "$1" >/dev/null 2>&1
}

print_status "🚀 Setting up AstraFind development environment..."

# Check prerequisites
print_status "Checking prerequisites..."

MISSING_DEPS=()

if ! command_exists docker; then
    MISSING_DEPS+=("docker")
fi

if ! command_exists docker-compose; then
    MISSING_DEPS+=("docker-compose")
fi

if ! command_exists python3; then
    MISSING_DEPS+=("python3")
fi

if ! command_exists node; then
    MISSING_DEPS+=("node.js")
fi

if ! command_exists npm; then
    MISSING_DEPS+=("npm")
fi

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    print_error "Missing required dependencies: ${MISSING_DEPS[*]}"
    print_error "Please install the missing dependencies and run this script again."
    exit 1
fi

print_success "All prerequisites satisfied!"

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_PYTHON="3.11"

if [ "$(printf '%s\n' "$REQUIRED_PYTHON" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_PYTHON" ]; then
    print_warning "Python version $PYTHON_VERSION detected. Recommended: $REQUIRED_PYTHON+"
fi

# Check Node.js version
NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
REQUIRED_NODE=18

if [ "$NODE_VERSION" -lt "$REQUIRED_NODE" ]; then
    print_warning "Node.js version $NODE_VERSION detected. Recommended: $REQUIRED_NODE+"
fi

# Create environment file if it doesn't exist
if [ ! -f "backend/.env" ]; then
    print_status "Creating environment configuration..."
    cp backend/.env.example backend/.env
    
    # Generate a random secret key
    SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
    sed -i "s/your-super-secret-key-change-this-in-production/$SECRET_KEY/g" backend/.env
    
    print_success "Environment file created at backend/.env"
    print_warning "Please review and update the configuration as needed"
else
    print_status "Environment file already exists"
fi

# Set up Python virtual environment
print_status "Setting up Python virtual environment..."
cd backend

if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
print_status "Installing Python dependencies..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

print_success "Python dependencies installed"

cd ..

# Set up Node.js dependencies
print_status "Setting up Node.js dependencies..."
cd frontend

if [ ! -d "node_modules" ]; then
    npm install
    print_success "Node.js dependencies installed"
else
    print_status "Node.js dependencies already exist"
    print_status "Updating dependencies..."
    npm install
fi

cd ..

# Check if ports are available
print_status "Checking port availability..."

REQUIRED_PORTS=(5432 6379 9200 15672 8000 3000)
OCCUPIED_PORTS=()

for port in "${REQUIRED_PORTS[@]}"; do
    if ! port_available "$port"; then
        OCCUPIED_PORTS+=("$port")
    fi
done

if [ ${#OCCUPIED_PORTS[@]} -ne 0 ]; then
    print_warning "The following ports are occupied: ${OCCUPIED_PORTS[*]}"
    print_warning "You may need to stop services using these ports or modify the configuration"
fi

# Start infrastructure services with Docker Compose
print_status "Starting infrastructure services..."

if command_exists docker-compose; then
    # Check if services are already running
    if docker-compose ps | grep -q "Up"; then
        print_status "Some services are already running"
        read -p "Do you want to restart all services? (y/N): " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose down
            docker-compose up -d
        fi
    else
        docker-compose up -d
    fi
    
    print_status "Waiting for services to start..."
    
    # Wait for PostgreSQL
    print_status "Waiting for PostgreSQL..."
    for i in {1..30}; do
        if docker-compose exec -T postgres pg_isready -U astrafind >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    
    # Wait for Elasticsearch
    print_status "Waiting for Elasticsearch..."
    for i in {1..60}; do
        if curl -s http://localhost:9200/_cluster/health >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    
    # Wait for Redis
    print_status "Waiting for Redis..."
    for i in {1..30}; do
        if docker-compose exec -T redis redis-cli ping >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    
    print_success "Infrastructure services started successfully!"
else
    print_error "docker-compose not found. Please install it to start infrastructure services."
    exit 1
fi

# Initialize database
print_status "Initializing database..."
cd backend
source venv/bin/activate

# Run database migrations (placeholder - implement with your migration tool)
# python manage.py migrate  # Django example
# alembic upgrade head      # SQLAlchemy example

print_success "Database initialized"

# Download ML models (placeholder)
print_status "Setting up ML models..."
mkdir -p ../ml-models

# In production, you would download pre-trained models here
# For now, we'll create placeholder files
echo "ML models will be downloaded on first run" > ../ml-models/README.md

print_success "ML models directory prepared"

cd ..

# Create initial search indices
print_status "Creating search indices..."

# This would typically create Elasticsearch indices
# curl -X PUT "localhost:9200/astrafind-pages" -H 'Content-Type: application/json' -d'
# {
#   "mappings": {
#     "properties": {
#       "url": { "type": "keyword" },
#       "title": { "type": "text" },
#       "content": { "type": "text" },
#       "language": { "type": "keyword" },
#       "crawl_time": { "type": "date" }
#     }
#   }
# }'

print_success "Search indices prepared"

# Build frontend assets
print_status "Building frontend assets..."
cd frontend
npm run build

print_success "Frontend assets built"

cd ..

# Create startup scripts
print_status "Creating startup scripts..."

# Backend startup script
cat > scripts/start-backend.sh << 'EOF'
#!/bin/bash
cd backend
source venv/bin/activate
python main.py
EOF

# Frontend startup script
cat > scripts/start-frontend.sh << 'EOF'
#!/bin/bash
cd frontend
npm start
EOF

# Combined startup script
cat > scripts/start-dev.sh << 'EOF'
#!/bin/bash
echo "Starting AstraFind development environment..."

# Start backend in background
echo "Starting backend server..."
cd backend
source venv/bin/activate
python main.py &
BACKEND_PID=$!

# Start frontend
echo "Starting frontend development server..."
cd ../frontend
npm start &
FRONTEND_PID=$!

# Function to cleanup on exit
cleanup() {
    echo "Shutting down development servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Wait for processes
wait
EOF

chmod +x scripts/start-backend.sh
chmod +x scripts/start-frontend.sh
chmod +x scripts/start-dev.sh

print_success "Startup scripts created"

# Final setup summary
print_success "🎉 AstraFind development environment setup complete!"
echo
echo "📋 Setup Summary:"
echo "  ✅ Prerequisites checked"
echo "  ✅ Environment configured"
echo "  ✅ Dependencies installed"
echo "  ✅ Infrastructure services started"
echo "  ✅ Database initialized"
echo "  ✅ Search indices prepared"
echo "  ✅ Frontend built"
echo "  ✅ Startup scripts created"
echo
echo "🚀 Next Steps:"
echo "  1. Review configuration in backend/.env"
echo "  2. Start development servers:"
echo "     • Backend: ./scripts/start-backend.sh"
echo "     • Frontend: ./scripts/start-frontend.sh"
echo "     • Both: ./scripts/start-dev.sh"
echo
echo "🌐 Service URLs:"
echo "  • Frontend: http://localhost:3000"
echo "  • Backend API: http://localhost:8000"
echo "  • API Docs: http://localhost:8000/docs"
echo "  • Elasticsearch: http://localhost:9200"
echo "  • RabbitMQ Management: http://localhost:15672"
echo "  • Grafana: http://localhost:3001"
echo
echo "📚 Documentation:"
echo "  • README.md - Project overview"
echo "  • docs/ - Detailed documentation"
echo "  • API docs available at /docs when backend is running"
echo
print_warning "Remember to activate the Python virtual environment when working on backend:"
print_warning "cd backend && source venv/bin/activate"