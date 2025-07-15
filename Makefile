.PHONY: help install dev test clean build deploy

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  dev         - Start development environment"
	@echo "  test        - Run tests"
	@echo "  clean       - Clean up temporary files"
	@echo "  build       - Build Docker images"
	@echo "  deploy      - Deploy to production"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"

# Install dependencies
install:
	pip install -r requirements.txt
	cd src/frontend/react_app && npm install

# Start development environment
dev:
	docker-compose up -d postgres redis neo4j
	cd src/backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
	cd src/frontend/react_app && npm run dev &
	cd src/frontend/streamlit_app && streamlit run main.py

# Run tests
test:
	pytest src/backend/tests/ -v --cov=src/backend --cov-report=html
	cd src/frontend/react_app && npm test

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build

# Build Docker images
build:
	docker-compose build

# Deploy to production
deploy:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Lint code
lint:
	flake8 src/backend/
	mypy src/backend/
	cd src/frontend/react_app && npm run lint

# Format code
format:
	black src/backend/
	isort src/backend/
	cd src/frontend/react_app && npm run format

# Setup environment
setup:
	python -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt
	cp .env.example .env
	echo "Please edit .env with your configuration"

# Initialize databases
init-db:
	docker-compose up -d postgres redis neo4j
	sleep 10
	cd src/backend && alembic upgrade head
	python scripts/data/build_knowledge_graph.py

# Scrape MOSDAC data
scrape:
	python scripts/data/scrape_mosdac.py

# Process documents
process:
	python scripts/data/process_documents.py

# Monitor logs
logs:
	docker-compose logs -f

# Stop all services
stop:
	docker-compose down
