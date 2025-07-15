# =============================================================================
# MOSDAC AI Knowledge Navigator - Makefile
# =============================================================================
# Convenient commands for development, testing, and deployment

.DEFAULT_GOAL := help
.PHONY: help

# =============================================================================
# HELP
# =============================================================================
help: ## Show this help message
	@echo "MOSDAC AI Knowledge Navigator - Available Commands"
	@echo "================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
setup: ## Initial project setup
	@echo "Setting up MOSDAC AI Knowledge Navigator..."
	cp .env.example .env
	@echo "✅ Environment file created. Please update .env with your API keys."
	$(MAKE) install-deps
	@echo "🚀 Setup complete! Run 'make dev' to start development."

install-deps: ## Install all dependencies
	@echo "Installing backend dependencies..."
	cd backend && python -m pip install --upgrade pip
	cd backend && pip install -r requirements.txt
	cd backend && pip install -r requirements-dev.txt
	@echo "Installing frontend dependencies..."
	cd frontend && npm install
	@echo "✅ Dependencies installed."

install-backend: ## Install backend dependencies only
	cd backend && python -m venv venv
	cd backend && source venv/bin/activate && pip install --upgrade pip
	cd backend && source venv/bin/activate && pip install -r requirements.txt
	cd backend && source venv/bin/activate && pip install -r requirements-dev.txt

install-frontend: ## Install frontend dependencies only
	cd frontend && npm install

# =============================================================================
# DEVELOPMENT
# =============================================================================
dev: ## Start development environment
	docker-compose up -d postgres redis neo4j
	@echo "🚀 Starting development servers..."
	@echo "Backend will be available at http://localhost:8000"
	@echo "Frontend will be available at http://localhost:3000"
	@echo "Press Ctrl+C to stop"
	$(MAKE) dev-backend & $(MAKE) dev-frontend

dev-backend: ## Start backend development server
	cd backend && source venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend: ## Start frontend development server
	cd frontend && npm run dev

dev-full: ## Start full development environment with Docker
	docker-compose -f docker-compose.dev.yml up

dev-setup: ## Setup development environment
	@echo "Setting up development environment..."
	$(MAKE) install-deps
	$(MAKE) db-setup
	$(MAKE) data-seed
	@echo "✅ Development environment ready!"

# =============================================================================
# DATABASE
# =============================================================================
db-setup: ## Setup database and run migrations
	docker-compose up -d postgres redis neo4j
	@echo "Waiting for databases to be ready..."
	sleep 10
	cd backend && source venv/bin/activate && alembic upgrade head
	@echo "✅ Database setup complete."

db-migrate: ## Create new database migration
	cd backend && source venv/bin/activate && alembic revision --autogenerate -m "$(message)"

db-upgrade: ## Run database migrations
	cd backend && source venv/bin/activate && alembic upgrade head

db-downgrade: ## Rollback database migration
	cd backend && source venv/bin/activate && alembic downgrade -1

db-reset: ## Reset database (WARNING: destroys all data)
	docker-compose down
	docker volume rm mosdac-ai-knowledge-navigator_postgres_data || true
	docker volume rm mosdac-ai-knowledge-navigator_neo4j_data || true
	docker-compose up -d postgres redis neo4j
	sleep 10
	$(MAKE) db-upgrade
	@echo "⚠️  Database reset complete. All data has been lost."

# =============================================================================
# DATA MANAGEMENT
# =============================================================================
data-scrape: ## Scrape data from MOSDAC portal
	cd backend && source venv/bin/activate && python scripts/data_ingestion/scrape_mosdac.py

data-process: ## Process scraped data
	cd backend && source venv/bin/activate && python scripts/data_ingestion/process_documents.py

data-build-kg: ## Build knowledge graph
	cd backend && source venv/bin/activate && python scripts/data_ingestion/build_knowledge_graph.py

data-seed: ## Seed database with sample data
	$(MAKE) data-scrape
	$(MAKE) data-process
	$(MAKE) data-build-kg
	@echo "✅ Sample data seeded."

data-update: ## Update data from MOSDAC portal
	cd backend && source venv/bin/activate && python scripts/maintenance/update_data.py

# =============================================================================
# TESTING
# =============================================================================
test: ## Run all tests
	$(MAKE) test-backend
	$(MAKE) test-frontend

test-backend: ## Run backend tests
	cd backend && source venv/bin/activate && pytest -v

test-backend-cov: ## Run backend tests with coverage
	cd backend && source venv/bin/activate && pytest --cov=app --cov-report=html --cov-report=term

test-frontend: ## Run frontend tests
	cd frontend && npm test -- --watchAll=false

test-frontend-cov: ## Run frontend tests with coverage
	cd frontend && npm run test:coverage

test-e2e: ## Run end-to-end tests
	cd frontend && npm run test:e2e

test-integration: ## Run integration tests
	docker-compose -f docker-compose.test.yml up -d
	cd backend && source venv/bin/activate && pytest tests/integration/
	docker-compose -f docker-compose.test.yml down

# =============================================================================
# CODE QUALITY
# =============================================================================
lint: ## Run linting for all code
	$(MAKE) lint-backend
	$(MAKE) lint-frontend

lint-backend: ## Run backend linting
	cd backend && source venv/bin/activate && black --check .
	cd backend && source venv/bin/activate && isort --check-only .
	cd backend && source venv/bin/activate && flake8 .
	cd backend && source venv/bin/activate && mypy .

lint-frontend: ## Run frontend linting
	cd frontend && npm run lint

format: ## Format all code
	$(MAKE) format-backend
	$(MAKE) format-frontend

format-backend: ## Format backend code
	cd backend && source venv/bin/activate && black .
	cd backend && source venv/bin/activate && isort .

format-frontend: ## Format frontend code
	cd frontend && npm run format

type-check: ## Run type checking
	cd backend && source venv/bin/activate && mypy .
	cd frontend && npm run type-check

# =============================================================================
# SECURITY
# =============================================================================
security-check: ## Run security checks
	cd backend && source venv/bin/activate && bandit -r app/
	cd frontend && npm audit --audit-level=moderate

security-fix: ## Fix security vulnerabilities
	cd backend && source venv/bin/activate && pip-audit --fix
	cd frontend && npm audit fix

# =============================================================================
# DOCKER OPERATIONS
# =============================================================================
docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start all services with Docker
	docker-compose up -d

docker-down: ## Stop all Docker services
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

docker-clean: ## Clean Docker resources
	docker-compose down -v
	docker system prune -f
	docker volume prune -f

docker-rebuild: ## Rebuild Docker images from scratch
	docker-compose down -v
	docker-compose build --no-cache
	docker-compose up -d

# =============================================================================
# PRODUCTION
# =============================================================================
build: ## Build for production
	$(MAKE) build-backend
	$(MAKE) build-frontend

build-backend: ## Build backend for production
	cd backend && source venv/bin/activate && pip install --no-dev

build-frontend: ## Build frontend for production
	cd frontend && npm run build

deploy-prod: ## Deploy to production
	docker-compose -f docker-compose.prod.yml up -d --build

deploy-staging: ## Deploy to staging
	docker-compose -f docker-compose.staging.yml up -d --build

health-check: ## Check application health
	curl -f http://localhost:8000/health || exit 1
	curl -f http://localhost:3000 || exit 1

# =============================================================================
# MONITORING
# =============================================================================
logs: ## View application logs
	docker-compose logs -f backend frontend

logs-backend: ## View backend logs
	docker-compose logs -f backend

logs-frontend: ## View frontend logs
	docker-compose logs -f frontend

metrics: ## View application metrics
	curl http://localhost:8000/metrics

monitor: ## Start monitoring stack
	docker-compose up -d prometheus grafana elasticsearch kibana

# =============================================================================
# CLEANUP
# =============================================================================
clean: ## Clean all generated files
	$(MAKE) clean-backend
	$(MAKE) clean-frontend
	$(MAKE) clean-data

clean-backend: ## Clean backend artifacts
	cd backend && find . -type d -name "__pycache__" -exec rm -rf {} +
	cd backend && find . -type f -name "*.pyc" -delete
	cd backend && find . -type f -name "*.pyo" -delete
	cd backend && find . -type f -name ".coverage" -delete
	cd backend && rm -rf htmlcov/
	cd backend && rm -rf .pytest_cache/
	cd backend && rm -rf .mypy_cache/

clean-frontend: ## Clean frontend artifacts
	cd frontend && rm -rf node_modules/
	cd frontend && rm -rf dist/
	cd frontend && rm -rf build/
	cd frontend && rm -rf .cache/
	cd frontend && rm -f package-lock.json

clean-data: ## Clean data files
	rm -rf data/raw/*
	rm -rf data/processed/*
	rm -rf data/temp/*
	rm -rf logs/*.log

clean-docker: ## Clean Docker resources
	docker-compose down -v --remove-orphans
	docker system prune -af
	docker volume prune -f

# =============================================================================
# UTILITIES
# =============================================================================
shell-backend: ## Open backend shell
	cd backend && source venv/bin/activate && python

shell-db: ## Open database shell
	docker-compose exec postgres psql -U postgres -d mosdac_db

shell-redis: ## Open Redis shell
	docker-compose exec redis redis-cli

shell-neo4j: ## Open Neo4j shell
	docker-compose exec neo4j cypher-shell -u neo4j -p password

backup: ## Backup data
	./scripts/deployment/backup.sh

restore: ## Restore data from backup
	./scripts/deployment/restore.sh

docs: ## Generate documentation
	cd backend && source venv/bin/activate && sphinx-build -b html docs/ docs/_build/
	cd frontend && npm run build-storybook

docs-serve: ## Serve documentation locally
	cd backend/docs/_build && python -m http.server 8080

pre-commit: ## Run pre-commit checks
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) security-check

ci: ## Run CI pipeline locally
	$(MAKE) install-deps
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) security-check
	$(MAKE) build

# =============================================================================
# DEVELOPMENT HELPERS
# =============================================================================
jupyter: ## Start Jupyter notebook
	docker-compose up -d jupyter
	@echo "Jupyter available at http://localhost:8888"
	@echo "Token: mosdac-jupyter-token"

notebook: ## Start Jupyter notebook locally
	cd notebooks && source ../backend/venv/bin/activate && jupyter lab

install-hooks: ## Install Git hooks
	cd backend && source venv/bin/activate && pre-commit install

update-deps: ## Update all dependencies
	cd backend && source venv/bin/activate && pip-tools compile --upgrade requirements.in
	cd frontend && npm update

check-deps: ## Check for dependency updates
	cd backend && source venv/bin/activate && pip list --outdated
	cd frontend && npm outdated

# =============================================================================
# TEAM COLLABORATION
# =============================================================================
team-setup: ## Setup for new team member
	@echo "Welcome to MOSDAC AI Knowledge Navigator! 👋"
	@echo "Setting up your development environment..."
	$(MAKE) setup
	$(MAKE) install-hooks
	@echo "Installing VS Code extensions..."
	code --install-extension ms-python.python
	code --install-extension ms-vscode.vscode-typescript-next
	code --install-extension esbenp.prettier-vscode
	code --install-extension ms-vscode.vscode-eslint
	code --install-extension ms-vsliveshare.vsliveshare
	@echo "✅ Team setup complete!"
	@echo "📖 Next steps:"
	@echo "   1. Update .env with your API keys"
	@echo "   2. Run 'make dev' to start development"
	@echo "   3. Visit http://localhost:3000 to see the app"

sync: ## Sync with remote repository
	git fetch origin
	git pull origin main
	$(MAKE) install-deps
	$(MAKE) db-upgrade

# =============================================================================
# TROUBLESHOOTING
# =============================================================================
doctor: ## Diagnose common issues
	@echo "🔍 Running diagnostics..."
	@echo "Checking Docker..."
	docker --version || echo "❌ Docker not found"
	@echo "Checking Docker Compose..."
	docker-compose --version || echo "❌ Docker Compose not found"
	@echo "Checking Python..."
	python --version || echo "❌ Python not found"
	@echo "Checking Node.js..."
	node --version || echo "❌ Node.js not found"
	@echo "Checking npm..."
	npm --version || echo "❌ npm not found"
	@echo "Checking environment file..."
	test -f .env && echo "✅ .env file exists" || echo "❌ .env file missing"
	@echo "Checking backend virtual environment..."
	test -d backend/venv && echo "✅ Backend venv exists" || echo "❌ Backend venv missing"
	@echo "Checking frontend node_modules..."
	test -d frontend/node_modules && echo "✅ Frontend dependencies installed" || echo "❌ Frontend dependencies missing"
	@echo "🏥 Diagnosis complete"

fix-permissions: ## Fix file permissions
	chmod +x scripts/deployment/*.sh
	chmod +x scripts/setup/*.sh
	chmod +x scripts/maintenance/*.py

reset: ## Reset entire development environment
	@echo "⚠️  This will reset your entire development environment!"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read
	$(MAKE) clean
	$(MAKE) docker-clean
	$(MAKE) db-reset
	$(MAKE) setup
	@echo "🔄 Environment reset complete"