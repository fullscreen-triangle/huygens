# Universal Oscillatory Framework for Cardiovascular Analysis
# Development Automation Makefile

.PHONY: help install install-dev test test-fast test-coverage lint format clean docs build publish check-all validate setup-dev demo

# Default Python version
PYTHON := python3
PIP := pip
PYTEST := pytest

# Colors for output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

help: ## Show this help message
	@echo "$(BLUE)Universal Oscillatory Framework for Cardiovascular Analysis$(RESET)"
	@echo "$(BLUE)Development Automation Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

install: ## Install package in production mode
	@echo "$(BLUE)Installing Universal Oscillatory Framework...$(RESET)"
	$(PIP) install -e .

install-dev: ## Install package in development mode with all dependencies
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	$(PIP) install -e ".[dev,docs,gpu,ml]"
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(RESET)"

setup-dev: install-dev ## Complete development environment setup
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	@mkdir -p tests/unit tests/integration tests/fixtures
	@mkdir -p docs/source/_static docs/source/_templates
	@mkdir -p logs tmp results
	@echo "$(GREEN)Development environment setup complete!$(RESET)"

test: ## Run full test suite
	@echo "$(BLUE)Running full test suite...$(RESET)"
	$(PYTEST) tests/ demo/tests/ -v --cov=src --cov=demo --cov-report=html --cov-report=term

test-fast: ## Run fast tests only (excluding slow/integration tests)
	@echo "$(BLUE)Running fast tests...$(RESET)"
	$(PYTEST) tests/ demo/tests/ -v -m "not slow and not integration"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	$(PYTEST) tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(PYTEST) tests/integration/ -v

test-coverage: ## Generate detailed coverage report
	@echo "$(BLUE)Generating coverage report...$(RESET)"
	$(PYTEST) tests/ demo/tests/ --cov=src --cov=demo --cov-report=html --cov-report=xml --cov-report=term-missing

lint: ## Run all linters
	@echo "$(BLUE)Running linters...$(RESET)"
	flake8 src/ demo/ tests/
	mypy src/
	bandit -r src/ -f json -o bandit-report.json || true
	@echo "$(GREEN)Linting complete!$(RESET)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	black src/ demo/ tests/
	isort src/ demo/ tests/
	@echo "$(GREEN)Code formatting complete!$(RESET)"

format-check: ## Check code formatting without making changes
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	black --check src/ demo/ tests/
	isort --check-only src/ demo/ tests/

pre-commit-all: ## Run pre-commit on all files
	@echo "$(BLUE)Running pre-commit hooks on all files...$(RESET)"
	pre-commit run --all-files

clean: ## Clean up build artifacts and cache files
	@echo "$(BLUE)Cleaning up...$(RESET)"
	rm -rf build/ dist/ *.egg-info/ .eggs/
	rm -rf .pytest_cache/ .mypy_cache/ .coverage htmlcov/
	rm -rf .tox/ .nox/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf tmp/ logs/*.log results/
	@echo "$(GREEN)Cleanup complete!$(RESET)"

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(RESET)"
	cd docs && make html
	@echo "$(GREEN)Documentation built! Open docs/_build/html/index.html$(RESET)"

docs-clean: ## Clean documentation build files
	@echo "$(BLUE)Cleaning documentation...$(RESET)"
	cd docs && make clean

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation locally...$(RESET)"
	@echo "$(YELLOW)Open http://localhost:8000 in your browser$(RESET)"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

build: clean ## Build source and wheel distribution
	@echo "$(BLUE)Building distributions...$(RESET)"
	$(PYTHON) -m build
	@echo "$(GREEN)Build complete! Check dist/ directory$(RESET)"

publish-test: build ## Publish to TestPyPI
	@echo "$(BLUE)Publishing to TestPyPI...$(RESET)"
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI (production)
	@echo "$(YELLOW)Publishing to PyPI (production)...$(RESET)"
	@echo "$(RED)Are you sure? This will publish to the main PyPI index.$(RESET)"
	@read -p "Type 'yes' to continue: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		$(PYTHON) -m twine upload dist/*; \
		echo "$(GREEN)Published to PyPI!$(RESET)"; \
	else \
		echo "$(YELLOW)Publish cancelled.$(RESET)"; \
	fi

check-all: ## Run all checks (lint, test, format-check)
	@echo "$(BLUE)Running all checks...$(RESET)"
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) test
	@echo "$(GREEN)All checks passed!$(RESET)"

validate: ## Validate oscillatory framework implementation
	@echo "$(BLUE)Validating oscillatory framework...$(RESET)"
	$(PYTHON) -c "from src.cardiovascular_oscillatory_suite import UniversalCardiovascularFramework; print('✓ Framework imports successfully')"
	$(PYTHON) -m src.validate_framework
	@echo "$(GREEN)Framework validation complete!$(RESET)"

demo: ## Run comprehensive demo
	@echo "$(BLUE)Running comprehensive demo...$(RESET)"
	$(PYTHON) demo/run_comprehensive_demo.py
	@echo "$(GREEN)Demo complete! Check results directory$(RESET)"

demo-cardiovascular: ## Run cardiovascular analysis demo
	@echo "$(BLUE)Running cardiovascular analysis demo...$(RESET)"
	$(PYTHON) analyze_cardiovascular_data.py --demo
	@echo "$(GREEN)Cardiovascular demo complete!$(RESET)"

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(RESET)"
	$(PYTHON) -m pytest benchmarks/ -v --benchmark-only
	@echo "$(GREEN)Benchmarks complete!$(RESET)"

profile: ## Profile code performance
	@echo "$(BLUE)Profiling code performance...$(RESET)"
	$(PYTHON) -m cProfile -o profile_results.prof src/cardiovascular_oscillatory_suite.py
	@echo "$(GREEN)Profiling complete! Results in profile_results.prof$(RESET)"

security: ## Run security analysis
	@echo "$(BLUE)Running security analysis...$(RESET)"
	bandit -r src/ demo/ -f json -o bandit-report.json
	safety check
	@echo "$(GREEN)Security analysis complete!$(RESET)"

deps-update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(RESET)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -r requirements.txt
	pre-commit autoupdate
	@echo "$(GREEN)Dependencies updated!$(RESET)"

deps-check: ## Check for dependency vulnerabilities
	@echo "$(BLUE)Checking dependencies for vulnerabilities...$(RESET)"
	pip-audit
	@echo "$(GREEN)Dependency check complete!$(RESET)"

install-tools: ## Install development tools
	@echo "$(BLUE)Installing development tools...$(RESET)"
	$(PIP) install build twine pip-audit
	@echo "$(GREEN)Development tools installed!$(RESET)"

# Scientific validation targets
validate-science: ## Validate scientific implementation
	@echo "$(BLUE)Validating scientific implementation...$(RESET)"
	$(PYTHON) -m src.scientific_validation
	@echo "$(GREEN)Scientific validation complete!$(RESET)"

test-oscillatory: ## Test oscillatory framework specifically
	@echo "$(BLUE)Testing oscillatory framework...$(RESET)"
	$(PYTEST) tests/unit/test_oscillatory_framework.py -v
	$(PYTEST) tests/integration/test_cardiovascular_analysis.py -v
	@echo "$(GREEN)Oscillatory framework tests complete!$(RESET)"

# Docker targets
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(RESET)"
	docker build -t cardiovascular-oscillatory-framework .
	@echo "$(GREEN)Docker image built!$(RESET)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(RESET)"
	docker run -it --rm -v $(PWD)/data:/app/data cardiovascular-oscillatory-framework

# Development workflow shortcuts
quick-check: format lint test-fast ## Quick development checks
	@echo "$(GREEN)Quick checks complete!$(RESET)"

release-check: check-all docs security ## Pre-release validation
	@echo "$(GREEN)Release checks complete!$(RESET)"

# Help target should be first
.DEFAULT_GOAL := help
