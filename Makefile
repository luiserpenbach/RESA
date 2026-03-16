.PHONY: dev-api dev-web dev build install-api install-web test-api lint format clean

install-api:
	pip install -e ".[dev]"

install-web:
	cd web && npm install

dev-api:
	uvicorn api.main:app --reload --port 8000

dev-web:
	cd web && npm run dev

dev:
	$(MAKE) -j2 dev-api dev-web

build:
	cd web && npm run build

test-api:
	pytest api/tests/ -v

lint:
	ruff check .
	black --check .

format:
	black .
	ruff check --fix .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
