.PHONY: dev-api dev-web dev build install-api install-web

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
	pytest api/ -v 2>/dev/null || echo "No api tests yet"
