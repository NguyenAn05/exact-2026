.PHONY: install dev mock test curl-type1 curl-type2 docker-build docker-up docker-down package

install:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

dev:
	uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

mock:
	MOCK_MODE=true uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

test:
	python -m py_compile app/*.py app/pipelines/*.py app/pipelines/graph/*.py app/utils/*.py
	PYTHONPATH=. pytest -q

curl-type1:
	bash scripts/test_type1.sh

curl-type2:
	bash scripts/test_type2.sh

docker-build:
	docker build -t exact2026-api:latest .

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

package:
	bash scripts/build_submission_package.sh my_team
