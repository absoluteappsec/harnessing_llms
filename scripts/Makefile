all: help

install:
	@pip install -r requirements.txt

test:
	@make test-integration

test-integration:
	@echo
	@echo
	@echo Integration Tests
	@python -m pytest -s -m integration
	@echo
	@echo

help:
	@echo
	@echo Agentic Testing Script
	@echo ----------------------
	@echo "make test                Run all the test suites."
	@echo "make test-integration    Run all the integration tests."
	@echo

.PHONY: install update-tools test test-only test-unit test-integration
