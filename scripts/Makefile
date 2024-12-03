all: help

install:
	@pip install -r requirements.txt

update-tools:
	@./bin/update-tools.sh

test:
	@make test-unit
	@make test-integration

test-only:
	@echo
	@echo
	@echo "Tests Marked as \"only\""
	@python -m pytest -s -m only
	@echo
	@echo

test-unit:
	@echo
	@echo
	@echo Unit Tests
	@python -m pytest -s -m unit
	@echo
	@echo

test-integration:
	@echo
	@echo
	@echo Integration Tests
	@python -m pytest -s -m integration
	@echo
	@echo

help:
	@echo
	@echo GraphQL Analyzer
	@echo ----------------------
	@echo "make install             Install the dependencies."
	@echo "make update-tools        Update analyzer_tools."
	@echo "make test                Run all the test suites."
	@echo "make test-only           Run all the tests marked as \"only\"."
	@echo "make test-unit           Run all the unit tests."
	@echo "make test-integration    Run all the integration tests."
	@echo

.PHONY: install update-tools test test-only test-unit test-integration
