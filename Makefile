# Beach Tennis Recorder — Makefile
# AGENT-05 | Sprint 3
#
# Targets for running tests, generating mocks, and validating the pipeline.

PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest
PROJECT_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

.PHONY: setup
setup: ## Install all dependencies (backend + vision + video + tests)
	cd $(PROJECT_ROOT) && $(PYTHON) -m pip install -r tests/requirements.txt
	cd $(PROJECT_ROOT) && $(PYTHON) -m pip install -r backend/requirements.txt 2>/dev/null || true
	cd $(PROJECT_ROOT) && $(PYTHON) -m pip install numpy opencv-python ffmpeg-python 2>/dev/null || true

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

.PHONY: test-all
test-all: ## Run all tests (video + vision + integration)
	cd $(PROJECT_ROOT) && $(PYTEST) -v --timeout=120 tests/ video/tests/

.PHONY: test-video
test-video: ## Run video module tests
	cd $(PROJECT_ROOT) && $(PYTEST) -v --timeout=60 video/tests/

.PHONY: test-vision
test-vision: ## Run vision module tests (if any)
	cd $(PROJECT_ROOT) && $(PYTEST) -v --timeout=60 vision/ -k "test_" 2>/dev/null || echo "No vision tests found"

.PHONY: test-integration
test-integration: ## Run integration tests only
	cd $(PROJECT_ROOT) && $(PYTEST) -v --timeout=120 tests/integration/

.PHONY: test-detection
test-detection: ## Run detection-to-rally tests
	cd $(PROJECT_ROOT) && $(PYTEST) -v --timeout=60 tests/integration/test_detection_to_rally.py

.PHONY: test-tracking
test-tracking: ## Run tracking pipeline tests
	cd $(PROJECT_ROOT) && $(PYTEST) -v --timeout=60 tests/integration/test_tracking_pipeline.py

.PHONY: test-clip
test-clip: ## Run clip pipeline tests
	cd $(PROJECT_ROOT) && $(PYTEST) -v --timeout=120 tests/integration/test_clip_pipeline.py

.PHONY: test-full
test-full: ## Run full pipeline tests
	cd $(PROJECT_ROOT) && $(PYTEST) -v --timeout=120 tests/integration/test_full_pipeline.py

.PHONY: test-video-processing
test-video-processing: ## Run video processing integration tests
	cd $(PROJECT_ROOT) && $(PYTEST) -v --timeout=120 tests/integration/test_video_processing.py

# ---------------------------------------------------------------------------
# Mock data generation
# ---------------------------------------------------------------------------

.PHONY: generate-mocks
generate-mocks: ## Generate mock videos for testing
	cd $(PROJECT_ROOT) && $(PYTHON) tests/mock_data/generate_mock_video.py --scenario all --duration 10 --output-dir tests/mock_data/videos/

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

.PHONY: validate
validate: ## Run vision validation pipeline (requires model + video)
	cd $(PROJECT_ROOT) && $(PYTHON) -m vision.scripts.validate_pipeline

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
