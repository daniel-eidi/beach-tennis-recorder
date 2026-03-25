# Beach Tennis Recorder - Vision Module (AGENT-02)
# Common development commands

.PHONY: setup setup-pretrained validate benchmark test train export \
        export-tflite export-onnx youtube-test dataset-analysis lint clean help

PYTHON ?= python
PIP ?= pip
MODELS_DIR ?= models
DATASET_DIR ?= dataset
RESULTS_DIR ?= validation_results
BENCHMARK_DIR ?= benchmark_results
CHANNEL ?= https://www.youtube.com/@btcanallive

help:  ## Show this help
	@echo "Beach Tennis Recorder - Vision Module (AGENT-02)"
	@echo ""
	@echo "Available targets:"
	@echo "  make setup             Install Python dependencies"
	@echo "  make setup-pretrained  Download and export pre-trained YOLOv8n model"
	@echo "  make validate          Run full end-to-end pipeline validation"
	@echo "  make benchmark         Run model performance benchmarks"
	@echo "  make test              Run all pytest unit tests"
	@echo "  make youtube-test      Run YouTube test with default channel"
	@echo "  make train             Run YOLOv8 training on annotated dataset"
	@echo "  make export            Export model to TFLite + ONNX"
	@echo "  make export-tflite     Export model to TFLite only"
	@echo "  make export-onnx       Export model to ONNX only"
	@echo "  make dataset-analysis  Analyze dataset quality"
	@echo "  make lint              Run basic code quality checks"
	@echo "  make clean             Remove generated files"

setup:  ## Install Python dependencies
	$(PIP) install -r requirements.txt

setup-pretrained:  ## Download and export pre-trained YOLOv8n model
	$(PYTHON) -m vision.scripts.setup_pretrained --output-dir $(MODELS_DIR)

validate:  ## Run full end-to-end pipeline validation
	$(PYTHON) -m vision.scripts.validate_pipeline \
		--model $(MODELS_DIR)/best.tflite \
		--output-dir $(RESULTS_DIR) \
		--channel $(CHANNEL)

benchmark:  ## Run model performance benchmarks
	$(PYTHON) -m vision.scripts.benchmark_model \
		--model $(MODELS_DIR)/best.tflite \
		--output-dir $(BENCHMARK_DIR) \
		--iterations 100

test:  ## Run all pytest unit tests
	$(PYTHON) -m pytest tests/ -v --tb=short

youtube-test:  ## Run YouTube test with default channel
	$(PYTHON) -m vision.utils.youtube_test \
		--channel $(CHANNEL) \
		--max-videos 1 \
		--model $(MODELS_DIR)/best.tflite \
		--duration 60

train:  ## Run YOLOv8 training on annotated dataset
	$(PYTHON) train.py --config $(DATASET_DIR)/data.yaml --model yolov8n.pt

export: export-tflite export-onnx  ## Export model to TFLite + ONNX

export-tflite:  ## Export model to TFLite
	$(PYTHON) export_tflite.py --model $(MODELS_DIR)/best.pt

export-onnx:  ## Export model to ONNX
	$(PYTHON) export_onnx.py --model $(MODELS_DIR)/best.pt

dataset-analysis:  ## Analyze dataset quality
	$(PYTHON) -m vision.utils.dataset_analyzer \
		--data-yaml $(DATASET_DIR)/data.yaml \
		--output-dir dataset_analysis

lint:  ## Run basic code quality checks
	$(PYTHON) -m py_compile train.py
	$(PYTHON) -m py_compile export_tflite.py
	$(PYTHON) -m py_compile export_onnx.py
	$(PYTHON) -m py_compile inference_test.py
	$(PYTHON) -m py_compile tracking/byte_tracker.py
	$(PYTHON) -m py_compile tracking/rally_detector.py
	$(PYTHON) -m py_compile tracking/ball_physics.py
	$(PYTHON) -m py_compile utils/frame_extractor.py
	$(PYTHON) -m py_compile utils/visualization.py
	$(PYTHON) -m py_compile utils/youtube_download.py
	$(PYTHON) -m py_compile utils/dataset_analyzer.py
	@echo "All files compiled successfully."

clean:  ## Remove generated files
	rm -rf $(RESULTS_DIR) $(BENCHMARK_DIR) dataset_analysis
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache tests/.pytest_cache
	rm -rf runs/
	rm -f debug_visualization.jpg
	@echo "Cleaned generated files."
