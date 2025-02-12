SHELL := /bin/bash
VERSION ?= 0.0.1
IMAGE_TAG ?= quay.io/jianrzha/label-studio-ml-backend
IMG ?= $(IMAGE_TAG):$(VERSION)

install:
	pip install -r requirements.txt	

build:
	podman build -f label_studio_ml/examples/huggingface_llm/Dockerfile -t ${IMG} .
push: ## Push docker image with the manager.
	podman push ${IMG}

test:
	pytest tests
