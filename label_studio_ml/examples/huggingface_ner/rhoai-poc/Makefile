SHELL := /bin/bash
VERSION ?= 0.0.10
IMAGE_TAG ?= quay.io/jianrzha/huggingface-ner-ml-backend
IMG ?= $(IMAGE_TAG):$(VERSION)

all: build push deploy

install:
	pip install -r ../requirements.txt	

build:
	podman build  --platform=linux/amd64 -f ../Dockerfile -t ${IMG} .
push: ## Push docker image with the manager.
	podman push ${IMG}

deploy:
	oc delete -f deployment.yaml
	oc apply -f deployment.yaml

test:
	pytest tests
