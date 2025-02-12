SHELL := /bin/bash
VERSION ?= 0.0.11
IMAGE_TAG ?= quay.io/jianrzha/label-studio-ml-backend
IMG ?= $(IMAGE_TAG):$(VERSION)

all: build push deploy

install:
	pip install -r requirements.txt	

build:
	podman build  --platform=linux/amd64 -t ${IMG} .
push: ## Push docker image with the manager.
	podman push ${IMG}

deploy:
	oc delete -f deployment.yaml
	oc apply -f deployment.yaml

test:
	pytest tests
