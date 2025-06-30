SHELL := /bin/bash
VERSION ?= 0.0.8
IMAGE_TAG ?= quay.io/jianrzha/ner
IMG ?= $(IMAGE_TAG):$(VERSION)

all: build push deploy

install:
	pip install -r ./requirements.txt	

build:
	podman build  --platform=linux/amd64 -f ./Containerfile -t ${IMG} .
push: ## Push docker image with the manager.
	podman push ${IMG}

deploy-servingruntime:
	- oc delete -f CustomServingRuntime.yaml
	oc apply -f CustomServingRuntime.yaml
