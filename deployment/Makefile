VERSION := v0.3.19-mnist-dev
IMAGE_NAME_BASE=pytorch-prepackaged
IMAGE_BASE=infuseaidev/${IMAGE_NAME_BASE}

build:
	docker build . -t ${IMAGE_BASE}:${VERSION}

push:
	docker push ${IMAGE_BASE}:${VERSION}

