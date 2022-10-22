# Created by: George Corrêa de Araújo (george.gcac@gmail.com)

# ==================================================================
# environment variables
# ------------------------------------------------------------------

DOCKERFILE_CONTEXT = $(PWD)
DOCKERFILE = $(PWD)/Dockerfile
WORK_DIR = $(PWD)
RUN_STRING = bash start_here.sh

# ==================================================================
# Docker settings
# ------------------------------------------------------------------

CONTAINER_NAME = flask-$(USER)-$(shell echo $$STY | cut -d'.' -f2)
CONTAINER_FILE = flask-$(USER).tar
HOSTNAME = docker-$(shell hostname)
IMAGE_NAME = $(USER)/flask
PORT = 5000
USERNAME = $(USER)
WORK_PATH = /work

RUN_CONFIG_STRING = --name $(CONTAINER_NAME) --hostname $(HOSTNAME) --rm -it --dns 8.8.8.8 \
	--userns=host --ipc=host --ulimit memlock=-1 -w $(WORK_PATH) -p $(PORT):5000 $(IMAGE_NAME):latest
WORK_MOUNT_STRING = --mount type=bind,source=$(WORK_DIR),target=$(WORK_PATH)

# ==================================================================
# Make commands
# ------------------------------------------------------------------

# Build image
build:
	docker build \
		--build-arg GROUPID=$(shell id -g) \
		--build-arg GROUPNAME=$(shell id -gn) \
		--build-arg USERID=$(shell id -u) \
		--build-arg USERNAME=$(USERNAME) \
		-f $(DOCKERFILE) \
		--pull --no-cache --force-rm \
		-t $(IMAGE_NAME) \
		$(DOCKERFILE_CONTEXT)

# Remove the image
clean:
	docker rmi $(IMAGE_NAME)


# Load image from file
load:
	docker load -i $(CONTAINER_FILE)


# Kill running container
kill:
	docker kill $(CONTAINER_NAME)


# Run RUN_STRING inside container
run:
	docker run \
		$(WORK_MOUNT_STRING) \
		$(RUN_CONFIG_STRING) \
		$(RUN_STRING)


# Save image to file
save:
	docker save -o $(CONTAINER_FILE) $(IMAGE_NAME)


# Start container by opening shell
start:
	docker run \
		$(WORK_MOUNT_STRING) \
		$(RUN_CONFIG_STRING)


# Test image by printing some info
test:
	docker run \
		$(RUN_CONFIG_STRING) \
		python -V

