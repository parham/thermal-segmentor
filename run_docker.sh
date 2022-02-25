
# For building the docker

docker build --pull --rm -f "Dockerfile" -t thermal-segmentor:latest "."

# For running the docker

docker run --rm -it --init \
    --gpus=all \
    --ipc=host \
    --volume="$PWD/datasets:/phm/datasets" \
    --volume="$PWD/results:/phm/results" \
    thermal-segmentor:latest
