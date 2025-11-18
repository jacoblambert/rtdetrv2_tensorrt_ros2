#!/bin/bash
docker run \
-it --rm --name=rtdetrv2-training --privileged \
--gpus all \
--env DISPLAY=${env:DISPLAY} \
--env XAUTHORITY=/run/user/1000/gdm/Xauthority \
--env XDG_RUNTIME_DIR=/run/user/1000" \
--env "NVIDIA_DRIVER_CAPABILITIES=all \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
--net=host \
--volume=.:/workspace:rw \
--volume=/tmp/.X11-unix/:/tmp/.X11-unix \
--volume=/etc/localtime:/etc/localtime:ro \
--volume=/dev/dri:/dev/dri:ro \
--volume=/mnt/qnapdata/external/bdd100k:/data/:ro \
--shm-size=64gb --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
rtdetrv2:latest bash