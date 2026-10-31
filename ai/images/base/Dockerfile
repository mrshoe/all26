# The image every wildhat job runs: Python, PyTorch with CUDA, the simulator's
# packages, and the entrypoint that fetches the run's source and saves its
# outputs. Runs add their own packages with a requirements.txt.
#
# Python 3.11, because Isaac Sim needs it, and so does bpy 4.5 (Blender 4.5
# LTS; bpy 5 needs 3.13). PyPI's torch is built for CUDA 13.0, which Cloud
# Run's L4 driver (580) supports; without a GPU it runs on the CPU.
#
# The simulator's packages are pinned to match
# ai/src/simulator/requirements.txt. bpy 4.5 needs numpy 1.x, and pyarrow 26
# needs numpy 2.
#
# Terraform builds this (`wildhat admin build-image`) whenever a file in this
# folder changes, and tags it with a hash of the folder.
FROM python:3.11-slim-bookworm

# OpenCV, which ultralytics uses, needs libgl1 and libglib2.0-0. bpy needs
# the X11 and OpenGL libraries too, even though it renders headless.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
       libopengl0 libx11-6 libxext6 libxfixes3 libxi6 libxrender1 libxt6 \
       libxxf86vm1 libsm6 libice6 libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    torch==2.14.1 torchvision==0.29.1 ultralytics==8.4.187 \
    numpy==1.26.4 scipy==1.17.1 bpy==4.5.4 mujoco==3.17.0 pyarrow==25.0.0 \
    imageio-ffmpeg==0.7.0

COPY wildhat_run.py /usr/local/bin/wildhat-run
WORKDIR /work
ENTRYPOINT ["python", "/usr/local/bin/wildhat-run"]
