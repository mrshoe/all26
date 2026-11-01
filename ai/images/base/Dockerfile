# The image every wildhat job runs: Python, PyTorch with CUDA, and the
# entrypoint that fetches the run's source and saves its outputs. Runs add
# their own packages with a requirements.txt.
#
# Python 3.11, because Isaac Sim needs it. PyPI's torch is built for CUDA 13.0,
# which Cloud Run's L4 driver (580) supports; without a GPU it runs on the CPU.
#
# Terraform builds this (`wildhat admin build-image`) whenever a file in this
# folder changes, and tags it with a hash of the folder.
FROM python:3.11-slim-bookworm

# OpenCV, which ultralytics uses, needs these.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    torch==2.14.1 torchvision==0.29.1 ultralytics==8.4.187

COPY wildhat_run.py /usr/local/bin/wildhat-run
WORKDIR /work
ENTRYPOINT ["python", "/usr/local/bin/wildhat-run"]
