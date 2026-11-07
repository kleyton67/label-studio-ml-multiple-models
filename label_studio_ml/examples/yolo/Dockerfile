FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
ARG DEBIAN_FRONTEND=noninteractive
ARG TEST_ENV

WORKDIR /app

RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    apt-get -y update \
    && apt-get install -y --no-install-recommends \
        git wget curl ffmpeg libsm6 libxext6 libffi-dev \
        python3-dev gcc g++ libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4 \
    CUDA_HOME=/usr/local/cuda

ENV CUDA_HOME=/opt/conda \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;9.0"

# install base requirements
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# install model requirements
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip3 install -r requirements.txt

# install test requirements if needed
COPY requirements-test.txt .
# build only when TEST_ENV="true"
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip3 install -r requirements-test.txt; \
    fi

WORKDIR /app

COPY . ./

# Download the YOLO models
RUN mkdir -p /app/models && \
    /bin/sh -c 'if [ ! -f /app/models/yolov8m.pt ]; then \
    yolo predict model=/app/models/yolov8m.pt source=/app/tests/car.jpg \
    && yolo predict model=/app/models/yolov8n.pt source=/app/tests/car.jpg \
    && yolo predict model=/app/models/yolov8n-cls.pt source=/app/tests/car.jpg \
    && yolo predict model=/app/models/yolov8n-seg.pt source=/app/tests/car.jpg; \
    fi'

ENV PYTHONPATH=/app

CMD ["/app/start.sh"]
