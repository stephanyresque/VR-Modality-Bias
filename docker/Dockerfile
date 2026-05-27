ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:23.10-py3
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /workspace/VR-Modality-Bias

COPY pyproject.toml README.md ./
COPY src ./src

RUN uv pip install --system -e ".[dev]"

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY Makefile ./

ENV PYTHONPATH=/workspace/VR-Modality-Bias/src

CMD ["bash"]
