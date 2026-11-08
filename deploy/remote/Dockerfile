# syntax=docker/dockerfile:1
# GPU worker for machinum-sft. Build is heavy (CUDA + torch + unsloth), ~10-12 GB.
# The repo is NOT copied in: it is bind-mounted at /workspace so the container always runs
# the pushed code. Only the Python stack lives in the image.

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf \
    PATH=/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git cmake python3.12 python3.12-venv python3.12-dev \
        build-essential libgomp1 tini \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /opt/msft

# Install the Python stack once, into an image-level venv. The repo is mounted later, so we
# install the package itself at container start (worker-entrypoint.sh) to pick up the mount.
RUN uv venv --python 3.12 /opt/venv \
    && . /opt/venv/bin/activate \
    && uv pip install --index-url https://download.pytorch.org/whl/cu128 torch \
    && uv pip install --index-url https://download.pytorch.org/whl/cu128 --no-deps torchvision==0.26.0 \
    && uv pip install unsloth trl peft datasets accelerate bitsandbytes pyyaml \
    && uv pip install "transformers>=5.2.0" \
    && python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    && python -c "import transformers; assert tuple(int(x) for x in transformers.__version__.split('.')[:2]) >= (5, 2), transformers.__version__"

# llama.cpp lives in the image, not in the container's writable layer: a `docker compose up`
# after any compose edit recreates the container and would otherwise throw away the build.
# Same reasoning as transformers above — anything a run needs must be in the image.
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
    && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/llama.cpp/build --config Release -j "$(nproc)" --target llama-quantize \
    && . /opt/venv/bin/activate \
    && uv pip install -r /opt/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    LLAMA_CPP_DIR=/opt/llama.cpp

RUN pip install --no-cache-dir "huggingface_hub[cli]" || true

COPY deploy/remote/worker-entrypoint.sh /usr/local/bin/worker-entrypoint.sh
RUN chmod +x /usr/local/bin/worker-entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/worker-entrypoint.sh"]
CMD ["python", "-m", "msft", "worker", "serve", "--dir", "/workspace/.worker", "--interval", "15"]
