# Hugging Face Space (Docker SDK). turbovec is a Rust extension, so we build
# the Python wheel from source at image-build time, then run the Gradio app.
FROM rust:1.82-slim-bookworm

# System deps: Python + BLAS (turbovec links OpenBLAS on Linux) + build tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        libopenblas-dev pkg-config patchelf git \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run as a non-root user; set up a writable home.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    MPLCONFIGDIR=/tmp/mpl
USER user
WORKDIR /home/user/app

# Python deps first (cached across app.py edits).
RUN pip3 install --user --no-cache-dir \
        "maturin[patchelf]>=1.12,<2.0" numpy gradio matplotlib

# Build + install turbovec (with the fp16 refine mode) from the repo.
# The whole repo is cloned so the path-dependency `../turbovec` resolves.
RUN pip3 install --user --no-cache-dir \
    "git+https://github.com/aabhimittal/turbovec-test.git@main#subdirectory=turbovec-python"

COPY --chown=user app.py .

EXPOSE 7860
CMD ["python3", "app.py"]
