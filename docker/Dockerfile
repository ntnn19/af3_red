# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

# Blackwell port: CUDA 12.9 host image, Python 3.12, and pip CUDA libraries.
FROM docker.io/nvidia/cuda:12.9.0-base-ubuntu24.04

# Some RUN statements are combined together to make Docker build run faster.
# Get latest package listing, install Python, git, wget,
# compilers and libraries.
# git is required for pyproject.toml toolchain's use of CMakeLists.txt.
# gcc, g++, make are required for compiling hmmer and AlphaFold 3 libaries.
# zlib is a required dependency of AlphaFold 3.
RUN apt-get update --quiet \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet \
       ca-certificates git wget gcc g++ make zlib1g-dev zstd \
       python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Use Ubuntu's Python 3.12 and a version-pinned package installer.
COPY --from=ghcr.io/astral-sh/uv:0.9.24 /uv /uvx /bin/
RUN python3.12 -m venv /alphafold3_venv
ENV PATH="/hmmer/bin:/alphafold3_venv/bin:$PATH"
# Keep image size down and bound native build parallelism.
ENV UV_NO_CACHE=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=4

# Install HMMER. Do so before copying the source code, so that docker can cache
# the image layer containing HMMER. Alternatively, you could also install it 
# using `apt-get install hmmer` instead of bulding it from source.
RUN mkdir /hmmer_build /hmmer \
    && wget https://eddylab.org/software/hmmer/hmmer-3.4.tar.gz --directory-prefix /hmmer_build \
    && (cd /hmmer_build && tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz) \
    && (cd /hmmer_build/hmmer-3.4 && ./configure --prefix /hmmer) \
    && (cd /hmmer_build/hmmer-3.4 && make -j4) \
    && (cd /hmmer_build/hmmer-3.4 && make install) \
    && (cd /hmmer_build/hmmer-3.4/easel && make install) \
    && rm -R /hmmer_build

# Copy the AlphaFold 3 source code from the local machine to the container and
# set the working directory to there.
COPY . /app/alphafold
WORKDIR /app/alphafold

# Install the Python dependencies AlphaFold 3 needs.
ENV UV_HTTP_TIMEOUT=600
RUN uv pip install --python /alphafold3_venv/bin/python \
    --require-hashes -r requirements.txt
RUN uv pip install --python /alphafold3_venv/bin/python --no-deps .
RUN uv pip check --python /alphafold3_venv/bin/python
# Build chemical components database (this binary was installed by pip).
RUN build_data

# To work around a known XLA issue causing the compilation time to greatly
# increase, the following environment variable setting XLA flags must be enabled
# when running AlphaFold 3. Note that if using CUDA capability 7 GPUs, it is
# necessary to set the following XLA_FLAGS value instead:
# ENV XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
# (no need to disable gemm in that case as it is not supported for such GPU).
ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
# Memory settings used for folding up to 5,120 tokens on A100 80 GB.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=true
ENV XLA_CLIENT_MEM_FRACTION=0.95
# Do not set LD_LIBRARY_PATH to an old system CUDA/cuDNN installation.
# Persist this directory with a Docker volume only on trusted hosts.
ENV JAX_COMPILATION_CACHE_DIR=/root/.cache/af3

CMD ["python3", "run_alphafold.py"]
