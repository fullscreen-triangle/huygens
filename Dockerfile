# Universal Oscillatory Framework for Cardiovascular Analysis
# Production Docker Container

FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    pkg-config \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements first (for better caching)
COPY requirements.txt pyproject.toml setup.py ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install -e ".[dev,docs]"

# Copy entire project
COPY . .

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set default command for development
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Production stage
FROM base as production

# Copy only necessary files for production
COPY src/ ./src/
COPY demo/ ./demo/
COPY config.yaml ./
COPY setup.py pyproject.toml ./

# Install the package
RUN pip install .

# Create necessary directories
RUN mkdir -p /app/data /app/results /app/logs /app/tmp && \
    chown -R appuser:appuser /app

# Copy configuration and data
COPY config.yaml ./
COPY sample_*.json ./

# Switch to non-root user
USER appuser

# Expose port for API (if applicable)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.cardiovascular_oscillatory_suite import UniversalCardiovascularFramework; print('✓ Health check passed')" || exit 1

# Default command
CMD ["python", "analyze_cardiovascular_data.py", "--demo"]

# Scientific computing optimized stage
FROM base as scientific

# Install additional scientific dependencies
RUN pip install \
    tensorflow-cpu>=2.8.0,<3.0.0 \
    torch>=1.11.0,<2.0.0 \
    jupyter>=1.0.0,<2.0.0 \
    jupyterlab>=3.0.0,<4.0.0 \
    ipywidgets>=7.6.0,<8.0.0

# Install the package with ML extras
COPY . .
RUN pip install ".[ml,docs]"

# Create jupyter config
RUN mkdir -p /home/appuser/.jupyter && \
    echo "c.NotebookApp.ip = '0.0.0.0'" > /home/appuser/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /home/appuser/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = False" >> /home/appuser/.jupyter/jupyter_notebook_config.py && \
    chown -R appuser:appuser /home/appuser

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user  
USER appuser

# Expose Jupyter port
EXPOSE 8888

# Command for scientific computing
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# GPU-accelerated stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    pkg-config \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Set working directory
WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements and install dependencies
COPY requirements.txt pyproject.toml setup.py ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install cupy-cuda11x>=11.0.0,<12.0.0

# Copy and install the package
COPY . .
RUN pip install ".[gpu,ml]"

# Create necessary directories and set ownership
RUN mkdir -p /app/data /app/results /app/logs /app/tmp && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# GPU health check
HEALTHCHECK --interval=60s --timeout=20s --start-period=10s --retries=3 \
    CMD python -c "
import cupy as cp; 
cp.cuda.Device(0).use(); 
print('✓ GPU health check passed')
" || exit 1

# Default command for GPU processing
CMD ["python", "analyze_cardiovascular_data.py", "--demo", "--gpu"]

# Multi-stage final selection based on build arg
ARG BUILD_TARGET=production
FROM ${BUILD_TARGET} as final

# Labels for metadata
LABEL maintainer="Kundai Farai Sachikonye <kundai.sachikonye@wzw.tum.de>" \
      description="Universal Oscillatory Framework for Cardiovascular Analysis" \
      version="1.0.0" \
      license="MIT" \
      org.opencontainers.image.source="https://github.com/fullscreen-triangle/huygens" \
      org.opencontainers.image.documentation="https://huygens.readthedocs.io/" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.title="Cardiovascular Oscillatory Framework" \
      org.opencontainers.image.description="Multi-scale oscillatory coupling for cardiovascular analysis"

# Final user and working directory
USER appuser
WORKDIR /app
