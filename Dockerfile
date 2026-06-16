# ==============================================================================
# Stage 1: Rust Builder
# ==============================================================================
FROM rust:1.75-slim AS rust-builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Cargo files and source code
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/

# Build Rust binaries in release mode
RUN cargo build --release

# ==============================================================================
# Stage 2: Python Package Builder
# ==============================================================================
FROM python:3.9-slim AS python-builder

WORKDIR /app

# Install build tools for PyTorch / C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy python dependencies
COPY web/requirements.txt ./web/requirements.txt

# Install python dependencies to a prefix directory
RUN pip install --no-cache-dir --prefix=/install -r web/requirements.txt

# ==============================================================================
# Stage 3: Final Production Runtime
# ==============================================================================
FROM python:3.9-slim AS runtime

WORKDIR /app

# Install system dependencies (OpenMP for PyTorch, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=python-builder /install /usr/local

# Copy compiled Rust binaries from rust-builder stage
COPY --from=rust-builder /app/target/release/research-sim /app/target/release/research-sim
COPY --from=rust-builder /app/target/release/rl-env /app/target/release/rl-env

# Copy application code and folders
COPY web/ /app/web/
COPY rl/ /app/rl/
# Create directories for upload, results, and raw data
RUN mkdir -p /app/uploads /app/results /app/TradeData

# Setup runtime configuration environment variables
ENV DATABASE_URL=sqlite+aiosqlite:///./strataexec.db
ENV REDIS_URL=redis://redis:6379
ENV RUST_BINARY_PATH=/app/target/release
ENV DATA_PATH=/app/TradeData
ENV MODEL_PATH=/app/rl/models
ENV UPLOAD_PATH=/app/uploads
ENV RESULTS_PATH=/app/results
ENV PORT=8000

# Create upload and results directories
RUN mkdir -p /app/uploads /app/results

EXPOSE 8000

# Default entrypoint starts the API
CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000"]
