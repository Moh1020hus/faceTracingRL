# ---- build stage: compilers live here and never reach the final image -------
FROM python:3.9-slim AS builder

# build-essential + cmake: compiling dlib (pulled in by face-recognition).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# The bundled pip (23.0.1) rejects the typing_extensions wheel over a
# hyphen/underscore name mismatch, falls back to its sdist, and then cannot
# reach PyPI for the build deps because of the --index-url below.
RUN pip install --no-cache-dir --upgrade pip

# stable-baselines3 pulls in torch, and the default PyPI wheel bundles the CUDA
# runtime (~7GB unpacked). train.py pins device="cpu", so install the CPU wheel
# first and let the requirements resolve against it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# ---- runtime stage ----------------------------------------------------------
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Shared objects OpenCV links against at import time. No compiler needed here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY . /app

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "train.py"]
