FROM colmap/colmap:latest
LABEL maintainer="Paul-Edouard Sarlin"
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends git python3 python3-venv unzip wget && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . && \
    pip install --no-cache-dir notebook
