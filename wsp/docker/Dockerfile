# Sandbox for the weight editor's Python code: numpy only, run with no network.
# Build: docker build -t admiral-wsp-sandbox:1 wsp/docker
FROM python:3.10-slim
RUN pip install --no-cache-dir numpy==1.26.4 \
 && useradd --uid 1000 --create-home editor
USER editor
WORKDIR /work
