FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    MODELS_DIR=/app/models \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

# CPU-only PyTorch. The default PyPI build bundles CUDA libraries and adds
# gigabytes; Cloud Run has no GPU here, so they would never be used.
RUN pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
# ultralytics depends on the full opencv-python, which needs GUI system
# libraries that the slim image lacks. Swap in the headless build.
RUN pip install -r requirements-api.txt \
    && pip uninstall -y opencv-python \
    && pip install opencv-python-headless==5.0.0.93

COPY src/ ./src/
# The demo pages, served by the API at /app/.
COPY web/ ./web/
# .dockerignore lets only synthetic.pt and finetuned.pt through from models/.
# The weights are gitignored, so this only works when building from a checkout
# that has them (locally or with Cloud Build). CI creates an empty models/ and
# builds anyway; such an image builds fine but the app will not start.
COPY models/ ./models/

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8080

# Model loading can take a few seconds, hence the start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'])"]

# Cloud Run supplies $PORT. `exec` lets uvicorn receive SIGTERM directly.
CMD ["sh", "-c", "exec uvicorn app:app --app-dir src --host 0.0.0.0 --port ${PORT}"]
