FROM python:3.12-slim

# Set to 1 to include Demucs (AI drum separation): better drum/harmony separation, but adds
# ~1 GB to the image and ~1-2 min of CPU time per track.  docker compose build --build-arg WITH_DEMUCS=1
ARG WITH_DEMUCS=0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    DATA_DIR=/data \
    PORT=8000

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN if [ "$WITH_DEMUCS" = "1" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install demucs; \
    fi

COPY musicanalyze.py instruments.py job_runner.py app.py ./
COPY templates ./templates

RUN useradd --create-home appuser && mkdir -p /data && chown appuser /data
USER appuser
# Download the Demucs model at build time so the first job doesn't have to.
RUN if [ "$WITH_DEMUCS" = "1" ]; then \
      python -c "from demucs.pretrained import get_model; get_model('htdemucs')"; \
    fi
VOLUME /data
EXPOSE 8000

# One worker: a background thread queues jobs and runs each one in its own subprocess.
CMD gunicorn --workers 1 --threads 4 --timeout 120 --access-logfile - --bind 0.0.0.0:${PORT} app:app
