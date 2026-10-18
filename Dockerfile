FROM python:3.12-slim

# Set to 1 to include Demucs (AI drum separation): better drum/harmony separation, but adds
# ~1 GB to the image and ~1-2 min of CPU time per track.  docker compose build --build-arg WITH_DEMUCS=1
ARG WITH_DEMUCS=0
# Set to 0 to leave out the style model used by "Find music like this" (a much smaller image; tags then come
# from our own measurements and the song names only).
ARG WITH_STYLE_MODEL=1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    DATA_DIR=/data \
    MUSIC_MODELS_DIR=/app/models \
    PORT=8000

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN if [ "$WITH_DEMUCS" = "1" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install demucs; \
    fi
# The style model (Essentia + TensorFlow, CC BY-NC-SA 4.0) gets its own Python environment so its pinned
# libraries can't clash with ours; tagger.py runs it in a short-lived subprocess. A failed install only turns
# the style tags off (the self-test reports it); it doesn't fail the build.
COPY tagger.py .
RUN if [ "$WITH_STYLE_MODEL" = "1" ]; then \
      (python -m venv /opt/essentia && /opt/essentia/bin/pip install essentia-tensorflow==2.1b6.dev1389 \
       && (/opt/essentia/bin/python -c "import essentia.standard" 2>/dev/null \
           || /opt/essentia/bin/pip install "numpy<2") \
       && /opt/essentia/bin/python -c "import essentia.standard" \
       && python tagger.py --download) || echo "!! style model not installed - tags will come from measurements and names only"; \
    fi
ENV MUSIC_ESSENTIA_PYTHON=/opt/essentia/bin/python

COPY musicanalyze.py instruments.py inspiration.py notation.py accounts.py job_runner.py app.py selftest.py discover.py demos.py memory.py ./
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
