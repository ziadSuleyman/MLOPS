# ── Production image for the inference API ──
# Small, no notebooks, no dev tools.

# Python 3.13: the version the Task 2 model was trained under (see requirements/base.txt)
FROM python:3.13-slim AS base

# Prevent Python from writing .pyc and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Dependencies first (cached layer) ──
COPY requirements/base.txt requirements/base.txt
RUN pip install --no-cache-dir -r requirements/base.txt

# ── Application code ──
COPY src/ src/
COPY app/ app/
COPY config/ config/
COPY models/ models/
COPY scripts/ scripts/

# ── Non-root user, owning the logs directory ──
# (created as root, logs/ would be unwritable for appuser: the service could not
#  start without a bind mount over it)
RUN useradd --create-home appuser \
    && mkdir -p logs \
    && chown appuser:appuser logs
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
