# ============================================================
# Dockerfile
# ------------------------------------------------------------
# Container image for the CallOS API (Cloud Run target).
#
# Builds the FastAPI + ADK service. The heavy fine-tuning deps in
# requirements.txt are only used on the local GPU box; for a slimmer
# API image you can split them out, but the README keeps one file.
# ============================================================

FROM python:3.12-slim

WORKDIR /app

# Install deps first so this layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source.
COPY . .

# Cloud Run sends traffic to $PORT (defaults to 8080).
EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
