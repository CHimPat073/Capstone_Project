FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Camelot's PDF backends need basic system libraries. Ghostscript is optional,
# but installing it enables lattice table extraction in container deployments.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ghostscript libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
