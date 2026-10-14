FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv

# System dependencies for OpenCV and video handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and helper scripts
COPY app ./app
COPY scripts ./scripts

# Prepare runtime storage folders and non-root security context
RUN mkdir -p data/uploads data/frames && \
    useradd -m appuser && \
    chown -R appuser /srv

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]