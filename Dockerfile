FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and clean up apt cache to keep image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application including the model/ directory
COPY . .

# Expose the API port
EXPOSE 8000

# Run the FastAPI application using uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
