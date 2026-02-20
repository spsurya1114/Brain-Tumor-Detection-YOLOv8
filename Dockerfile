# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other libs
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Add src to PYTHONPATH so imports work
ENV PYTHONPATH="/app/src"

# Expose port
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
