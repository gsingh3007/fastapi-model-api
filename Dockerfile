# Use Python 3.9
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy dependencies first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Command to run your app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
