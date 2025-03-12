# Use Python 3.10 slim-bullseye image
FROM python:3.10-slim-bullseye

ENV PYTHONUNBUFFERED=1

# Set working directory to /app
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire repository into /app
COPY . .

# Run the training script located in the src directory
CMD ["python", "src/train.py"]
