# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for images
RUN mkdir -p /app/images
RUN mkdir -p /app/static

# Command to run when container starts
CMD ["python", "./prac_4.py"]