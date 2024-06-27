# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application code to the container
COPY . .
COPY app/.env.template app/.env

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r ./requirements.txt

# Install the application using setup.py
RUN pip install --no-cache-dir -e app/.

# Expose the port the app runs on
EXPOSE 8080