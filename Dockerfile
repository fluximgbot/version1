# Use an official PyTorch image as the base image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any necessary dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on (e.g., 5000 for a web app)
EXPOSE 5000

# Define environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Command to run the application
CMD ["python3", "app.py"]
