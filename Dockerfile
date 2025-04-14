# Use Python 3.8 image for compatibility
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Start the application
CMD ["python", "train.py"]