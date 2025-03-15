# Use an official Python slim image as the base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Upgrade pip and install dependencies from requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 8000 so the container can serve the app on that port
EXPOSE 8000

# Command to run the app using Uvicorn. Adjust "app:app" if your app variable is named differently.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
