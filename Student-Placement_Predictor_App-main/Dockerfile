# Using slim python image for faster builds and lower resource footprint
FROM python:3.10-slim

# Setting the internal container working directory
WORKDIR /app

# Copying requirements first to leverage Docker container layer caching
COPY requirements.txt .

# Installing required libraries inside the sandbox
RUN pip install --no-cache-dir -r requirements.txt

# Copying the rest of the application files to the container
COPY . .

# Exposing port 10000 which is Render's default routing port
EXPOSE 10000

# Executing gunicorn production WSGI server, dynamically binding to the Render-provided PORT or defaulting to 10000
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} app:app"]