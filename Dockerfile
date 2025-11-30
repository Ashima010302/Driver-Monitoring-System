FROM python:3.10-slim


# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
ffmpeg \
libsm6 \
libxext6 \
libxrender1 \
libglib2.0-0 \
libsndfile1 \
&& rm -rf /var/lib/apt/lists/*


WORKDIR /app


# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt


# Copy project files
COPY . /app


# Ensure folders exist
RUN mkdir -p /app/uploads /app/models && chmod -R 777 /app/uploads /app/models


# Expose Flask port
EXPOSE 5000


ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1


CMD ["python", "app.py"]