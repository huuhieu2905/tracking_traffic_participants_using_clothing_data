FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgtk2.0-dev \
    libgtk-3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "tracking_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
