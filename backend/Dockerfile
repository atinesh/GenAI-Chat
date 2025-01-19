FROM python:3.11-slim-bullseye
MAINTAINER Atinesh Singh <atinesh.s@gmail.com>

WORKDIR /app
COPY backend/ .

RUN 

RUN pip3 install --upgrade pip && \
    # pip3 install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install -r requirements.txt

# PRODUCTION
CMD ["gunicorn", "-w 3", "-b", "0.0.0.0:5000", "app:app"]    
