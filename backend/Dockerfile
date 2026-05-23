FROM python:3.11-slim-bullseye

# Set metadata
ENV PROJECT_NAME="GenAI Chat"
ENV VERSION="1.3"
ENV BUILD_DATE="23-05-2026"
ENV MAINTAINER="Atinesh Singh <atinesh.s@gmail.com>"

WORKDIR /app
COPY backend/ .

RUN echo "Installing Dependencies ..."
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Copy startup script and give execution permission
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Run the startup script
CMD ["/start.sh"]  
