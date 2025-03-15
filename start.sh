#!/bin/sh

# Print metadata to container logs
echo "Starting $PROJECT_NAME v$VERSION (Built: $BUILD_DATE, Maintainer: $MAINTAINER)"

# Start the Gunicorn server
exec gunicorn -w 3 -b 0.0.0.0:5000 app:app
