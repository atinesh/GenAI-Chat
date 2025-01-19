# Use the official Nginx image as the base image
FROM nginx:latest

# Copy the static HTML files into the Nginx root directory
COPY nginx/frontend/ /usr/share/nginx/html/

# Copy the custom default.conf file into the Nginx configuration directory
COPY nginx/default.conf /etc/nginx/conf.d/default.conf