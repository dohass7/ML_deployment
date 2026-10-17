FROM nginx:1.27-alpine

# Config Nginx custom
COPY frontend/nginx.conf /etc/nginx/nginx.conf

# Ton site
COPY frontend/index.html /usr/share/nginx/html/index.html

EXPOSE 80