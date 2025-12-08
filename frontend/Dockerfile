# Use the official nginx image as base
FROM nginx:alpine

# Copy the built React app to nginx html directory
COPY build/ /usr/share/nginx/html

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
