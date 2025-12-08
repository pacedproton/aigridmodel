# Use Node.js for production server with API proxying
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy built React app
COPY build/ ./build/

# Copy server files
COPY server.js ./

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

# Change ownership
RUN chown -R nextjs:nodejs /app
USER nextjs

# Expose port 60000
EXPOSE 60000

# Set environment variable for port
ENV PORT=60000

# Start the server
CMD ["node", "server.js"]
