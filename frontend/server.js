const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const path = require('path');
const http = require('http');

const app = express();
const PORT = process.env.PORT || 60000;

// Proxy API requests to the backend
app.use('/api', (req, res) => {
  const backendPath = req.originalUrl;

  const options = {
    hostname: 'localhost',
    port: 5001,
    path: backendPath,
    method: req.method,
    headers: {
      ...req.headers,
      host: 'localhost:5001'
    }
  };

  const proxyReq = http.request(options, (proxyRes) => {
    // Copy response headers
    res.status(proxyRes.statusCode);
    Object.keys(proxyRes.headers).forEach(key => {
      res.setHeader(key, proxyRes.headers[key]);
    });

    // Pipe response body
    proxyRes.pipe(res);
  });

  proxyReq.on('error', (err) => {
    console.error('Proxy error:', err.message);
    res.status(500).send('Backend service unavailable');
  });

  // Pipe request body if present
  req.pipe(proxyReq);
});

// Serve static files from the React app build directory
app.use(express.static(path.join(__dirname, 'build')));

// Handle client-side routing - serve React app for all non-API routes
app.use((req, res, next) => {
  // If it's not an API request and not a static file, serve the React app
  if (!req.path.startsWith('/api') && !req.path.includes('.')) {
    res.sendFile(path.join(__dirname, 'build', 'index.html'));
  } else if (req.path.startsWith('/api')) {
    // This shouldn't happen if proxy is working
    console.log('API request reached catch-all middleware - proxy failed!');
    res.status(500).send('Proxy configuration error');
  } else {
    next(); // Let static files middleware handle it
  }
});

app.listen(PORT, () => {
  console.log(`Frontend server running on port ${PORT}`);
  console.log(`API requests proxied to http://localhost:5001`);
});
