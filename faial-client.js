#!/usr/bin/env node

// Simple HTTP client for Faial MCP server
const https = require('https');

// Connect to the SSE endpoint
const url = 'https://faial-mcp-production.up.railway.app/sse';

console.log('Connecting to Faial MCP server at:', url);

const req = https.get(url, (res) => {
  console.log('Connected to Faial MCP server');
  console.log('Status:', res.statusCode);
  console.log('Headers:', res.headers);

  res.on('data', (chunk) => {
    process.stdout.write(chunk);
  });

  res.on('end', () => {
    console.log('\nConnection ended');
  });
});

req.on('error', (err) => {
  console.error('Error connecting to Faial MCP server:', err.message);
  process.exit(1);
});

// Keep the connection alive
req.setTimeout(0);
