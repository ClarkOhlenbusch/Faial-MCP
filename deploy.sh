#!/bin/bash

# Deployment script for Faial MCP Server
# This script helps deploy to various platforms

set -e

echo "🚀 Deploying Faial MCP Server..."

# Build the application
echo "📦 Building application..."
npm run build

# Check which platform to deploy to
if [ -n "$VERCEL_TOKEN" ]; then
    echo "🎯 Deploying to Vercel..."
    vercel --prod
elif [ -n "$NETLIFY_AUTH_TOKEN" ] && [ -n "$NETLIFY_SITE_ID" ]; then
    echo "🎯 Deploying to Netlify..."
    netlify deploy --prod --dir=dist
elif [ -n "$DOCKER_BUILD" ]; then
    echo "🐳 Building Docker image..."
    docker build -t faial-mcp .
    echo "✅ Docker image built successfully!"
    echo "📋 To run locally: docker run -p 3000:3000 faial-mcp"
    echo "📋 To push to registry: docker tag faial-mcp your-registry/faial-mcp && docker push your-registry/faial-mcp"
else
    echo "⚠️  No deployment target specified."
    echo "💡 Set one of these environment variables:"
    echo "   - VERCEL_TOKEN for Vercel deployment"
    echo "   - NETLIFY_AUTH_TOKEN and NETLIFY_SITE_ID for Netlify deployment"
    echo "   - DOCKER_BUILD=1 for Docker build only"
    exit 1
fi

echo "✅ Deployment completed!"
