# Test the deployed Faial MCP server
# Update with your actual Railway app URL
$baseUrl = "https://faial-mcp-production.up.railway.app"

Write-Host "🚀 Testing Railway Deployment: $baseUrl" -ForegroundColor Green
Write-Host "" -ForegroundColor Green

# Test 1: Health endpoint
$uri = "$baseUrl/health"
Write-Host "📋 Test 1: Health Check" -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri $uri -UseBasicParsing -TimeoutSec 10
    Write-Host "✅ Health: SUCCESS" -ForegroundColor Green
    Write-Host "   Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "   Content: $($response.Content)" -ForegroundColor Yellow
} catch {
    Write-Host "❌ Health: FAILED - $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "" -ForegroundColor Green

# Test 2: SSE endpoint
$uri = "$baseUrl/sse"
Write-Host "📋 Test 2: SSE Endpoint" -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri $uri -UseBasicParsing -TimeoutSec 10
    Write-Host "✅ SSE: SUCCESS" -ForegroundColor Green
    Write-Host "   Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "   Content: $($response.Content)" -ForegroundColor Yellow
} catch {
    Write-Host "❌ SSE: FAILED - $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "" -ForegroundColor Green

# Test 3: Messages endpoint (should return 404/405 for GET)
$uri = "$baseUrl/messages"
Write-Host "📋 Test 3: Messages Endpoint (GET - should fail)" -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri $uri -UseBasicParsing -TimeoutSec 5
    Write-Host "⚠️ Messages: Unexpectedly succeeded (this endpoint should only accept POST)" -ForegroundColor Yellow
} catch {
    Write-Host "✅ Messages: Expected behavior (only accepts POST requests)" -ForegroundColor Green
}

Write-Host "" -ForegroundColor Green
Write-Host "🎯 MCP Configuration for Users:" -ForegroundColor Green
Write-Host "URL: $baseUrl" -ForegroundColor Yellow
Write-Host "" -ForegroundColor Green
