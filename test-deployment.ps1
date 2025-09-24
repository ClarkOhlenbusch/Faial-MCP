# Test the deployed Faial MCP server
# Replace YOUR-RAILWAY-APP-NAME with your actual Railway app name
$uri = "https://YOUR-RAILWAY-APP-NAME.railway.app/health"

try {
    $response = Invoke-WebRequest -Uri $uri -UseBasicParsing -TimeoutSec 10
    Write-Host "✅ SUCCESS!" -ForegroundColor Green
    Write-Host "Status Code: $($response.StatusCode)" -ForegroundColor Cyan
    Write-Host "Content: $($response.Content)" -ForegroundColor Yellow
} catch {
    Write-Host "❌ ERROR: $($_.Exception.Message)" -ForegroundColor Red

    # Try with headers
    try {
        Write-Host "Trying with headers..." -ForegroundColor Yellow
        $headers = @{
            "User-Agent" = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            "Accept" = "application/json"
        }
        $response = Invoke-WebRequest -Uri $uri -Headers $headers -UseBasicParsing -TimeoutSec 10
        Write-Host "✅ SUCCESS with headers!" -ForegroundColor Green
        Write-Host "Status Code: $($response.StatusCode)" -ForegroundColor Cyan
        Write-Host "Content: $($response.Content)" -ForegroundColor Yellow
    } catch {
        Write-Host "❌ Still failing: $($_.Exception.Message)" -ForegroundColor Red
    }
}
