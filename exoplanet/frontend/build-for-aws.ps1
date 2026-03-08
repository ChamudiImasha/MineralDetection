# Frontend Build Script for AWS S3 Deployment
# Run this script to build the React app for production

Write-Host "Building React app for production..." -ForegroundColor Cyan

# Build the app
npm run build

Write-Host "`nBuild complete! Files are in the 'dist' folder." -ForegroundColor Green
Write-Host "Next step: Upload the 'dist' folder contents to S3" -ForegroundColor Yellow
