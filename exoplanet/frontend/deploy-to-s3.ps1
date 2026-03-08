# S3 Deployment Script
# Prerequisites: AWS CLI installed and configured (aws configure)

param(
    [Parameter(Mandatory=$true)]
    [string]$BucketName
)

Write-Host "Deploying to S3 bucket: $BucketName" -ForegroundColor Cyan

# Build the app first
Write-Host "`nStep 1: Building the app..." -ForegroundColor Yellow
npm run build

# Upload to S3
Write-Host "`nStep 2: Uploading to S3..." -ForegroundColor Yellow
aws s3 sync dist/ s3://$BucketName --delete

# Set bucket for static website hosting
Write-Host "`nStep 3: Configuring S3 bucket for static hosting..." -ForegroundColor Yellow
aws s3 website s3://$BucketName --index-document index.html --error-document index.html

Write-Host "`n✅ Deployment complete!" -ForegroundColor Green
Write-Host "Your site should be available at: http://$BucketName.s3-website-us-east-1.amazonaws.com" -ForegroundColor Cyan
Write-Host "(Replace 'us-east-1' with your actual region)" -ForegroundColor Gray
