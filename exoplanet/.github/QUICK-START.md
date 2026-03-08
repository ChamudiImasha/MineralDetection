# Quick GitHub Actions Setup

## 🎯 One-Time Setup (5 minutes)

### 1. Create AWS Resources
```bash
# S3 bucket
aws s3 mb s3://mineral-detection-app
aws s3 website s3://mineral-detection-app --index-document index.html

# EC2: Launch Ubuntu t2.medium via AWS Console
# - Open ports 22 and 8000
# - Download key pair (.pem file)
```

### 2. Add GitHub Secrets
**Repository → Settings → Secrets → New secret**

Required secrets:
- `AWS_ACCESS_KEY_ID` - From AWS IAM
- `AWS_SECRET_ACCESS_KEY` - From AWS IAM
- `AWS_REGION` - e.g., `us-east-1`
- `S3_BUCKET_NAME` - e.g., `mineral-detection-app`
- `API_URL` - `http://YOUR_EC2_IP:8000`
- `EC2_HOST` - Your EC2 public IP
- `EC2_USERNAME` - `ubuntu`
- `EC2_SSH_KEY` - Full content of your `.pem` file

### 3. Initial EC2 Setup
```bash
# SSH into EC2
ssh -i key.pem ubuntu@YOUR_EC2_IP

# Install Python
sudo apt update
sudo apt install -y python3.9 python3-pip

# Create directory
mkdir -p ~/crism-backend

# Exit
exit
```

## 🚀 Deploy

Just push to main:
```bash
git add .
git commit -m "Deploy to AWS"
git push origin main
```

GitHub Actions will:
- ✅ Build frontend → Upload to S3
- ✅ Deploy backend → Restart on EC2
- ✅ Takes ~2-3 minutes

## 📱 Access Your App

- Frontend: `http://mineral-detection-app.s3-website-us-east-1.amazonaws.com`
- Backend: `http://YOUR_EC2_IP:8000/docs`

---

See [GITHUB-ACTIONS-SETUP.md](GITHUB-ACTIONS-SETUP.md) for complete guide.
