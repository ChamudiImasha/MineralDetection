# AWS Deployment Guide - Minimal Setup for Presentation

This guide provides simple deployment steps for S3 (frontend) and EC2 (backend).

## Prerequisites

- AWS Account
- AWS CLI installed: `pip install awscli`
- AWS CLI configured: `aws configure`
- SSH key pair for EC2 access

---

## 🎨 Frontend Deployment (S3 + CloudFront)

### Option 1: Using PowerShell Script (Windows)

1. **Navigate to frontend directory:**

   ```powershell
   cd frontend
   ```

2. **Run deployment script:**
   ```powershell
   .\deploy-to-s3.ps1 -BucketName "your-unique-bucket-name"
   ```

### Option 2: Manual Steps

1. **Build the app:**

   ```bash
   cd frontend
   npm run build
   ```

2. **Create S3 bucket (via AWS Console or CLI):**

   ```bash
   aws s3 mb s3://your-mineral-detection-app
   ```

3. **Upload files:**

   ```bash
   aws s3 sync dist/ s3://your-mineral-detection-app --acl public-read
   ```

4. **Enable static website hosting:**
   - Go to S3 Console → Your Bucket → Properties → Static website hosting
   - Enable it and set `index.html` for both index and error documents
   - Make bucket public (Permissions → Uncheck "Block all public access")

5. **Access your site:**
   ```
   http://your-mineral-detection-app.s3-website-{region}.amazonaws.com
   ```

### Optional: Add CloudFront CDN

1. Create CloudFront distribution pointing to your S3 bucket
2. This gives you HTTPS and global CDN
3. Takes 10-15 minutes to deploy

---

## 🖥️ Backend Deployment (EC2)

### Step 1: Launch EC2 Instance

1. **Go to EC2 Console** and click "Launch Instance"

2. **Configuration:**
   - **Name:** `mineral-detection-backend`
   - **AMI:** Ubuntu Server 22.04 LTS
   - **Instance Type:** `t2.medium` or `t3.medium` (recommended for ML)
   - **Key Pair:** Create or select existing
   - **Security Group:** Create new with these rules:
     - SSH (port 22) - Your IP
     - Custom TCP (port 8000) - Anywhere (0.0.0.0/0)
   - **Storage:** 20-30 GB

3. **Launch the instance**

### Step 2: Connect and Setup

1. **Connect via SSH:**

   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-public-ip
   ```

2. **Upload backend code:**
   Open a NEW terminal on your local machine:

   ```bash
   scp -i your-key.pem -r backend/app/* ubuntu@your-ec2-ip:~/crism-backend/
   ```

3. **Run the setup script ON EC2:**
   ```bash
   cd ~/crism-backend
   chmod +x deploy-to-ec2.sh
   ./deploy-to-ec2.sh
   ```

### Step 3: Manual Setup (if script doesn't work)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python
sudo apt install -y python3.9 python3-pip python3.9-venv

# Install dependencies
pip3 install -r requirements.txt
pip3 install -r requirements_api.txt

# Run the server
python3 main.py
```

### Step 4: Keep Server Running (Background)

Option A - Using nohup:

```bash
nohup python3 main.py > app.log 2>&1 &
```

Option B - Using screen:

```bash
sudo apt install screen
screen -S backend
python3 main.py
# Press Ctrl+A then D to detach
# Reconnect with: screen -r backend
```

### Step 5: Test Backend

```bash
curl http://your-ec2-ip:8000/health
```

Or visit in browser: `http://your-ec2-ip:8000/docs`

---

## 🔗 Connect Frontend to Backend

Update your frontend API endpoint:

1. **Edit frontend configuration:**

   ```typescript
   // In your frontend API config file
   const API_URL = "http://your-ec2-public-ip:8000";
   ```

2. **Rebuild and redeploy frontend**

---

## 📊 Presentation Demo URLs

After deployment, you'll have:

- **Frontend:** `http://your-bucket-name.s3-website-us-east-1.amazonaws.com`
- **Backend API:** `http://your-ec2-ip:8000`
- **API Docs:** `http://your-ec2-ip:8000/docs`

---

## 💰 Cost Estimate (for presentation)

- **S3:** ~$0.50/month (minimal traffic)
- **EC2 t2.medium:** ~$0.05/hour (~$35/month, but can stop when not demoing)
- **Data Transfer:** Minimal for presentation

**Tip:** Stop (don't terminate) your EC2 instance when not presenting to save costs!

---

## 🛑 Clean Up After Presentation

```bash
# Delete S3 bucket
aws s3 rb s3://your-bucket-name --force

# Terminate EC2 instance from AWS Console
# Go to EC2 → Instances → Select instance → Instance State → Terminate
```

---

## 🚨 Troubleshooting

### Frontend Issues

- **404 errors:** Ensure index.html is set as error document in S3
- **CORS errors:** Check backend CORS settings in `api_server.py`

### Backend Issues

- **Can't connect:** Check EC2 security group allows port 8000
- **Model not found:** Model file needs to be included in uploaded files
- **Service won't start:** Check logs with `sudo journalctl -u crism-api -f`

### Quick Fixes

```bash
# Check if backend is running
curl http://localhost:8000/health

# Check backend logs
tail -f ~/crism-backend/output/*.log

# Restart backend service
sudo systemctl restart crism-api
```

---

## 📝 Notes

- This setup is for demonstration purposes
- For production, consider:
  - HTTPS (use AWS Certificate Manager + CloudFront)
  - Load balancer for backend
  - Auto-scaling
  - Database for persistent storage
  - Proper secrets management
