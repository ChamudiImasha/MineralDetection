# GitHub Actions Deployment Setup

Automated deployment to AWS using GitHub Actions - no manual steps needed!

## 🎯 Overview

Three workflows are configured:
1. **Deploy Frontend** - Builds and deploys React app to S3 on push to `main`
2. **Deploy Backend** - Deploys Python API to EC2 on push to `main`
3. **Build & Test** - Runs tests on PRs and `develop` branch

---

## ⚙️ Initial Setup (One-Time)

### Step 1: Create AWS Resources

#### Frontend (S3 Bucket)
```bash
# Create S3 bucket
aws s3 mb s3://mineral-detection-app

# Enable static website hosting
aws s3 website s3://mineral-detection-app \
  --index-document index.html \
  --error-document index.html

# Set bucket policy for public read
aws s3api put-bucket-policy --bucket mineral-detection-app --policy '{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "PublicReadGetObject",
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::mineral-detection-app/*"
  }]
}'
```

#### Backend (EC2 Instance)
1. Launch EC2 instance (Ubuntu 22.04, t2.medium)
2. Security Group: Allow ports 22 (SSH) and 8000 (API)
3. Save the private key (e.g., `mineral-key.pem`)
4. SSH in and run initial setup:

```bash
ssh -i mineral-key.pem ubuntu@YOUR_EC2_IP

# Install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.9 python3-pip

# Create app directory
mkdir -p ~/crism-backend
```

### Step 2: Configure GitHub Secrets

Go to your GitHub repository → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these secrets:

#### For Frontend Deployment
| Secret Name | Value | Example |
|------------|-------|---------|
| `AWS_ACCESS_KEY_ID` | Your AWS access key | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |
| `AWS_REGION` | AWS region | `us-east-1` |
| `S3_BUCKET_NAME` | Your S3 bucket name | `mineral-detection-app` |
| `API_URL` | Backend API URL | `http://YOUR_EC2_IP:8000` |
| `CLOUDFRONT_DISTRIBUTION_ID` | (Optional) CloudFront ID | `E1234567890ABC` |

#### For Backend Deployment
| Secret Name | Value | Example |
|------------|-------|---------|
| `EC2_HOST` | EC2 public IP or domain | `54.123.45.67` |
| `EC2_USERNAME` | SSH username | `ubuntu` |
| `EC2_SSH_KEY` | Private key content | Copy full content of `.pem` file |

**Getting AWS Credentials:**
```bash
# Create IAM user with programmatic access
aws iam create-user --user-name github-actions-deployer

# Attach policies (S3 + CloudFront)
aws iam attach-user-policy --user-name github-actions-deployer \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Create access key
aws iam create-access-key --user-name github-actions-deployer
```

**Getting EC2 SSH Key:**
```bash
# View your private key file
cat mineral-key.pem
# Copy entire content including -----BEGIN RSA PRIVATE KEY----- and -----END RSA PRIVATE KEY-----
```

---

## 🚀 How to Deploy

### Automatic Deployment
Just push to the `main` branch:

```bash
git add .
git commit -m "Update frontend/backend"
git push origin main
```

GitHub Actions will automatically:
1. Build your frontend
2. Upload to S3
3. Deploy backend to EC2
4. Restart services

### Manual Deployment
Go to **Actions** tab → Choose workflow → **Run workflow**

---

## 📋 Workflow Details

### Frontend Deployment (`.github/workflows/deploy-frontend.yml`)
Triggers:
- Push to `main` with changes in `frontend/`
- Manual trigger

Steps:
1. Checkout code
2. Install Node.js dependencies
3. Build React app with production API URL
4. Upload to S3
5. (Optional) Invalidate CloudFront cache

### Backend Deployment (`.github/workflows/deploy-backend.yml`)
Triggers:
- Push to `main` with changes in `backend/`
- Manual trigger

Steps:
1. Checkout code
2. SCP files to EC2
3. SSH into EC2
4. Install/update Python dependencies
5. Restart API service

### Build & Test (`.github/workflows/build-and-test.yml`)
Triggers:
- Pull requests to `main`
- Push to `develop` branch

Steps:
1. Build frontend and backend
2. Run tests (if configured)

---

## 🔍 Monitoring Deployments

### View Deployment Status
- Go to **Actions** tab in GitHub
- Click on the latest workflow run
- View logs for each step

### Check Deployment
```bash
# Frontend
curl http://mineral-detection-app.s3-website-us-east-1.amazonaws.com

# Backend
curl http://YOUR_EC2_IP:8000/health
```

---

## 🛠️ Troubleshooting

### Frontend Issues

**Build fails:**
- Check Node version in workflow matches your local version
- Verify all dependencies in `package.json`

**S3 upload fails:**
- Verify AWS credentials in GitHub secrets
- Check IAM user has S3 permissions
- Verify bucket name is correct

**Site not loading:**
- Check S3 bucket is public
- Verify static website hosting is enabled
- Check bucket policy allows public read

### Backend Issues

**SSH connection fails:**
- Verify EC2 IP address is correct
- Check EC2 security group allows SSH (port 22)
- Verify SSH key format (include BEGIN/END lines)

**Deployment succeeds but site doesn't work:**
- SSH into EC2 and check logs: `tail -f ~/crism-backend/app.log`
- Check if Python dependencies installed: `pip3 list`
- Verify API is running: `curl localhost:8000/health`
- Check EC2 security group allows port 8000

**Service won't restart:**
```bash
# SSH into EC2
ssh -i key.pem ubuntu@YOUR_EC2_IP

# Check process
ps aux | grep python

# Kill and restart manually
pkill -f "python.*main.py"
cd ~/crism-backend
nohup python3 main.py > app.log 2>&1 &
```

---

## 🎨 Customization

### Deploy to Different Environments

Create separate workflows for staging/production:

```yaml
# .github/workflows/deploy-staging.yml
on:
  push:
    branches:
      - develop

env:
  S3_BUCKET: mineral-detection-staging
  EC2_HOST: ${{ secrets.EC2_HOST_STAGING }}
```

### Add Notifications

Add Slack/Discord notifications:

```yaml
- name: Notify on success
  if: success()
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK }}
    payload: |
      {
        "text": "🚀 Deployment successful!"
      }
```

---

## 📊 Deployment Workflow Diagram

```
┌─────────────────┐
│  Push to main   │
└────────┬────────┘
         │
         ├──► Frontend changed? ──► Build React ──► Upload to S3
         │
         └──► Backend changed? ──► Upload to EC2 ──► Restart service
                                                    
                                    ✅ Deployed!
```

---

## 🔐 Security Best Practices

1. **Never commit secrets** - Use GitHub Secrets only
2. **Use least-privilege IAM roles** - Give only necessary permissions
3. **Rotate credentials regularly** - Update AWS keys every 90 days
4. **Use SSH keys, not passwords** - For EC2 access
5. **Enable MFA on AWS account** - Extra security layer

---

## 💰 Cost Optimization

- **S3**: ~$0.50/month (minimal traffic)
- **EC2 t2.medium**: Can use spot instances for 70% savings
- **Stop EC2 when not in use**: Create workflow to stop/start EC2

**Auto-stop EC2 workflow (optional):**
```yaml
# .github/workflows/stop-ec2.yml
name: Stop EC2
on:
  schedule:
    - cron: '0 22 * * *'  # Stop at 10 PM daily
```

---

## ✅ Checklist

Before first deployment:

- [ ] AWS account created
- [ ] S3 bucket created and configured
- [ ] EC2 instance launched and accessible
- [ ] All GitHub secrets added
- [ ] SSH key works (test manually first)
- [ ] Pushed code to `main` branch

---

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [AWS S3 Static Hosting](https://docs.aws.amazon.com/AmazonS3/latest/userguide/WebsiteHosting.html)
- [AWS EC2 User Guide](https://docs.aws.amazon.com/ec2/)

---

## 🎓 For Your Presentation

**What to show:**
1. Make a small code change
2. Push to GitHub: `git push origin main`
3. Show GitHub Actions tab running
4. Visit deployed URLs after workflow completes

**Demo script:**
```bash
# 1. Make a change
echo "Updated for demo" >> README.md

# 2. Commit and push
git add .
git commit -m "Demo deployment"
git push origin main

# 3. Show: GitHub → Actions tab
# 4. Wait ~2-3 minutes
# 5. Visit your live site!
```

This impresses with:
- ✅ Professional CI/CD pipeline
- ✅ Automated testing and deployment
- ✅ No manual server access needed
- ✅ Scales to production easily
