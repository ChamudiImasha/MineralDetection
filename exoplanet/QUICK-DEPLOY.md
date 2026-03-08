# Quick AWS Deployment Reference

## ⚡ Quick Deploy for Presentation

### Frontend (S3) - 5 minutes

1. Build: `cd frontend; npm run build`
2. Create S3 bucket: `aws s3 mb s3://mineral-detection-demo`
3. Upload: `aws s3 sync dist/ s3://mineral-detection-demo --acl public-read`
4. Enable static hosting in S3 Console
5. URL: `http://mineral-detection-demo.s3-website-us-east-1.amazonaws.com`

### Backend (EC2) - 10 minutes

1. Launch EC2: Ubuntu t2.medium, open port 8000
2. Upload code: `scp -i key.pem -r backend/app/* ubuntu@EC2-IP:~/app/`
3. SSH in: `ssh -i key.pem ubuntu@EC2-IP`
4. Install:
   ```bash
   sudo apt update
   pip3 install -r requirements.txt -r requirements_api.txt
   nohup python3 main.py &
   ```
5. Test: `http://EC2-IP:8000/docs`

### Connect Them

1. Edit `frontend/.env.production`: `VITE_API_URL=http://EC2-IP:8000`
2. Rebuild frontend: `npm run build`
3. Re-upload to S3

✅ Done!

## 📝 What You Need

- AWS Account
- AWS CLI: `pip install awscli && aws configure`
- SSH key pair from EC2

## 💰 Cost for Demo

- S3: <$1/month
- EC2 t2.medium: $0.05/hour
- **Remember to STOP EC2 when not presenting!**

See [AWS-DEPLOYMENT.md](AWS-DEPLOYMENT.md) for detailed instructions.
