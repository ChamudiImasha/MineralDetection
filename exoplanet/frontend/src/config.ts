// API Configuration
// Update this URL when deploying to AWS

// For local development
export const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// After deploying backend to EC2, update .env.production with:
// VITE_API_URL=http://your-ec2-ip:8000

export const API_ENDPOINTS = {
  predict: `${API_BASE_URL}/predict`,
  health: `${API_BASE_URL}/health`,
};
