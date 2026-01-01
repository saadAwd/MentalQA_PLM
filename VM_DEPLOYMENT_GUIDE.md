# VM Station Deployment Guide - Complete Setup

This guide provides step-by-step instructions to deploy the Gradio app with manual evaluation features on your VM station with Nginx reverse proxy.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Environment Setup](#environment-setup)
4. [Application Setup](#application-setup)
5. [Nginx Configuration](#nginx-configuration)
6. [Systemd Service Setup](#systemd-service-setup)
7. [SSL/HTTPS Setup (Optional)](#sslhttps-setup-optional)
8. [Verification & Testing](#verification--testing)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Ubuntu 20.04+ or Debian 11+ VM
- Root or sudo access
- Python 3.8+ installed
- Git installed
- At least 16GB RAM (recommended for model loading)
- GPU with CUDA support (optional but recommended)

---

## Initial Setup

### 1. Update System

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Install Essential Tools

```bash
sudo apt install -y git curl wget build-essential python3-pip python3-venv nginx
```

### 3. Clone/Update Repository

```bash
# If first time, clone the repository
cd ~
git clone https://github.com/saadAwd/MentalQA_PLM.git
cd MentalQA_PLM

# If repository already exists, pull latest changes
git pull origin main
```

---

## Environment Setup

### 1. Create Python Virtual Environment

```bash
cd ~/MentalQA_PLM
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# If you have specific requirements for the Gradio app
pip install gradio torch transformers sentence-transformers chromadb
```

### 3. Verify Installation

```bash
python -c "import gradio; print('Gradio version:', gradio.__version__)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

---

## Application Setup

### 1. Prepare Data Files

Ensure you have the required data files:

```bash
# Check if retrieval_test_results.json exists
ls -lh retrieval_test_results.json

# If not, you may need to generate it or copy it from another location
```

### 2. Set Up Knowledge Base (if needed)

```bash
# Check knowledge base status
python check_kb_status.py

# If knowledge base needs to be built, follow the KB setup instructions
```

### 3. Test Gradio App Locally

```bash
# Activate virtual environment
source venv/bin/activate

# Test the app (runs on port 7860)
cd knowldege_base/rag_staging
python -m gradio_retriever_test

# Or use the app_hf_spaces.py
cd ~/MentalQA_PLM
python app_hf_spaces.py
```

**Expected output:**
```
Running on local URL:  http://127.0.0.1:7860
```

Press `Ctrl+C` to stop the test.

---

## Nginx Configuration

### 1. Create Nginx Configuration File

```bash
sudo nano /etc/nginx/sites-available/gradio-app
```

Add the following configuration (replace `yourdomain.com` with your actual domain or IP):

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    # Increase timeouts for long-running requests
    proxy_read_timeout 300s;
    proxy_connect_timeout 75s;
    proxy_send_timeout 300s;

    # Client body size (for file uploads)
    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (for Gradio's real-time features)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Buffer settings
        proxy_buffering off;
        proxy_cache off;
    }

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

**Save and exit:** `Ctrl+X`, then `Y`, then `Enter`

### 2. Enable the Configuration

```bash
# Create symbolic link
sudo ln -s /etc/nginx/sites-available/gradio-app /etc/nginx/sites-enabled/

# Remove default site (optional)
sudo rm -f /etc/nginx/sites-enabled/default

# Test configuration
sudo nginx -t
```

**Expected output:**
```
nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
nginx: configuration file /etc/nginx/nginx.conf test is successful
```

### 3. Restart Nginx

```bash
sudo systemctl restart nginx
sudo systemctl enable nginx
sudo systemctl status nginx
```

---

## Systemd Service Setup

### 1. Create Systemd Service File

```bash
sudo nano /etc/systemd/system/gradio-app.service
```

Add the following content (adjust paths as needed):

```ini
[Unit]
Description=Gradio KB Retriever App with Manual Evaluation
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/your_username/MentalQA_PLM
Environment="PATH=/home/your_username/MentalQA_PLM/venv/bin"
ExecStart=/home/your_username/MentalQA_PLM/venv/bin/python -m knowldege_base.rag_staging.gradio_retriever_test
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Important:** Replace `your_username` with your actual username!

**Save and exit:** `Ctrl+X`, then `Y`, then `Enter`

### 2. Reload Systemd and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable gradio-app

# Start the service
sudo systemctl start gradio-app

# Check status
sudo systemctl status gradio-app
```

### 3. View Logs

```bash
# View recent logs
sudo journalctl -u gradio-app -n 50

# Follow logs in real-time
sudo journalctl -u gradio-app -f
```

---

## SSL/HTTPS Setup (Optional but Recommended)

### 1. Install Certbot

```bash
sudo apt install -y certbot python3-certbot-nginx
```

### 2. Obtain SSL Certificate

```bash
# Replace yourdomain.com with your actual domain
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

Follow the prompts:
- Enter your email address
- Agree to terms of service
- Choose whether to redirect HTTP to HTTPS (recommended: Yes)

### 3. Test Auto-Renewal

```bash
sudo certbot renew --dry-run
```

Certbot automatically renews certificates. You can verify this with a cron job:

```bash
sudo crontab -e
# Add this line:
0 0,12 * * * certbot renew --quiet
```

---

## Verification & Testing

### 1. Check Service Status

```bash
# Check if Gradio app is running
sudo systemctl status gradio-app

# Check if Nginx is running
sudo systemctl status nginx

# Check if port 7860 is listening
sudo netstat -tlnp | grep 7860
# Or
sudo ss -tlnp | grep 7860
```

### 2. Test Local Access

```bash
# Test from VM itself
curl http://localhost:7860
curl http://localhost/health
```

### 3. Test External Access

- **HTTP:** Open `http://yourdomain.com` or `http://your-ip-address` in browser
- **HTTPS:** Open `https://yourdomain.com` (if SSL is configured)

### 4. Verify Manual Evaluation Tab

1. Open the Gradio app in your browser
2. Navigate to the "📝 Manual Evaluation" tab
3. Verify you can see:
   - Question and answers displayed
   - Rating slider (1-5)
   - Comment box
   - Navigation buttons
   - Statistics panel
   - Fast access indicators

### 5. Test Evaluation Functionality

1. Select a rating (1-5)
2. Add a comment
3. Click "Save Evaluation"
4. Navigate to next item
5. Verify data is saved in `retrieval_test_results.json`

---

## Troubleshooting

### Issue: Service fails to start

**Check logs:**
```bash
sudo journalctl -u gradio-app -n 100
```

**Common causes:**
- Python path incorrect
- Virtual environment not activated
- Missing dependencies
- Port 7860 already in use

**Solution:**
```bash
# Check if port is in use
sudo lsof -i :7860

# Kill process if needed
sudo kill -9 <PID>

# Restart service
sudo systemctl restart gradio-app
```

### Issue: Nginx 502 Bad Gateway

**Check:**
1. Is Gradio app running?
   ```bash
   sudo systemctl status gradio-app
   ```

2. Is app listening on port 7860?
   ```bash
   curl http://localhost:7860
   ```

3. Check Nginx error logs:
   ```bash
   sudo tail -f /var/log/nginx/error.log
   ```

**Solution:**
- Restart Gradio service: `sudo systemctl restart gradio-app`
- Check firewall: `sudo ufw status`
- Verify Nginx config: `sudo nginx -t`

### Issue: Manual Evaluation tab not showing

**Check:**
1. Is `manual_evaluation.py` in the project root?
   ```bash
   ls -lh ~/MentalQA_PLM/manual_evaluation.py
   ```

2. Is `retrieval_test_results.json` accessible?
   ```bash
   ls -lh ~/MentalQA_PLM/retrieval_test_results.json
   ```

3. Check Python import:
   ```bash
   source venv/bin/activate
   python -c "from manual_evaluation import ManualEvaluator; print('OK')"
   ```

**Solution:**
- Ensure files are in correct location
- Check file permissions
- Restart the service

### Issue: Can't save evaluations

**Check:**
1. File permissions:
   ```bash
   ls -l retrieval_test_results.json
   sudo chmod 664 retrieval_test_results.json
   sudo chown $USER:$USER retrieval_test_results.json
   ```

2. Disk space:
   ```bash
   df -h
   ```

### Issue: App is slow or timing out

**Solutions:**
1. Increase Nginx timeouts (already in config)
2. Check system resources:
   ```bash
   htop
   free -h
   ```
3. Consider using GPU if available
4. Reduce model size or use quantization

---

## Quick Reference Commands

```bash
# Service management
sudo systemctl start gradio-app
sudo systemctl stop gradio-app
sudo systemctl restart gradio-app
sudo systemctl status gradio-app

# View logs
sudo journalctl -u gradio-app -f

# Nginx management
sudo systemctl restart nginx
sudo nginx -t
sudo tail -f /var/log/nginx/error.log

# Update application
cd ~/MentalQA_PLM
git pull origin main
sudo systemctl restart gradio-app

# Check ports
sudo netstat -tlnp | grep 7860
sudo ss -tlnp | grep 7860
```

---

## Features Available

After deployment, your Gradio app includes:

1. **📋 Results Tab**: View retrieval results with scores
2. **🔬 Comparison Tab**: Compare different retrieval methods
3. **📊 Statistics Tab**: View KB statistics
4. **📝 Manual Evaluation Tab** (NEW):
   - Rate retrieved answers (1-5 scale)
   - Add comments
   - Navigate between questions
   - View statistics and progress
   - Fast access to completed evaluations
   - Auto-save functionality

---

## Maintenance

### Regular Updates

```bash
# Pull latest code
cd ~/MentalQA_PLM
git pull origin main

# Update dependencies (if requirements.txt changed)
source venv/bin/activate
pip install -r requirements.txt --upgrade

# Restart service
sudo systemctl restart gradio-app
```

### Backup Evaluation Data

```bash
# Backup retrieval_test_results.json
cp retrieval_test_results.json retrieval_test_results.json.backup.$(date +%Y%m%d)

# Or use git (if file is tracked)
git add retrieval_test_results.json
git commit -m "Backup evaluation data"
```

---

## Support

For issues or questions:
1. Check logs: `sudo journalctl -u gradio-app -n 100`
2. Review this guide's troubleshooting section
3. Check the `MANUAL_EVALUATION_GUIDE.md` for evaluation-specific help

---

**Last Updated:** 2024
**Version:** 1.0

