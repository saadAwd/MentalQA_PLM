# Quick Deployment Commands - Reference Card

## 🚀 Initial Setup on VM

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install essentials
sudo apt install -y git curl wget build-essential python3-pip python3-venv nginx

# 3. Clone/Update repository
cd ~
git clone https://github.com/saadAwd/MentalQA_PLM.git
cd MentalQA_PLM
git pull origin main

# 4. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 5. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install gradio torch transformers sentence-transformers chromadb
```

## 🔧 Setup Nginx

```bash
# Create config file
sudo nano /etc/nginx/sites-available/gradio-app

# Enable config
sudo ln -s /etc/nginx/sites-available/gradio-app /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

## ⚙️ Setup Systemd Service

```bash
# Create service file (edit paths first!)
sudo nano /etc/systemd/system/gradio-app.service

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable gradio-app
sudo systemctl start gradio-app
sudo systemctl status gradio-app
```

## 📋 Daily Operations

```bash
# View logs
sudo journalctl -u gradio-app -f

# Restart app
sudo systemctl restart gradio-app

# Restart nginx
sudo systemctl restart nginx

# Update code
cd ~/MentalQA_PLM
git pull origin main
sudo systemctl restart gradio-app

# Check status
sudo systemctl status gradio-app
sudo systemctl status nginx
```

## 🔍 Troubleshooting

```bash
# Check if app is running
curl http://localhost:7860

# Check port
sudo netstat -tlnp | grep 7860

# Check nginx logs
sudo tail -f /var/log/nginx/error.log

# Check app logs
sudo journalctl -u gradio-app -n 100
```

## 📝 Nginx Config Template

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    proxy_read_timeout 300s;
    proxy_connect_timeout 75s;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_buffering off;
    }
}
```

## 🔐 Systemd Service Template

```ini
[Unit]
Description=Gradio KB Retriever App
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/MentalQA_PLM
Environment="PATH=/home/YOUR_USERNAME/MentalQA_PLM/venv/bin"
ExecStart=/home/YOUR_USERNAME/MentalQA_PLM/venv/bin/python -m knowldege_base.rag_staging.gradio_retriever_test
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Remember to replace:**
- `YOUR_USERNAME` with your actual username
- `yourdomain.com` with your domain or IP

