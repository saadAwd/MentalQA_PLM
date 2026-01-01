# Update Existing Google Cloud Deployment - Manual Evaluation Tab

This guide shows how to add the new manual evaluation tab to your existing Gradio app deployment.

## 🚀 Quick Update Steps

### 1. SSH into Your Google Cloud VM

```bash
# Use your existing SSH method
gcloud compute ssh YOUR_VM_NAME --zone YOUR_ZONE
# Or use regular SSH with your key
ssh -i your-key.pem user@YOUR_VM_IP
```

### 2. Navigate to Project Directory

```bash
cd ~/MentalQA_PLM
# Or wherever your project is located
```

### 3. Pull Latest Changes

```bash
git pull origin main
```

### 4. Verify New Files

```bash
# Check if new files exist
ls -lh manual_evaluation.py
ls -lh MANUAL_EVALUATION_GUIDE.md
```

### 5. Update Dependencies (if needed)

```bash
# Activate virtual environment
source venv/bin/activate

# Install any new dependencies (usually not needed, but just in case)
pip install -r requirements.txt
```

### 6. Restart the Gradio Service

```bash
# Restart the systemd service
sudo systemctl restart gradio-app

# Or if your service has a different name:
# sudo systemctl restart YOUR_SERVICE_NAME

# Check status
sudo systemctl status gradio-app
```

### 7. Verify the Service is Running

```bash
# Check logs for any errors
sudo journalctl -u gradio-app -n 50

# Test if app is responding
curl http://localhost:7860
```

### 8. Test the New Tab

1. Open your Gradio app in browser: `http://YOUR_DOMAIN` or `http://YOUR_IP`
2. Look for the new **"📝 Manual Evaluation"** tab
3. Click on it and verify:
   - Question and answers are displayed
   - Rating slider (1-5) is visible
   - Comment box is available
   - Navigation buttons work
   - Statistics panel shows

## 🔍 Troubleshooting

### Issue: Tab not showing

**Check:**
```bash
# Verify files are in place
ls -lh ~/MentalQA_PLM/manual_evaluation.py
ls -lh ~/MentalQA_PLM/retrieval_test_results.json

# Check Python can import the module
cd ~/MentalQA_PLM
source venv/bin/activate
python -c "from manual_evaluation import ManualEvaluator; print('OK')"
```

**Solution:**
- Ensure `retrieval_test_results.json` exists in the project root
- Check file permissions: `chmod 644 retrieval_test_results.json`
- Restart service: `sudo systemctl restart gradio-app`

### Issue: Service fails to restart

**Check logs:**
```bash
sudo journalctl -u gradio-app -n 100 --no-pager
```

**Common fixes:**
```bash
# Check if port is in use
sudo lsof -i :7860

# Kill process if needed
sudo kill -9 <PID>

# Restart service
sudo systemctl restart gradio-app
```

### Issue: Import errors in logs

**Solution:**
```bash
source venv/bin/activate
pip install --upgrade gradio
pip install -r requirements.txt
sudo systemctl restart gradio-app
```

## ✅ Verification Checklist

- [ ] Git pull completed successfully
- [ ] `manual_evaluation.py` exists in project root
- [ ] `retrieval_test_results.json` exists and is readable
- [ ] Service restarted without errors
- [ ] App is accessible via browser
- [ ] "📝 Manual Evaluation" tab is visible
- [ ] Can load a question and see all fields
- [ ] Rating slider works
- [ ] Can save an evaluation

## 📝 Quick Command Reference

```bash
# All-in-one update command
cd ~/MentalQA_PLM && \
git pull origin main && \
sudo systemctl restart gradio-app && \
sudo systemctl status gradio-app
```

## 🎯 What's New

The new **Manual Evaluation Tab** includes:
- ✅ 1-5 rating system for relevance
- ✅ Comment section
- ✅ Navigation (First/Previous/Next/Last/Jump)
- ✅ Statistics and progress tracking
- ✅ Fast access indicators
- ✅ Auto-save on navigation

That's it! The update should be quick and simple since your infrastructure is already set up.

