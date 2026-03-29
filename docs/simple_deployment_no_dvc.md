# 🚀 Simpler Deployment (No DVC)

If the DVC + Google Drive setup feels too complex or you encounter permission errors on Render, here is the **faster, simpler alternative** that skips DVC completely.

We will use **Git LFS (Large File Storage)** to push the model directly to GitHub, and Render will automatically pull it.

---

## 🛑 Step 1: Remove DVC components

Run these commands in your project to remove DVC tracking from the model:

```bash
# Un-track the model from DVC
dvc remove models/smartwaste_final.keras.dvc

# Remove DVC completely (optional, if you want a fully clean slate)
rm -rf .dvc dvc.yaml dvc.lock render.yaml
```

---

## 📂 Step 2: Use Git LFS for the model

Deep learning models (`.keras`) are too large for standard Git. We must tell Git to track it using LFS.

```bash
# Initialize Git LFS (only needed once)
git lfs install

# Tell Git LFS to track all .keras files
git lfs track "*.keras"

# Add the .gitattributes file that was just created
git add .gitattributes
```

Now, add the actual model back to Git:

```bash
# Add the model
git add models/smartwaste_final.keras
```

---

## 🐳 Step 3: Simplify the Dockerfile

Since we are no longer using DVC, we can drastically simplify the `Dockerfile` and skip `start.sh`.

Replace your `Dockerfile` with this:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000

# Start app directly
CMD ["gunicorn", "app.app:app", "--bind", "0.0.0.0:10000", "--workers", "1"]
```

---

## 📦 Step 4: Update requirements.txt

You no longer need `dvc` in your requirements. Remove it:

```text
pandas
scikit-learn
joblib
mlflow
flask==3.0.0
tensorflow==2.15.0
numpy
pillow
gunicorn
```

---

## 🚀 Step 5: Push to GitHub & Deploy

Commit everything and push:

```bash
git add Dockerfile requirements.txt models/smartwaste_final.keras .gitattributes
git commit -m "Switch to Git LFS deployment (Removed DVC)"
git push
```

### On Render:
1. Go to your Web Service
2. **Clear build cache & deploy**
3. Render automatically supports Git LFS, so it will pull the `.keras` model natively over Git without needing any API keys or Google Drive credentials.

---

### Comparison: Why use this vs DVC?

| Feature | DVC + GDrive | Git LFS (This guide) |
|---|---|---|
| **Speed to deploy** | Slower setup | **Very fast** |
| **Complexity** | High (Requires service accounts) | **Low (Built into GitHub+Render)** |
| **Storage limits** | 15GB+ Free (Google Drive) | **2GB Free (GitHub LFS bandwidth limits)** |
| **Best for** | Heavy MLOps, multiple large datasets | **Quick prototypes, single model deployments** |
