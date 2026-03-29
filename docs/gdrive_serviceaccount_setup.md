# 🔐 GDrive Service Account Setup for DVC + Render

## What this does
Allows `dvc pull` to authenticate **without a browser** — required for
automated Docker builds and cloud deployments like Render.

---

## STEP 1 — Create a Google Cloud Project

1. Go to → https://console.cloud.google.com/
2. Click **"New Project"** → name it `smartwaste-dvc` → **Create**
3. Make sure the new project is selected in the top bar

---

## STEP 2 — Enable Google Drive API

1. Go to → https://console.cloud.google.com/apis/library/drive.googleapis.com
2. Click **Enable**

---

## STEP 3 — Create a Service Account

1. Go to → https://console.cloud.google.com/iam-admin/serviceaccounts
2. Click **"Create Service Account"**
   - Name: `smartwaste-dvc`
   - Click **Create and Continue** → **Done**
3. Click the service account you just created
4. Go to **Keys** tab → **Add Key** → **Create new key** → **JSON**
5. A file like `smartwaste-dvc-xxxx.json` downloads — **keep it safe**

---

## STEP 4 — Share your Google Drive folder with the Service Account

1. Open **Google Drive** → find the folder where your model is stored
2. Right-click → **Share**
3. Paste the **service account email** (looks like `smartwaste-dvc@your-project.iam.gserviceaccount.com`)
4. Set permission to **Editor** → **Send**

---

## STEP 5 — Configure DVC locally

Run these commands inside your project:

```bash
# Tell DVC to authenticate using the service account
dvc remote modify myremote gdrive_use_service_account true
dvc remote modify myremote gdrive_service_account_json_file_path smartwaste-dvc-xxxx.json

# Test it works
dvc pull
```

If `dvc pull` succeeds → push your updated `.dvc/config` to Git:

```bash
git add .dvc/config
git commit -m "Configure DVC service account auth"
git push
```

> ⚠️ **Never commit the JSON key file to Git.** Add it to `.gitignore`.

---

## STEP 6 — Add credentials to Render

1. In your Render dashboard → select your **smartwaste** service
2. Go to **Environment** tab → **Add Environment Variable**
3. Key: `GDRIVE_CREDENTIALS_DATA`
4. Value: paste the **entire contents** of `smartwaste-dvc-xxxx.json`

   ```bash
   # On Windows PowerShell — print the file contents to copy:
   Get-Content smartwaste-dvc-xxxx.json
   ```

5. Click **Save**

---

## STEP 7 — Deploy

```bash
git push
```

Render will:
1. Build the Docker image (install deps + configure DVC)
2. At startup: `start.sh` writes `GDRIVE_CREDENTIALS_DATA` → `/tmp/gdrive_sa.json`
3. `dvc pull` downloads `models/smartwaste_final.keras` from GDrive
4. `gunicorn` starts Flask app on port 10000

---

## Files changed summary

| File | What it does |
|---|---|
| `Dockerfile` | Installs `dvc[gdrive]`, configures service account path, runs `start.sh` |
| `start.sh` | Writes creds at runtime → dvc pull → gunicorn |
| `.dvc/config` | Stores service account auth config (no secrets) |
| **Render env var** | `GDRIVE_CREDENTIALS_DATA` = your service account JSON content |

---

## Troubleshooting

| Error | Fix |
|---|---|
| `GDRIVE_CREDENTIALS_DATA is not set` | Add the env var in Render → Environment |
| `403 Forbidden` | Service account email not added to GDrive folder (Step 4) |
| `model not found after dvc pull` | Run `dvc push` locally first to upload the model |
| `dvc: command not found` | `dvc[gdrive]` not installed — check Dockerfile |
