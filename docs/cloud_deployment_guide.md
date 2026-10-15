# Cloud Deployment Guide — AML Priority

## Deployment Strategy

Two complementary deployment targets:
1. **Hugging Face Spaces** — lightweight Gradio inference UI (free tier, public URL)
2. **Full app** — React + FastAPI (requires a paid host like Railway, Render, or AWS)

This guide covers HF Spaces deployment. The full app Docker deployment is documented in README.md.

---

## Hugging Face Spaces Deployment

### Prerequisites
- Hugging Face account (free): https://huggingface.co/join
- Git installed
- `huggingface_hub` Python package: `pip install huggingface_hub`

### Step 1 — Prepare artifacts
```powershell
# From the aml-priority/ directory:
python hf_space/setup_space.py
# Verifies: artifacts/, reports/, figures/ all copied into hf_space/
```

### Step 2 — Create a new Space
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Name it: `aml-priority` (or your preferred name)
4. SDK: **Gradio**
5. Hardware: **CPU Basic** (free tier, 2 vCPU, 16GB RAM — sufficient)
6. Visibility: Public

### Step 3 — Push the Space
```powershell
# Authenticate (one time)
python -c "from huggingface_hub import login; login()"

# Clone your new Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/aml-priority hf_deploy
Copy-Item -Recurse "aml-priority/hf_space/*" "hf_deploy/" -Force

# Push
Push-Location hf_deploy
git add .
git commit -m "Initial AML Priority deployment"
git push
Pop-Location
```

### Step 4 — Verify
- Visit: `https://huggingface.co/spaces/YOUR_USERNAME/aml-priority`
- Check build logs in the Space's "Logs" tab
- Test a prediction on the "Score Transaction" tab

---

## Resource Requirements

| Resource | HF Free Tier | Our Usage | Status |
|---|---|---|---|
| RAM | 16 GB | ~500 MB (model + gradio) | OK |
| Storage | 50 GB | ~2 MB (artifacts + figures) | OK |
| Build time | 10 min | ~3 min | OK |
| CPU | 2 vCPU | Low (inference only) | OK |
| GPU | None | Not needed | OK |

---

## Environment Variables (if needed)

No secrets required for the HF Space deployment.
The model is loaded from the `artifacts/` directory bundled in the repo.

---

## Checklist Before Pushing

- [ ] `python hf_space/setup_space.py` completed without errors
- [ ] `hf_space/artifacts/xgboost_model.joblib` exists (>400KB)
- [ ] `hf_space/artifacts/model_metadata.json` exists
- [ ] `hf_space/figures/` contains at least 6 PNG files
- [ ] `hf_space/README.md` has correct `sdk: gradio` front-matter
- [ ] `hf_space/requirements.txt` is up to date
- [ ] No secrets, credentials, or dataset files in hf_space/

---

## Full App Docker Deployment (local verified, cloud pending)

```powershell
# Start Docker Desktop first, then:
Push-Location aml-priority

# Build and start
docker-compose up --build

# Backend: http://localhost:8000/api/docs
# Frontend: http://localhost:3000

Pop-Location
```

### Docker notes
- Model artifacts are mounted via volume (not baked into image)
- Raw dataset is excluded via .dockerignore
- Database persists in backend/data/

### Cloud options for full app
- **Railway.app** — supports Docker Compose, free tier available
- **Render.com** — supports Docker, free tier with sleep
- **AWS ECS** — requires account setup, beyond free tier

---

## Next Steps (joint action required)

1. You create the HF Space at https://huggingface.co/spaces (needs your account)
2. Run: `python hf_space/setup_space.py`
3. Push to Space using the commands in Step 3 above
4. Share the live URL for the submission
