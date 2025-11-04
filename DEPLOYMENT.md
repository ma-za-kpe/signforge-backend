# SignForge Backend Deployment Guide

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Railway Deployment](#railway-deployment)
3. [Critical Configuration](#critical-configuration)
4. [Environment Variables](#environment-variables)
5. [Testing & Verification](#testing--verification)
6. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### Service Structure
The SignForge platform consists of **two main services**:

1. **signforge-backend** (FastAPI Backend)
   - **Dockerfile**: `Dockerfile`
   - **Port**: 9000 (hardcoded in railway.json)
   - **Includes**:
     - Ghana Sign Language Dictionary API
     - AI-powered search service (FAISS + sentence-transformers)
     - Contribution system (MediaPipe landmarks)
     - Training queue (background processing)
     - Analytics dashboard
     - Format creator (PDF, audio, video generation)
   - **Database**: PostgreSQL (automatically seeds on startup)

2. **signforge-frontend** (Next.js)
   - **Deployed**: Vercel
   - **Calls**: FastAPI backend via `NEXT_PUBLIC_API_URL`
   - **Purpose**: User interface for dictionary browsing and contributions

**CRITICAL**: Backend uses port 9000, different from typical FastAPI port 8000!

---

## Railway Deployment

### Initial Setup

#### 1. Link to Railway Project

```bash
# Navigate to backend directory
cd signforge-hackathon/backend

# Link to Railway project (invigirating-forginess)
railway link

# Expected output:
# ✓ Linked to project invigirating-forgiveness
```

#### 2. Configure Service

**Project**: `invigirating-forginess`

**Configuration File**: `railway.json` (CRITICAL: Do NOT create `railway.toml`)

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "/bin/sh -c \"uvicorn main:app --host 0.0.0.0 --port 9000\"",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

**Domain Configuration**:
- **Target Port**: 9000 (MUST match hardcoded port in startCommand)
- **Custom Domain** (if configured): TBD
- **Railway URL**: `https://[service-name]-production.up.railway.app`

#### 3. Connect GitHub Repository

**Recommended Setup**:
1. In Railway dashboard → Settings
2. Connect GitHub repository: `[your-org]/signforge-hackathon`
3. Set root directory: `backend` (CRITICAL!)
4. Enable automatic deployments: `main` branch
5. Build command: Auto-detected (uses Dockerfile)
6. Start command: Override with railway.json

---

## Critical Configuration

### 1. PORT Configuration (Avoid 502 Errors)

**Problem**: Railway domain targets port 9000, but service might listen on dynamic `$PORT`

**Solution**: Hardcode port 9000 in railway.json:

```json
"startCommand": "/bin/sh -c \"uvicorn main:app --host 0.0.0.0 --port 9000\""
```

**Why This Works**:
- Railway overrides Dockerfile CMD with `startCommand`
- Wrapped in `/bin/sh -c "..."` for proper shell execution
- Port 9000 hardcoded to match Railway domain configuration

**DON'T DO THIS**:
```json
"startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT"  // ❌ Won't expand
"startCommand": "uvicorn main:app --host 0.0.0.0 --port 9000"   // ❌ No shell wrapping
```

---

### 2. Configuration File Priority

**Railway Priority**:
1. `railway.toml` (if exists - takes precedence)
2. `railway.json`
3. Auto-detection

**IMPORTANT**: Only use `railway.json`, delete `railway.toml` if it exists

**Why JSON over TOML**:
- JSON schema validation (`"$schema"` field)
- Better IDE support and autocomplete
- Clearer error messages
- Matches Railway dashboard configuration

---

### 3. Root Directory Configuration

**CRITICAL**: Railway must build from `backend/` subdirectory

**Railway Dashboard Settings**:
- Source: GitHub repository
- Root Directory: `backend` ← Set this!
- Watch Paths: Leave empty (watches all changes in root directory)

**Why This Matters**:
- Repository has multiple folders: `frontend/`, `backend/`, `docs/`
- Railway needs to see `Dockerfile`, `requirements.txt`, and `main.py` in root
- Without this, Railway will fail to find Dockerfile

---

### 4. CORS Configuration

**Problem**: Frontend blocked by CORS policy

**Solution**: `main.py` already configured with regex pattern:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.vercel\.app|http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Covered Origins**:
- ✅ All Vercel deployments: `https://*.vercel.app`
- ✅ Local development: `http://localhost:3000`, `http://localhost:3001`, etc.
- ✅ Preview deployments: Automatically included

**If you need to add specific origins**:
```python
allow_origins=[
    "http://localhost:3000",
    "https://signforge.vercel.app",
    "https://signforge-custom-domain.com",
],
```

---

## Environment Variables

### Required Environment Variables

#### signforge-backend Service

```bash
# Database (automatically provided by Railway if you add PostgreSQL service)
DATABASE_URL=postgresql://postgres:password@postgres.railway.internal:5432/railway

# Optional: Port (not needed if hardcoded in startCommand)
PORT=9000

# Optional: Environment indicator
ENVIRONMENT=production
```

### Optional Environment Variables (for specific features)

```bash
# If using Railway Volumes for persistent storage
BRAIN_DIR=/data/ghsl_brain

# If using external AI services (future)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# If using Twilio for WhatsApp (rural delivery)
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_WHATSAPP_NUMBER=whatsapp:+...

# If using URL shortening (format creator)
BITLY_ACCESS_TOKEN=...
```

### Setting Environment Variables

```bash
# Via Railway CLI
railway variables set DATABASE_URL="postgresql://..."

# Via Railway Dashboard
# Go to: Service → Variables → Add Variable
```

---

## Testing & Verification

### 1. Health Check

```bash
curl https://[your-service]-production.up.railway.app/
```

Expected response:
```json
{
  "message": "Ghana Sign Language API",
  "status": "healthy",
  "version": "1.0.0",
  "features": {
    "search": true,
    "contribution": true,
    "training_queue": true,
    "format_creator": true,
    "analytics": true
  }
}
```

### 2. Test Search Endpoint

```bash
curl "https://[your-service]-production.up.railway.app/search?q=hello"
```

Expected response:
```json
[
  {
    "word": "HELLO",
    "definition": "A greeting...",
    "image_path": "/sign_images/hello.jpg",
    "confidence": 0.95,
    "metadata": {
      "category": "greetings"
    }
  }
]
```

### 3. Test Contribution Endpoint

```bash
curl "https://[your-service]-production.up.railway.app/api/contributions/words"
```

Expected response:
```json
{
  "words": ["HELLO", "GOODBYE", "THANK YOU", ...],
  "total": 1582,
  "stats": {
    "complete": 45,
    "in_progress": 123,
    "not_started": 1414
  }
}
```

### 4. Database Verification

```bash
# Check if database is initialized
railway logs | grep -i "database"

# Expected output:
# ✓ Database initialized
# ✓ Database tables created
```

### 5. Test Static Files (Sign Images)

```bash
# Test if sign images are accessible
curl -I https://[your-service]-production.up.railway.app/sign_images/hello.jpg

# Expected: HTTP 200 OK
```

---

## Troubleshooting

### Check Deployment Logs

```bash
# View recent logs
railway logs --tail 100

# Filter for errors
railway logs | grep -i error

# Filter for warnings
railway logs | grep -i warning

# Watch live logs
railway logs --follow

# Check database initialization
railway logs | grep -E "(Database|init|startup)"
```

---

### Common Error Patterns

#### 1. "Application failed to respond" (502)
**Symptoms**: Service starts but all requests return 502

**Causes**:
- Port mismatch (Railway domain targets wrong port)
- Service not binding to 0.0.0.0 (using localhost instead)
- Healthcheck failing

**Solutions**:
```bash
# 1. Check Railway domain port configuration
railway service

# 2. Verify startCommand in railway.json
cat railway.json | grep startCommand

# 3. Check logs for binding errors
railway logs | grep -i "bind\|port\|listen"

# Expected output:
# INFO:     Uvicorn running on http://0.0.0.0:9000
```

---

#### 2. "$PORT is not a valid integer"
**Symptoms**: Service crashes on startup

**Cause**: Environment variable not expanding in railway.json

**Solution**: Ensure startCommand is wrapped in shell:
```json
"startCommand": "/bin/sh -c \"uvicorn main:app --host 0.0.0.0 --port 9000\""
```

---

#### 3. "No such file or directory: Dockerfile"
**Symptoms**: Build fails immediately

**Cause**: Railway root directory not set correctly

**Solution**:
1. Railway Dashboard → Service → Settings
2. Set "Root Directory" to `backend`
3. Redeploy

---

#### 4. "ModuleNotFoundError: No module named 'X'"
**Symptoms**: Service crashes on startup, missing Python package

**Cause**: Package not in requirements.txt or build failed

**Solutions**:
```bash
# 1. Verify requirements.txt includes package
cat requirements.txt | grep [package-name]

# 2. If missing, add package and redeploy
echo "[package-name]==[version]" >> requirements.txt
git add requirements.txt
git commit -m "fix: add missing package"
git push

# 3. Check build logs for installation errors
railway logs | grep -i "installing\|error"
```

---

#### 5. CORS Errors from Frontend
**Symptoms**: Frontend blocked by CORS policy

**Cause**: Frontend origin not in allowed list

**Solution**:
```python
# Check main.py:76-82
# Regex pattern should cover all Vercel deployments:
allow_origin_regex=r"https://.*\.vercel\.app|http://localhost:\d+"

# If using custom domain, add it explicitly:
allow_origins=[
    "https://your-custom-domain.com",
],
```

---

#### 6. "Cannot connect to database"
**Symptoms**: Service starts but database operations fail

**Cause**: DATABASE_URL not set or incorrect

**Solutions**:
```bash
# 1. Check if PostgreSQL service exists
railway service

# 2. Check DATABASE_URL variable
railway variables | grep DATABASE_URL

# 3. Add PostgreSQL service if missing
# Railway Dashboard → Add Service → PostgreSQL

# 4. Link DATABASE_URL (automatic if in same project)
railway variables set DATABASE_URL="${{Postgres.DATABASE_URL}}"
```

---

#### 7. Static Files Not Found (404 on /sign_images)
**Symptoms**: Sign images return 404

**Cause**: `ghsl_brain/` folder not in deployment

**Solutions**:

**Option A**: Use Railway Volumes (Persistent Storage)
```bash
# 1. Create volume in Railway dashboard
# 2. Mount at /data
# 3. Upload ghsl_brain/ to volume
# 4. main.py will detect /data/ghsl_brain automatically
```

**Option B**: Include in Docker image (not recommended - large size)
```dockerfile
# Add to Dockerfile
COPY ghsl_brain /app/ghsl_brain
```

**Option C**: Use external storage (S3, CloudFlare R2)
```python
# Update main.py to fetch from external storage
SIGN_IMAGES_BASE_URL = os.getenv("SIGN_IMAGES_CDN", "https://cdn.example.com")
```

---

### Verify Configuration

```bash
# Check Railway service settings
railway service

# Check environment variables
railway variables

# Check active deployment
railway status

# View full deployment info
railway status --json

# Force redeploy (if needed)
railway redeploy

# Open service in browser
railway open
```

---

### Manual Redeploy

```bash
# Method 1: Push code changes (automatic)
git add .
git commit -m "fix: description"
git push
# Railway auto-deploys on push to main branch
# Wait ~2-3 minutes for build + deploy

# Method 2: Manual redeploy via CLI
railway redeploy

# Method 3: Manual redeploy via dashboard
# Railway Dashboard → Service → Deployments → Redeploy
```

---

## Production Checklist

Before deploying to production:

### Configuration
- [ ] `railway.json` exists in backend/ folder
- [ ] `railway.json` uses correct `dockerfilePath`: `"Dockerfile"`
- [ ] `startCommand` wraps command in `/bin/sh -c "..."`
- [ ] Port 9000 is hardcoded in `startCommand`
- [ ] No `railway.toml` file exists (delete if present)

### Railway Dashboard
- [ ] Root Directory set to `backend`
- [ ] GitHub repository connected
- [ ] Auto-deploy enabled on `main` branch
- [ ] Railway domain targets port 9000
- [ ] PostgreSQL service added (if using database)
- [ ] `DATABASE_URL` environment variable set

### Application
- [ ] CORS configuration covers all frontend domains
- [ ] Database initialization works on startup
- [ ] Static files accessible (sign images)
- [ ] Health endpoint returns 200 OK
- [ ] Search endpoint returns results
- [ ] Contribution endpoints accessible

### Testing
- [ ] Health check passes
- [ ] Search endpoint tested
- [ ] Contribution endpoints tested
- [ ] Frontend can connect to backend
- [ ] Database queries work
- [ ] Logs show no errors

---

## Deployment Timeline

**Typical deployment**:
1. Push to GitHub: 0s
2. Railway detects push: ~5-10s
3. Docker build: ~2-3 minutes (due to ML dependencies)
4. Deploy + health check: ~10-20s
5. Service available: **~3-4 minutes total**

**First deployment** (cold start):
- May take 5-7 minutes due to:
  - Installing PyTorch (~800MB)
  - Installing sentence-transformers
  - Installing FAISS
  - Loading AI models

**Verify deployment**:
```bash
# Wait for deployment
echo "Waiting for deployment (3-4 minutes)..."
sleep 240

# Test health endpoint
curl https://[your-service]-production.up.railway.app/
```

---

## Continuous Deployment (CI/CD)

### Current Setup: Railway Auto-Deploy

**How it works**:
1. Developer pushes to `main` branch
2. GitHub sends webhook to Railway
3. Railway automatically builds and deploys
4. Service restarts with new version
5. Zero-downtime deployment (Railway handles gracefully)

**No GitHub Actions needed** - Railway handles CI/CD natively!

### Benefits
- ✅ Automatic deployments on every push
- ✅ Build logs visible in Railway dashboard
- ✅ Rollback capability (previous deployments saved)
- ✅ Zero-downtime deployments
- ✅ Automatic environment variable injection
- ✅ Automatic HTTPS certificates

---

## Advanced Configuration

### Using Railway Volumes (Persistent Storage)

For storing `ghsl_brain/` data persistently:

```bash
# 1. Create volume in Railway dashboard
# Name: ghsl-brain-data
# Mount path: /data

# 2. Upload data to volume (via Railway CLI)
railway volume mount ghsl-brain-data
# This opens a shell in the volume
cd /data
# Upload files here

# 3. Application will auto-detect at /data/ghsl_brain
# See main.py:89-94
```

### Environment-Specific Configuration

```python
# main.py - Add environment detection
import os
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    # Production-specific settings
    DEBUG = False
    ALLOW_ORIGINS = [os.getenv("FRONTEND_URL")]
else:
    # Development settings
    DEBUG = True
    ALLOW_ORIGINS = ["*"]
```

---

## Monitoring & Observability

### Railway Built-in Metrics

Railway dashboard provides:
- CPU usage
- Memory usage
- Network traffic
- Request count
- Response times

### Custom Logging

```python
# Add structured logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logging.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.2f}s")
    return response
```

### Health Check Endpoint

Already implemented in `main.py`:

```python
@app.get("/")
async def root():
    return {
        "message": "Ghana Sign Language API",
        "status": "healthy",
        "version": "1.0.0",
        "features": {
            "search": True,
            "contribution": True,
            "training_queue": training_router is not None,
            "format_creator": FORMAT_CREATOR_AVAILABLE,
            "analytics": DASHBOARD_AVAILABLE
        }
    }
```

---

## Security Considerations

### Environment Variables

```bash
# Never commit secrets to git
# Use Railway variables instead

# Set sensitive variables
railway variables set DATABASE_URL="postgresql://..."
railway variables set OPENAI_API_KEY="sk-..."
railway variables set TWILIO_AUTH_TOKEN="..."
```

### CORS Configuration

```python
# Restrict to known domains in production
allow_origin_regex=r"https://signforge\.vercel\.app|https://signforge-.*\.vercel\.app"

# Or use explicit allowlist
allow_origins=[
    "https://signforge.vercel.app",
    "https://signforge-custom.com",
]
```

### Rate Limiting (Future Enhancement)

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/search")
@limiter.limit("60/minute")  # 60 requests per minute
async def search(request: Request, q: str):
    # ...
```

---

## Additional Resources

- [Railway Documentation](https://docs.railway.app)
- [Railway Environment Variables Guide](https://docs.railway.app/guides/variables)
- [Railway Dockerfile Deployment](https://docs.railway.app/guides/dockerfiles)
- [Railway Volumes Guide](https://docs.railway.app/guides/volumes)
- [FastAPI Deployment Best Practices](https://fastapi.tiangolo.com/deployment/)
- [SignForge GitHub Repository](https://github.com/[your-org]/signforge-hackathon)

---

## Change Log

- **2025-11-04**: Initial deployment guide
  - Created railway.json configuration
  - Updated Dockerfile for Railway compatibility
  - Documented port 9000 configuration
  - Added CORS configuration guide
  - Documented Railway Volumes usage
  - Added troubleshooting section

---

*Last updated: 2025-11-04*
*Status: Production-ready*
*Railway Project: invigirating-forginess*
# Railway Deployment Test
