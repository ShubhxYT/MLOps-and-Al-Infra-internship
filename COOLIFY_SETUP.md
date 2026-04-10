# Coolify Deployment Guide — MemO-Al Voice Agent

## Prerequisites
- Coolify instance running and accessible
- Docker support enabled on your Coolify server
- GROQ_API_KEY obtained from [console.groq.com](https://console.groq.com)

## Deployment Steps

### 1. Add Application to Coolify

1. Go to your Coolify dashboard
2. Click **New Application**
3. Select **Docker Compose** or **Docker** build method
4. Choose your Git repository (GitHub/GitLab/Gitea)
5. Select the branch (e.g., `main`)

### 2. Configure Build Settings

- **Build Pack**: Docker
- **Dockerfile Path**: `./Dockerfile` (default)
- **Docker Compose File**: `./docker-compose.yml` (if using Compose)

### 3. Add Environment Variables

In Coolify's environment variables section, add:

```
GROQ_API_KEY=your_actual_groq_api_key
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=8501
```

### 4. Set Port & Domain

- **Container Port**: 8501
- **Public Port**: 80 (or 443 for HTTPS)
- **Domain**: Set your custom domain (e.g., `memo-al.yourdomain.com`)

### 5. Configure Volumes (Optional)

If you want persistent storage for session history:
- **Container Path**: `/app/output`
- **Volume Mount**: Create a persistent volume

### 6. Deploy

1. Click **Deploy** in Coolify
2. Monitor logs in real-time
3. Once deployed, access via your configured domain

## Health & Monitoring

Coolify will automatically check the health endpoint:
- Health Check: `http://localhost:8501/_stcore/health`
- Will auto-restart on failure
- Logs visible in Coolify dashboard

## Troubleshooting

### Application fails to start
- Check GROQ_API_KEY is set correctly
- View logs in Coolify dashboard for detailed errors
- Ensure port 8501 is not blocked

### Whisper model not downloading
- First startup takes ~30-60s to download the base model (~145 MB)
- Subsequent restarts are instant (model is cached in container)
- If needed, extend the startup period in health checks

### Audio/Upload not working
- Ensure `/app/output` directory has write permissions
- Check volume mount is correctly configured in Coolify

## Local Testing Before Deployment

Test the Docker build locally:

```bash
# Build the image
docker build -t memo-al .

# Run with environment variable
docker run -p 8501:8501 \
  -e GROQ_API_KEY=your_key_here \
  memo-al
```

Then open `http://localhost:8501`

## Rollback

If deployment has issues:
1. Go to Coolify dashboard
2. Click the application
3. Select a previous deployment from history
4. Click **Rollback**

## Updating the Application

When you push new changes to your Git branch:
1. Coolify can be configured for CI/CD (auto-deploy on push)
2. Or manually trigger a rebuild from the Coolify dashboard
3. Previous versions are kept for quick rollback

## Performance Notes

- **Cold Start**: ~30-60 seconds (downloads Whisper model on first run)
- **Warm Start**: ~5-10 seconds
- **Memory Usage**: Base Streamlit ~200-300 MB + model cache
- **Whisper Model Size**: ~145 MB (base model, runs on CPU)

## Security

- `.env` is in `.gitignore` — never commit secrets
- Use Coolify's secrets management for GROQ_API_KEY
- All file operations in `output/` directory are sandboxed
- CSRF protection enabled in Streamlit config
