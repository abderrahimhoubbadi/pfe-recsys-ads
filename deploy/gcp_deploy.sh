#!/bin/bash
# ═══════════════════════════════════════════════════════════
# GCP Deployment Script — Option B: Cloud Production
# ═══════════════════════════════════════════════════════════
# Prerequisites:
#   1. gcloud CLI installed and authenticated
#   2. GCP project with Pub/Sub, Memorystore, Cloud Run APIs enabled
#   3. Artifact Registry repository created
# ═══════════════════════════════════════════════════════════

set -e

# ── Configuration ──
PROJECT_ID="${GCP_PROJECT_ID:-devoteam-pfe-recsys}"
REGION="${GCP_REGION:-europe-west1}"
SERVICE_NAME="recsys-api"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "═══════════════════════════════════════════════════"
echo " Deploying to GCP — Project: ${PROJECT_ID}"
echo "═══════════════════════════════════════════════════"

# ── Step 1: Enable required APIs ──
echo "📦 Enabling GCP APIs..."
gcloud services enable \
  run.googleapis.com \
  pubsub.googleapis.com \
  redis.googleapis.com \
  cloudbuild.googleapis.com \
  --project="${PROJECT_ID}"

# ── Step 2: Create Pub/Sub topics ──
echo "📨 Creating Pub/Sub topics..."
for TOPIC in impressions decisions feedback new_ads; do
  gcloud pubsub topics create ${TOPIC} --project="${PROJECT_ID}" 2>/dev/null || echo "  Topic '${TOPIC}' already exists"
  gcloud pubsub subscriptions create ${TOPIC}-sub --topic=${TOPIC} --project="${PROJECT_ID}" 2>/dev/null || echo "  Subscription '${TOPIC}-sub' already exists"
done

# ── Step 3: Create Memorystore Redis instance ──
echo "🗄️ Creating Memorystore instance..."
gcloud redis instances create recsys-state \
  --size=1 \
  --region="${REGION}" \
  --redis-version=redis_7_0 \
  --project="${PROJECT_ID}" 2>/dev/null || echo "  Memorystore instance already exists"

REDIS_HOST=$(gcloud redis instances describe recsys-state \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(host)" 2>/dev/null || echo "10.0.0.3")

echo "  Redis host: ${REDIS_HOST}"

# ── Step 4: Build and push Docker image ──
echo "🐳 Building and pushing Docker image..."
gcloud builds submit \
  --tag "${IMAGE}" \
  --project="${PROJECT_ID}"

# ── Step 5: Deploy to Cloud Run ──
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --platform managed \
  --region "${REGION}" \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10 \
  --timeout 60 \
  --set-env-vars "\
REDIS_HOST=${REDIS_HOST},\
REDIS_PORT=6379,\
GCP_PROJECT_ID=${PROJECT_ID},\
QUEUE_BACKEND=pubsub,\
STATE_BACKEND=redis,\
MODEL_TYPE=H-DeepBandit,\
MOO_POLICY=epsilon_constraint" \
  --allow-unauthenticated \
  --project="${PROJECT_ID}"

# ── Step 6: Get service URL ──
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(status.url)")

echo ""
echo "═══════════════════════════════════════════════════"
echo " ✅ Deployment complete!"
echo " Service URL: ${SERVICE_URL}"
echo " Health check: ${SERVICE_URL}/health"
echo " API docs: ${SERVICE_URL}/docs"
echo "═══════════════════════════════════════════════════"
