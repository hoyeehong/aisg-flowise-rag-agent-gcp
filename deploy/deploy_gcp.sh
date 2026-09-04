#!/usr/bin/env bash
# ==============================================================================
# LADP Essentials — Automated GCP Cloud Run Deployment Script
# Scenario 5: Digital Economy Research & Report Agent (Flowise Multi-Agent + HITL)
# Reference: Module 4 (Evaluations, Deployment, and Responsible AI)
# ==============================================================================

set -eo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}   LADP Essentials: Deploying Flowise Scenario 5 to Google Cloud Run   ${NC}"
echo -e "${BLUE}======================================================================${NC}\n"

# 1. Load .env if present in deploy folder or root
if [ -f "$(dirname "$0")/.env" ]; then
    echo -e "${GREEN}✓ Loading environment variables from deploy/.env${NC}"
    # shellcheck disable=SC1091
    source "$(dirname "$0")/.env"
elif [ -f "$(dirname "$0")/../.env" ]; then
    echo -e "${GREEN}✓ Loading environment variables from root .env${NC}"
    # shellcheck disable=SC1091
    source "$(dirname "$0")/../.env"
fi

# 2. Check gcloud CLI installation
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}[ERROR] gcloud CLI is not installed or not in PATH.${NC}"
    echo "Please install the Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# 3. Resolve GCP Project ID
GCP_PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
if [ -z "$GCP_PROJECT_ID" ] || [ "$GCP_PROJECT_ID" = "(unset)" ]; then
    read -rp "Enter your GCP Project ID: " GCP_PROJECT_ID
    gcloud config set project "$GCP_PROJECT_ID"
fi
echo -e "${GREEN}✓ Active GCP Project:${NC} $GCP_PROJECT_ID"

# 4. Default settings
GCP_REGION="${GCP_REGION:-asia-southeast1}"
SERVICE_NAME="${SERVICE_NAME:-flowise-capstone}"
BUCKET_NAME="${BUCKET_NAME:-flowise-data-${GCP_PROJECT_ID}}"
FLOWISE_USERNAME="${FLOWISE_USERNAME:-admin}"
if [ -z "$FLOWISE_PASSWORD" ]; then
    read -rsp "Enter Flowise Admin Password (leave blank to generate random): " FLOWISE_PASSWORD
    echo ""
    if [ -z "$FLOWISE_PASSWORD" ]; then
        FLOWISE_PASSWORD=$(openssl rand -base64 16)
        echo -e "${YELLOW}Generated random admin password: ${FLOWISE_PASSWORD}${NC}"
    fi
fi
CPU_LIMIT="${CPU_LIMIT:-2}"
MEMORY_LIMIT="${MEMORY_LIMIT:-2Gi}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-5}"

echo -e "${GREEN}✓ Region:${NC} $GCP_REGION (Singapore)"
echo -e "${GREEN}✓ Service Name:${NC} $SERVICE_NAME"
echo -e "${GREEN}✓ Storage Bucket:${NC} gs://$BUCKET_NAME"

# 5. Enable Required GCP APIs
echo -e "\n${BLUE}--> [Step 1/5] Enabling GCP APIs (Cloud Run, Storage, Secret Manager)...${NC}"
gcloud services enable \
    run.googleapis.com \
    storage.googleapis.com \
    secretmanager.googleapis.com \
    artifactregistry.googleapis.com \
    --project="$GCP_PROJECT_ID"

# 6. Create GCS Bucket for Flowise persistent state if it doesn't exist
echo -e "\n${BLUE}--> [Step 2/5] Checking Cloud Storage Bucket for Flowise persistence...${NC}"
if ! gcloud storage buckets describe "gs://${BUCKET_NAME}" --project="$GCP_PROJECT_ID" &>/dev/null; then
    echo "Creating bucket gs://${BUCKET_NAME} in ${GCP_REGION}..."
    gcloud storage buckets create "gs://${BUCKET_NAME}" \
        --project="$GCP_PROJECT_ID" \
        --location="$GCP_REGION" \
        --uniform-bucket-level-access
    echo -e "${GREEN}✓ Bucket gs://${BUCKET_NAME} created.${NC}"
else
    echo -e "${GREEN}✓ Bucket gs://${BUCKET_NAME} already exists.${NC}"
fi

# 7. Configure Secrets in Google Secret Manager
echo -e "\n${BLUE}--> [Step 3/5] Configuring Secrets in Google Secret Manager...${NC}"

create_or_update_secret() {
    local secret_name="$1"
    local secret_value="$2"

    if ! gcloud secrets describe "$secret_name" --project="$GCP_PROJECT_ID" &>/dev/null; then
        echo "Creating secret: $secret_name"
        echo -n "$secret_value" | gcloud secrets create "$secret_name" \
            --data-file=- \
            --project="$GCP_PROJECT_ID" \
            --replication-policy="automatic"
    else
        echo "Updating secret version: $secret_name"
        echo -n "$secret_value" | gcloud secrets versions add "$secret_name" \
            --data-file=- \
            --project="$GCP_PROJECT_ID"
    fi
}

# Prompt for Gemini API Key if not set
if [ -z "$GEMINI_API_KEY" ]; then
    read -rsp "Enter your Google Gemini API Key: " GEMINI_API_KEY
    echo ""
fi
create_or_update_secret "flowise-gemini-api-key" "$GEMINI_API_KEY"

# Optional Pinecone API Key
if [ -n "$PINECONE_API_KEY" ]; then
    create_or_update_secret "flowise-pinecone-api-key" "$PINECONE_API_KEY"
    SECRETS_PARAM="GEMINI_API_KEY=flowise-gemini-api-key:latest,PINECONE_API_KEY=flowise-pinecone-api-key:latest"
else
    SECRETS_PARAM="GEMINI_API_KEY=flowise-gemini-api-key:latest"
fi

# Grant Cloud Run default service account access to Secret Manager
PROJECT_NUMBER=$(gcloud projects describe "$GCP_PROJECT_ID" --format="value(projectNumber)")
RUN_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo -e "Granting Secret Accessor role to Service Account: ${RUN_SA}..."
gcloud secrets add-iam-policy-binding flowise-gemini-api-key \
    --member="serviceAccount:${RUN_SA}" \
    --role="roles/secretmanager.secretAccessor" \
    --project="$GCP_PROJECT_ID" --quiet &>/dev/null || true

if [ -n "$PINECONE_API_KEY" ]; then
    gcloud secrets add-iam-policy-binding flowise-pinecone-api-key \
        --member="serviceAccount:${RUN_SA}" \
        --role="roles/secretmanager.secretAccessor" \
        --project="$GCP_PROJECT_ID" --quiet &>/dev/null || true
fi

# Also grant project-level Secret Accessor to ensure all secret versions are accessible
gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:${RUN_SA}" \
    --role="roles/secretmanager.secretAccessor" \
    --condition=None --quiet &>/dev/null || true

# 8. Deploy Container to Cloud Run
echo -e "\n${BLUE}--> [Step 4/5] Deploying Flowise container to Google Cloud Run...${NC}"
gcloud run deploy "$SERVICE_NAME" \
    --image="flowiseai/flowise:3.1.2" \
    --project="$GCP_PROJECT_ID" \
    --region="$GCP_REGION" \
    --platform="managed" \
    --allow-unauthenticated \
    --port=3000 \
    --cpu="$CPU_LIMIT" \
    --memory="$MEMORY_LIMIT" \
    --min-instances="$MIN_INSTANCES" \
    --max-instances="$MAX_INSTANCES" \
    --execution-environment="gen2" \
    --set-env-vars="FLOWISE_USERNAME=${FLOWISE_USERNAME},FLOWISE_PASSWORD=${FLOWISE_PASSWORD},DATABASE_TYPE=sqlite" \
    --set-secrets="$SECRETS_PARAM" \
    --add-volume="name=flowise-persist,type=cloud-storage,bucket=${BUCKET_NAME}" \
    --add-volume-mount="volume=flowise-persist,mount-path=/root/.flowise"

# 9. Get Live Endpoint URL
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --project="$GCP_PROJECT_ID" --region="$GCP_REGION" --format="value(status.url)")

echo -e "\n${BLUE}--> [Step 5/5] Deployment Completed Successfully!${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN} 🎉 Flowise Live URL: ${SERVICE_URL}${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo -e " 🔐 Admin Username: ${FLOWISE_USERNAME}"
echo -e " 🔑 Admin Password: ${FLOWISE_PASSWORD}"
echo -e " 📂 Persistent Data: gs://${BUCKET_NAME} (mounted at /root/.flowise)"
echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e " 1. Open ${SERVICE_URL} in your browser."
echo -e " 2. Log in using the admin credentials above."
echo -e " 3. Go to Agentflows -> Settings -> Load / Import Chatflow."
echo -e " 4. Import 'scenario_5_imda_digital_economy_agentflow3.json'."
echo -e " 5. In Document Stores, upload 'imda_report.pdf' & Save & Upsert."
echo -e "${GREEN}======================================================================${NC}\n"
