#!/usr/bin/env bash
# ==============================================================================
# LADP Essentials — Automated GCP Cloud Run Deployment Script
# Scenario 5: Digital Economy Research & Report Agent (Flowise Multi-Agent + HITL)
# Reference: Module 4 (Evaluations, Deployment, and Responsible AI)
# ==============================================================================

set -euo pipefail

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
FLOWISE_PASSWORD="${FLOWISE_PASSWORD:-}"
GENERATED_PASSWORD="false"
if [ -z "$FLOWISE_PASSWORD" ]; then
    read -rsp "Enter Flowise Admin Password (leave blank to generate a random one): " FLOWISE_PASSWORD
    echo ""
    if [ -z "$FLOWISE_PASSWORD" ]; then
        FLOWISE_PASSWORD=$(openssl rand -base64 24)
        GENERATED_PASSWORD="true"
    fi
fi
CPU_LIMIT="${CPU_LIMIT:-2}"
MEMORY_LIMIT="${MEMORY_LIMIT:-2Gi}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"

# Flowise persists to SQLite on a GCS FUSE mount. FUSE does not provide the POSIX
# advisory locking SQLite requires, so two instances writing concurrently can corrupt
# the database. MAX_INSTANCES is therefore 1 by default. Raising it requires migrating
# DATABASE_TYPE to Postgres (Cloud SQL) first -- see README_DEPLOY_GCP.md.
MAX_INSTANCES="${MAX_INSTANCES:-1}"

# Pin by digest in production: FLOWISE_IMAGE="flowiseai/flowise@sha256:<digest>"
FLOWISE_IMAGE="${FLOWISE_IMAGE:-flowiseai/flowise:3.1.2}"

# The Flowise *builder UI* is served from the same origin as the chat widget, so
# public access exposes the flow editor, not just the chatbot. Kept enabled by
# default because the published demo depends on it; set to "false" for any
# deployment holding real data.
ALLOW_UNAUTHENTICATED="${ALLOW_UNAUTHENTICATED:-true}"

echo -e "${GREEN}✓ Region:${NC} $GCP_REGION (Singapore)"
echo -e "${GREEN}✓ Service Name:${NC} $SERVICE_NAME"
echo -e "${GREEN}✓ Storage Bucket:${NC} gs://$BUCKET_NAME"

# 5. Enable Required GCP APIs
echo -e "\n${BLUE}--> [Step 1/5] Enabling GCP APIs (Cloud Run, Storage, Secret Manager, IAM)...${NC}"
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
# NOTE ON FLOWISE CREDENTIALS
# The agents in flowise_scenario_5_workflow.json run on Groq (openai/gpt-oss-20b).
# Flowise stores per-node provider credentials in its own encrypted credential store,
# entered through the UI -- these Secret Manager entries do not auto-wire those nodes.
# They are provisioned so the keys live in one managed place rather than in a local
# .env, and so the evaluation suite can read them.
#   GROQ_API_KEY    -> the system under test (the agents)
#   GEMINI_API_KEY  -> the evaluation judge only (deliberately a different model family
#                      from the system, to avoid self-preference bias when grading)

if [ -z "${GROQ_API_KEY:-}" ]; then
    read -rsp "Enter your Groq API Key (system under test, blank to skip): " GROQ_API_KEY
    echo ""
fi
if [ -n "${GROQ_API_KEY:-}" ]; then
    create_or_update_secret "flowise-groq-api-key" "$GROQ_API_KEY"
fi

if [ -z "${GEMINI_API_KEY:-}" ]; then
    read -rsp "Enter your Google Gemini API Key (eval judge, blank to skip): " GEMINI_API_KEY
    echo ""
fi
if [ -n "${GEMINI_API_KEY:-}" ]; then
    create_or_update_secret "flowise-gemini-api-key" "$GEMINI_API_KEY"
fi

# Admin password goes to Secret Manager, NOT --set-env-vars. Environment variables on a
# Cloud Run revision are readable via `gcloud run services describe`, in revision
# metadata, and in Cloud Audit Logs -- which would make the README's "zero plaintext
# secrets" claim false.
create_or_update_secret "flowise-admin-password" "$FLOWISE_PASSWORD"

SECRETS_PARAM="FLOWISE_PASSWORD=flowise-admin-password:latest"
if [ -n "${GROQ_API_KEY:-}" ]; then
    SECRETS_PARAM="${SECRETS_PARAM},GROQ_API_KEY=flowise-groq-api-key:latest"
fi
if [ -n "${GEMINI_API_KEY:-}" ]; then
    SECRETS_PARAM="${SECRETS_PARAM},GEMINI_API_KEY=flowise-gemini-api-key:latest"
fi

# Optional Pinecone API Key
if [ -n "${PINECONE_API_KEY:-}" ]; then
    create_or_update_secret "flowise-pinecone-api-key" "$PINECONE_API_KEY"
    SECRETS_PARAM="${SECRETS_PARAM},PINECONE_API_KEY=flowise-pinecone-api-key:latest"
fi

# Dedicated least-privilege service account for this service.
# The default compute SA is deliberately NOT used: it is shared by every workload in
# the project and accumulates broad roles. No project-level grants are made here --
# each secret is bound individually, and bucket access is scoped to one bucket.
RUN_SA_NAME="${SERVICE_NAME}-sa"
RUN_SA="${RUN_SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

gcloud services enable iam.googleapis.com --project="$GCP_PROJECT_ID" --quiet

if ! gcloud iam service-accounts describe "$RUN_SA" --project="$GCP_PROJECT_ID" &>/dev/null; then
    echo "Creating service account: $RUN_SA"
    gcloud iam service-accounts create "$RUN_SA_NAME" \
        --display-name="Cloud Run runtime SA for ${SERVICE_NAME}" \
        --project="$GCP_PROJECT_ID"
    # IAM propagation is eventually consistent; brief settle before binding.
    sleep 10
else
    echo -e "${GREEN}✓ Service account ${RUN_SA} already exists.${NC}"
fi

grant_secret_access() {
    local secret_name="$1"
    echo "Binding secretAccessor on ${secret_name} -> ${RUN_SA}"
    gcloud secrets add-iam-policy-binding "$secret_name" \
        --member="serviceAccount:${RUN_SA}" \
        --role="roles/secretmanager.secretAccessor" \
        --project="$GCP_PROJECT_ID" --quiet >/dev/null
}

grant_secret_access "flowise-admin-password"
if [ -n "${GROQ_API_KEY:-}" ]; then
    grant_secret_access "flowise-groq-api-key"
fi
if [ -n "${GEMINI_API_KEY:-}" ]; then
    grant_secret_access "flowise-gemini-api-key"
fi
if [ -n "${PINECONE_API_KEY:-}" ]; then
    grant_secret_access "flowise-pinecone-api-key"
fi

# Bucket-scoped object access only (not project-wide storage admin).
echo "Binding objectAdmin on gs://${BUCKET_NAME} -> ${RUN_SA}"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
    --member="serviceAccount:${RUN_SA}" \
    --role="roles/storage.objectAdmin" \
    --project="$GCP_PROJECT_ID" --quiet >/dev/null

# 8. Deploy Container to Cloud Run
echo -e "\n${BLUE}--> [Step 4/5] Deploying Flowise container to Google Cloud Run...${NC}"

# Resolve the tag to an immutable digest so a re-deploy cannot silently pull
# different bits under the same tag. Requires docker; falls back with a warning.
DEPLOY_IMAGE="$FLOWISE_IMAGE"
if [[ "$FLOWISE_IMAGE" != *"@sha256:"* ]]; then
    if command -v docker &>/dev/null; then
        RESOLVED_DIGEST=$(docker buildx imagetools inspect "$FLOWISE_IMAGE" 2>/dev/null \
            | awk '/^Digest:/{print $2; exit}' || true)
        if [ -n "${RESOLVED_DIGEST:-}" ]; then
            DEPLOY_IMAGE="${FLOWISE_IMAGE%%:*}@${RESOLVED_DIGEST}"
            echo -e "${GREEN}✓ Pinned image to digest:${NC} $DEPLOY_IMAGE"
        fi
    fi
    if [[ "$DEPLOY_IMAGE" != *"@sha256:"* ]]; then
        echo -e "${YELLOW}[WARN] Deploying by mutable tag '${FLOWISE_IMAGE}'. For a reproducible"
        echo -e "       deployment, set FLOWISE_IMAGE=flowiseai/flowise@sha256:<digest>.${NC}"
    fi
fi

AUTH_FLAG="--no-allow-unauthenticated"
if [ "$ALLOW_UNAUTHENTICATED" = "true" ]; then
    AUTH_FLAG="--allow-unauthenticated"
    echo -e "${YELLOW}[WARN] Service will be PUBLIC. This exposes the Flowise flow editor, not"
    echo -e "       just the chat widget. Set ALLOW_UNAUTHENTICATED=false for private use.${NC}"
fi

gcloud run deploy "$SERVICE_NAME" \
    --image="$DEPLOY_IMAGE" \
    --project="$GCP_PROJECT_ID" \
    --region="$GCP_REGION" \
    --platform="managed" \
    --service-account="$RUN_SA" \
    "$AUTH_FLAG" \
    --port=3000 \
    --cpu="$CPU_LIMIT" \
    --memory="$MEMORY_LIMIT" \
    --min-instances="$MIN_INSTANCES" \
    --max-instances="$MAX_INSTANCES" \
    --execution-environment="gen2" \
    --set-env-vars="FLOWISE_USERNAME=${FLOWISE_USERNAME},DATABASE_TYPE=sqlite" \
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
echo -e " 🔑 Admin Password: stored in Secret Manager as 'flowise-admin-password'"
echo -e "                    retrieve with:"
echo -e "                    gcloud secrets versions access latest --secret=flowise-admin-password"
echo -e " 👤 Runtime SA:      ${RUN_SA}"
echo -e " 🐳 Image:           ${DEPLOY_IMAGE}"
echo -e " 📂 Persistent Data: gs://${BUCKET_NAME} (mounted at /root/.flowise)"
if [ "$GENERATED_PASSWORD" = "true" ]; then
    echo -e "\n${YELLOW} A random admin password was generated. It was not printed here;"
    echo -e " read it from Secret Manager using the command above.${NC}"
fi
echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e " 1. Open ${SERVICE_URL} in your browser."
echo -e " 2. Log in using the admin credentials above."
echo -e " 3. Go to Agentflows -> Settings -> Load / Import Chatflow."
echo -e " 4. Import 'flowise_scenario_5_workflow.json'."
echo -e " 5. In Document Stores, upload 'data/imda_report.pdf' & Save & Upsert."
echo -e "${GREEN}======================================================================${NC}\n"
