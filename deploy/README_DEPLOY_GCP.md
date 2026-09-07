# Deploying Flowise Scenario 5 to Google Cloud Platform (Cloud Run)

This guide documents the end-to-end deployment of **Scenario 5: Digital Economy Research & Report Agent** to **Google Cloud Run** in the Singapore region (`asia-southeast1`), referencing the deployment, security, and governance principles from **AISG Module 4 (Evaluations, Deployment, and Responsible AI)**.

---

## 🏗️ Architecture Overview

```
                                  ┌─────────────────────────────────────────────────────────┐
                                  │               Google Cloud Platform (GCP)               │
                                  │                  Region: asia-southeast1                │
                                  └─────────────────────────────────────────────────────────┘
                                                               │
     [End-User / Web Browser] ────────── (HTTPS / TLS) ────────┼
                                                               │
                                                               ▼
                                               ┌───────────────────────────────┐
                                               │       Google Cloud Run        │
                                               │ (flowiseai/flowise:2.1.4)     │
                                               └───────────────┬───────────────┘
                                                               │
                     ┌─────────────────────────────────────────┼─────────────────────────────────────────┐
                     ▼                                         ▼                                         ▼
       ┌───────────────────────────┐             ┌───────────────────────────┐             ┌───────────────────────────┐
       │   Google Secret Manager   │             │   Cloud Storage Bucket    │             │    External AI Engines    │
       │   - GEMINI_API_KEY        │             │   - gs://flowise-data-... │             │   - Google Gemini 3 Flash │
       │   - PINECONE_API_KEY      │             │   (Mounted to /root/.flowise)           │   - text-embedding-004    │
       │                           │             │   (Persistent SQLite DB)  │             │   - Pinecone Vector Store │
       └───────────────────────────┘             └───────────────────────────┘             └───────────────────────────┘
```

---

## 🚀 Quick Start (Automated Deployment)

### Prerequisites
1. **Google Cloud SDK (`gcloud`):** [Install gcloud CLI](https://cloud.google.com/sdk/docs/install) if not already installed.
2. **GCP Project:** An active GCP project with billing enabled.
3. **Google Gemini API Key:** From [Google AI Studio](https://aistudio.google.com/).

---

### Step 1: Configure Environment (Optional)
Copy the example environment file and fill in your custom credentials:
```bash
cp deploy/env.example deploy/.env
```
*(If you do not create a `.env` file, the script will prompt you interactively).*

---

### Step 2: Run Deployment Script
Execute the deployment script from your project root:
```bash
./deploy/deploy_gcp.sh
```

The script automatically:
1. Enables required GCP APIs (`run`, `storage`, `secretmanager`).
2. Creates a Cloud Storage bucket in `asia-southeast1` (`gs://flowise-data-<PROJECT_ID>`).
3. Securely stores your API keys in **Google Secret Manager** (Zero plaintext secrets).
4. Deploys the Flowise container to **Cloud Run** with GCS FUSE volume mount (`/root/.flowise`) for database persistence.
5. Outputs your live HTTPS URL.

---

## 🛠️ Post-Deployment: Initializing Flowise Cloud

1. **Access the Live URL:**
   Open the Cloud Run HTTPS URL provided at the end of the deployment script.

2. **Log In:**
   Enter the `FLOWISE_USERNAME` and `FLOWISE_PASSWORD` configured during deployment.

3. **Import Scenario 5 Workflow:**
   * Go to **Agentflows** $\rightarrow$ Click **Add New** $\rightarrow$ Click **Settings (Gear Icon)** $\rightarrow$ **Load / Import Chatflow**.
   * Upload [`flowise_scenario_5_workflow.json`](../flowise_scenario_5_workflow.json).

4. **Initialize Document Store:**
   * Go to **Document Stores** $\rightarrow$ Click **Add New** $\rightarrow$ Name: `imda_sea_digital_economy_report`.
   * **Document Loader:** PDF File Loader $\rightarrow$ Upload [`imda_report.pdf`](../imda_report.pdf).
   * **Text Splitter:** Recursive Character Text Splitter (`Chunk Size: 1000`, `Overlap: 200`).
   * **Embeddings:** Google GenerativeAI Embeddings (`text-embedding-004`).
   * **Vector Store:** In-Memory *(or Pinecone Cloud if configured)*.
   * Click **Save & Upsert Chunk**.

5. **Test Live Agent:**
   Open the chat interface and submit:
   > *"Write a brief report on the shift from 'Tech for Growth' to 'Tech for Good' in Southeast Asia."*

---

## 🔒 Security & Responsible AI (Module 4.4 Checklist)

| Security Aspect | Implementation in this Deployment |
| :--- | :--- |
| **Data Protection & Encryption** | Automated HTTPS / TLS 1.3 encryption on Cloud Run; encrypted storage at rest in GCS. |
| **Zero Plaintext Secrets** | All API keys are injected at container runtime using **Google Secret Manager**. |
| **Authentication & Access Control** | Admin dashboard (`/canvas`) protected by HTTP Basic Auth; external predictions can be guarded by Flowise API Keys. |
| **Data Residency** | All container compute, storage buckets, and model endpoints reside in Singapore (`asia-southeast1`). |
| **Cost & Abuse Protection** | Auto-scaling configured to scale to 0 instances when idle (`min-instances: 0`, `max-instances: 5`). |

---

## ⚙️ Operational Commands & Maintenance

### View Live Container Logs
```bash
gcloud run services logs tail flowise-capstone --region=asia-southeast1
```

### Update Flowise Password or Environment Variables
```bash
gcloud run services update flowise-capstone \
    --region=asia-southeast1 \
    --update-env-vars="FLOWISE_PASSWORD=NewStrongPassword2026!"
```

### Inspect Stored Database Files in GCS Bucket
```bash
gcloud storage ls --long gs://flowise-data-<YOUR_PROJECT_ID>/
```

### Tear Down Resources (To Stop All Cloud Costs)
```bash
gcloud run services delete flowise-capstone --region=asia-southeast1 --quiet
gcloud storage rm --recursive gs://flowise-data-<YOUR_PROJECT_ID>
```
