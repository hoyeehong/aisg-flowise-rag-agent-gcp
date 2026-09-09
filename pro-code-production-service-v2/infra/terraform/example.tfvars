# Copy to terraform.tfvars (gitignored) and fill in.
project_id = "your-gcp-project-id"
region     = "asia-southeast1"

# Pin by digest, not tag: a Cloud Run revision must be reproducible from this state.
image = "asia-southeast1-docker.pkg.dev/your-gcp-project-id/digital-economy-agent/digital-economy-agent@sha256:..."

# Explicit, because this API starts runs that spend model tokens: an open endpoint is a
# billing exposure as well as a data one.
allow_unauthenticated = false

database_tier                = "db-custom-1-3840"
database_deletion_protection = true
min_instances                = 0
max_instances                = 5
