# Secret *containers* are managed here; their values are not.
#
# Putting a credential in a Terraform variable puts it in the state file, in every plan
# output, and in CI logs. These resources therefore create empty secrets, and the
# versions are added out of band:
#
#   gcloud secrets versions add digital-economy-agent-groq-api-key --data-file=-
#
# `ignore_changes` on the version keeps a rotated credential from being reverted by the
# next apply.

locals {
  secrets = {
    groq-api-key   = "Groq API key: the chat models the agents run on"
    gemini-api-key = "Google API key: embeddings, and the evaluation judge"
    postgres-dsn   = "Full Postgres DSN including the password"
  }
}

resource "google_secret_manager_secret" "app" {
  for_each = local.secrets

  secret_id = "${var.service_name}-${each.key}"
  labels    = var.labels

  replication {
    user_managed {
      replicas {
        # Pinned to the service region rather than automatic: data residency is a
        # requirement for this workload, and automatic replication spans regions.
        location = var.region
      }
    }
  }

  depends_on = [google_project_service.required]
}

# One binding per secret, rather than a project-level grant. A project-level
# secretAccessor role would let this service read every secret in the project,
# including those belonging to unrelated workloads.
resource "google_secret_manager_secret_iam_member" "runtime_access" {
  for_each = google_secret_manager_secret.app

  secret_id = each.value.id
  role      = "roles/secretmanager.secretAccessor"
  member    = google_service_account.runtime.member
}
