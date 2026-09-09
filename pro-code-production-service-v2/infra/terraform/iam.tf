# A dedicated runtime identity. The default compute service account is shared by every
# workload in the project and accumulates broad roles, so binding secrets to it grants
# far more than this service needs.

resource "google_service_account" "runtime" {
  account_id   = "${var.service_name}-run"
  display_name = "Cloud Run runtime identity for ${var.service_name}"
  description  = "Least-privilege identity: per-secret access and Cloud SQL client only"

  depends_on = [google_project_service.required]
}

# Cloud SQL connectivity. Scoped to the project because the role has no resource-level
# form; it grants connection, not data access, which is governed by the database user.
resource "google_project_iam_member" "runtime_sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = google_service_account.runtime.member
}

# Telemetry export. Without these the service can emit traces and metrics that are
# silently dropped, which is worse than not emitting them.
resource "google_project_iam_member" "runtime_trace" {
  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = google_service_account.runtime.member
}

resource "google_project_iam_member" "runtime_metrics" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = google_service_account.runtime.member
}
