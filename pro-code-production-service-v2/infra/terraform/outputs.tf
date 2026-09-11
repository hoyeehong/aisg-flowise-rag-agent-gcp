output "service_url" {
  description = "Cloud Run URL for the deployed service."
  value       = google_cloud_run_v2_service.agent.uri
}

output "runtime_service_account" {
  description = "Least-privilege identity the service runs as."
  value       = google_service_account.runtime.email
}

output "artifact_registry" {
  description = "Docker repository to push images to."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "database_connection_name" {
  description = "Cloud SQL connection name for the connector."
  value       = google_sql_database_instance.main.connection_name
}

output "secret_ids" {
  description = "Secret Manager ids whose versions must be populated out of band."
  value       = [for s in google_secret_manager_secret.app : s.secret_id]
}

output "postgres_dsn_hint" {
  description = <<-EOT
    Shape of the DSN to store in the postgres-dsn secret. The password is generated in
    state, so it is read with `terraform output -raw generated_db_password` rather than
    printed here.
  EOT
  value       = "postgresql://${google_sql_user.app.name}:<password>@/agent?host=/cloudsql/${google_sql_database_instance.main.connection_name}"
}

output "generated_db_password" {
  description = "Generated application database password."
  value       = random_password.app_user.result
  sensitive   = true
}

output "documents_topic" {
  description = "Topic that ingestion events are published to."
  value       = google_pubsub_topic.documents.id
}

output "documents_subscription" {
  description = "Set as AGENT_PUBSUB_SUBSCRIPTION on the consumer workload."
  value       = google_pubsub_subscription.documents.id
}

output "documents_dead_letter_topic" {
  description = "Set as AGENT_PUBSUB_DEAD_LETTER_TOPIC. Watch its message count: anything here failed every delivery attempt and needs a human."
  value       = google_pubsub_topic.documents_dead_letter.id
}
