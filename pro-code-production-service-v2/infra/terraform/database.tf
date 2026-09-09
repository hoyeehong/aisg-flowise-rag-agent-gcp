# Cloud SQL Postgres with pgvector.
#
# Postgres rather than a dedicated vector database because this corpus also needs
# relational filters (tenant, source, page) and full-text search. Running one engine for
# vectors, lexical search and metadata avoids keeping two stores consistent, which at
# this scale costs more than the marginal vector performance difference.

resource "google_sql_database_instance" "main" {
  name             = "${var.service_name}-pg"
  database_version = "POSTGRES_17"
  region           = var.region

  # The database holds paused human-review runs. Losing it discards work awaiting a
  # reviewer, so deletion protection defaults on and must be turned off deliberately.
  deletion_protection = var.database_deletion_protection

  settings {
    tier              = var.database_tier
    availability_type = "ZONAL"
    disk_type         = "PD_SSD"
    disk_autoresize   = true
    user_labels       = var.labels

    database_flags {
      # pgvector ships with Cloud SQL for Postgres but must be allow-listed before
      # CREATE EXTENSION succeeds. Without this the service's first ensure_schema()
      # fails with an error that does not mention the flag.
      name  = "cloudsql.enable_pgvector"
      value = "on"
    }

    backup_configuration {
      enabled                        = true
      start_time                     = "18:00" # 02:00 SGT, outside working hours
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
    }

    ip_configuration {
      # No public IP. Cloud Run reaches the instance over the Cloud SQL connector using
      # the runtime service account, so exposing an internet-facing address would add
      # attack surface for no benefit.
      ipv4_enabled = false
      # Private Service Access must already exist on this network; see the module README.
      private_network                               = "projects/${var.project_id}/global/networks/default"
      enable_private_path_for_google_cloud_services = true
    }

    insights_config {
      query_insights_enabled = true
      # Query text is recorded for diagnostics. record_application_tags is off because
      # the tags can carry user-supplied values.
      record_application_tags = false
      record_client_address   = false
    }

    maintenance_window {
      day  = 7 # Sunday
      hour = 19
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_sql_database" "agent" {
  name     = "agent"
  instance = google_sql_database_instance.main.name
}

resource "random_password" "app_user" {
  length  = 32
  special = false # Avoids quoting problems inside a DSN
}

resource "google_sql_user" "app" {
  name     = "agent_app"
  instance = google_sql_database_instance.main.name
  password = random_password.app_user.result
}
