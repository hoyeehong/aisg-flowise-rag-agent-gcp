# Cloud Run service.
#
# Cloud Run rather than GKE for this deployment: the workload is a stateless HTTP
# service whose state lives in Cloud SQL, and it scales to zero between demos. The Helm
# chart in charts/ covers GKE for a cluster deployment; both consume the same image.

resource "google_cloud_run_v2_service" "agent" {
  name     = var.service_name
  location = var.region
  labels   = var.labels

  # Requests only. The service does no background work, so billing for idle CPU would
  # pay for nothing.
  deletion_protection = false
  ingress             = var.allow_unauthenticated ? "INGRESS_TRAFFIC_ALL" : "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"

  template {
    service_account = google_service_account.runtime.email

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    # Postgres over the Cloud SQL connector, authenticated by the runtime service
    # account rather than a network path.
    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [google_sql_database_instance.main.connection_name]
      }
    }

    containers {
      image = var.image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        cpu_idle = true
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }

      env {
        name  = "AGENT_ENVIRONMENT"
        value = "production"
      }

      env {
        name  = "AGENT_USE_IN_MEMORY_RETRIEVER"
        value = "false"
      }

      env {
        name  = "AGENT_USE_POSTGRES_CHECKPOINTER"
        value = "true"
      }

      # Secrets are referenced, never set as plain env vars: a value passed via
      # --set-env-vars is readable through `gcloud run services describe`, in revision
      # metadata, and in Cloud Audit Logs.
      dynamic "env" {
        for_each = {
          AGENT_GROQ_API_KEY   = "groq-api-key"
          AGENT_GEMINI_API_KEY = "gemini-api-key"
          AGENT_POSTGRES_DSN   = "postgres-dsn"
        }
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.app[env.value].secret_id
              version = "latest"
            }
          }
        }
      }

      startup_probe {
        http_get {
          path = "/healthz"
        }
        period_seconds    = 3
        failure_threshold = 30
      }

      liveness_probe {
        http_get {
          path = "/healthz"
        }
        period_seconds = 15
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_secret_manager_secret_iam_member.runtime_access,
    google_project_iam_member.runtime_sql_client,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  count = var.allow_unauthenticated ? 1 : 0

  location = google_cloud_run_v2_service.agent.location
  name     = google_cloud_run_v2_service.agent.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
