# Root module rather than one module per resource group.
#
# The naming convention in the v2 README anticipated per-group modules. For a single
# service with no second consumer, modules would add a variable-passing layer without
# any reuse to justify it -- so the resources are grouped by file instead, and a module
# boundary can be introduced when a second service actually needs one.

locals {
  # Enabled before anything else; a resource created against a disabled API fails with
  # an error that does not name the API.
  required_apis = [
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "secretmanager.googleapis.com",
    "artifactregistry.googleapis.com",
    "iam.googleapis.com",
    "monitoring.googleapis.com",
    "cloudtrace.googleapis.com",
    "pubsub.googleapis.com",
  ]
}

resource "google_project_service" "required" {
  for_each = toset(local.required_apis)

  project = var.project_id
  service = each.value

  # Leave the APIs enabled on destroy: other workloads in the project may depend on
  # them, and disabling an API is far more disruptive than leaving it on.
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = var.service_name
  description   = "Container images for ${var.service_name}"
  format        = "DOCKER"
  labels        = var.labels

  docker_config {
    # Published tags cannot be moved. This is the registry-side half of digest pinning:
    # without it, a tag can be repointed after an image has been reviewed and scanned.
    immutable_tags = true
  }

  depends_on = [google_project_service.required]
}
