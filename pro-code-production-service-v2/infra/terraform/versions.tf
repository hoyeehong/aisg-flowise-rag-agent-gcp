terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source = "hashicorp/google"
      # Pinned to a minor range: a major provider bump renames and removes resources,
      # so it should be a deliberate change with a plan reviewed, not something a
      # `terraform init` picks up.
      version = "~> 6.12"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # Remote state is required for anything shared. Local state cannot be locked, so two
  # concurrent applies silently corrupt it. Configured via `-backend-config` so the
  # bucket name is not committed.
  backend "gcs" {}
}

provider "google" {
  project = var.project_id
  region  = var.region
}
