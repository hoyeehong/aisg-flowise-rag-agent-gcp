variable "project_id" {
  description = "GCP project that owns every resource here."
  type        = string
}

variable "region" {
  description = "Primary region. asia-southeast1 keeps data residency in Singapore."
  type        = string
  default     = "asia-southeast1"
}

variable "service_name" {
  description = "Base name for the service and its resources."
  type        = string
  default     = "digital-economy-agent"

  validation {
    # Cloud Run and service account ids share this; both are stricter than a GCP label.
    condition     = can(regex("^[a-z]([-a-z0-9]{0,28}[a-z0-9])?$", var.service_name))
    error_message = "service_name must be lowercase alphanumeric with hyphens, 1-30 chars."
  }
}

variable "image" {
  description = <<-EOT
    Container image to deploy. Pin by digest (repo@sha256:...) in production: a mutable
    tag means a Cloud Run revision cannot be reproduced from this state.
  EOT
  type        = string
}

variable "database_tier" {
  description = "Cloud SQL machine type. db-f1-micro is for demos, not production."
  type        = string
  default     = "db-custom-1-3840"
}

variable "database_deletion_protection" {
  description = "Guard against destroying the database that holds paused review runs."
  type        = bool
  default     = true
}

variable "min_instances" {
  description = "Cloud Run minimum instances. 0 scales to zero and accepts cold starts."
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Cloud Run maximum instances."
  type        = number
  default     = 5
}

variable "allow_unauthenticated" {
  description = <<-EOT
    Whether the service is publicly reachable. Kept explicit rather than defaulted on:
    this API starts report runs that spend model tokens, so an open endpoint is a
    billing exposure as well as a data one.
  EOT
  type        = bool
  default     = false
}

variable "labels" {
  description = "Labels applied to every resource that supports them."
  type        = map(string)
  default = {
    application = "digital-economy-agent"
    managed-by  = "terraform"
  }
}
