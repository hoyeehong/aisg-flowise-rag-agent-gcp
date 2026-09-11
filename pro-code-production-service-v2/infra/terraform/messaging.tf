# Event-driven ingestion.
#
# The consumer reads document events from a subscription rather than being invoked
# directly, so an upstream system can hand off a document without waiting for it to be
# chunked, redacted and embedded -- work measured in seconds per document.

resource "google_pubsub_topic" "documents" {
  name   = "${var.service_name}-documents"
  labels = var.labels

  # Long enough to survive a consumer outage over a weekend without losing events.
  message_retention_duration = "604800s" # 7 days

  depends_on = [google_project_service.required]
}

# Messages that exhausted their delivery attempts land here instead of being retried
# forever. A poison message -- one that will never parse -- otherwise occupies the
# subscription indefinitely and, on an ordered subscription, blocks everything behind it.
resource "google_pubsub_topic" "documents_dead_letter" {
  name   = "${var.service_name}-documents-dlq"
  labels = var.labels

  # Longer than the live topic: a dead letter is evidence, and someone has to be able
  # to look at it after a weekend before it expires.
  message_retention_duration = "1209600s" # 14 days

  depends_on = [google_project_service.required]
}

resource "google_pubsub_subscription" "documents" {
  name   = "${var.service_name}-documents-sub"
  topic  = google_pubsub_topic.documents.id
  labels = var.labels

  # Ingesting a document costs seconds of embedding time. A short deadline would
  # redeliver a message that is still being processed, doubling the embedding bill for
  # no benefit -- the effects are idempotent, but paying twice is not free.
  ack_deadline_seconds = 120

  message_retention_duration = "604800s"
  retain_acked_messages      = false
  enable_message_ordering    = false

  expiration_policy {
    # Never expire. The default deletes a subscription after 31 days of inactivity,
    # which for a low-volume ingestion topic means it silently disappears.
    ttl = ""
  }

  # This block is what makes the consumer's retry ceiling work at all. Pub/Sub
  # populates `delivery_attempt` only on subscriptions that have a dead-letter policy;
  # without one the field is absent, every delivery looks like the first, and
  # max_delivery_attempts in the consumer never fires. PubSubSubscriber raises rather
  # than retrying without a ceiling, so a missing policy here is a startup failure, not
  # a silent one.
  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.documents_dead_letter.id
    max_delivery_attempts = var.consumer_max_delivery_attempts
  }

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }
}

# --- IAM -------------------------------------------------------------------

# The runtime identity reads the subscription. Bound on the subscription itself rather
# than at project level: a project-level subscriber role would grant access to every
# subscription in the project, including ones belonging to other workloads.
resource "google_pubsub_subscription_iam_member" "runtime_subscriber" {
  subscription = google_pubsub_subscription.documents.name
  role         = "roles/pubsub.subscriber"
  member       = google_service_account.runtime.member
}

# Publishing rights are deliberately *not* granted to the runtime identity. The
# consumer reads events; whatever produces them is a separate system with its own
# identity, and granting publish here would let a compromised consumer manufacture
# ingestion events for any tenant it names.

# Dead-lettering is performed by Pub/Sub itself, not by the subscriber, so the Pub/Sub
# service agent needs its own permissions. Omitting these is the most common reason a
# dead-letter policy silently does nothing: messages keep being redelivered past
# max_delivery_attempts and nothing appears in the dead-letter topic.
data "google_project" "current" {
  project_id = var.project_id
}

locals {
  pubsub_agent = "serviceAccount:service-${data.google_project.current.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

resource "google_pubsub_topic_iam_member" "dead_letter_publisher" {
  topic  = google_pubsub_topic.documents_dead_letter.name
  role   = "roles/pubsub.publisher"
  member = local.pubsub_agent
}

resource "google_pubsub_subscription_iam_member" "dead_letter_subscriber" {
  subscription = google_pubsub_subscription.documents.name
  role         = "roles/pubsub.subscriber"
  member       = local.pubsub_agent
}
