# Cost guardrails from ../INFRA.md, apart from the budget (billing.tf). Still
# to come with `wildhat launch`: a timeout and launched-by label on every job
# definition.
#
# GPU quota: today it's 0 L4s, and Google won't take an increase request
# ("not enough usage history"). Cloud Run grants 3 L4s with the first GPU job;
# once Google allows it, request max_gpus (and keep RTX Pro 6000 at 0). The
# watchdog enforces its own caps below either way; until the quota reaches
# max_gpus, the quota is the tighter of the two project limits.
#
# INFRA.md planned for the budget to trigger job cancellation through Pub/Sub.
# The organization's domain-restriction policy blocks that: the budget can't be
# granted publish access to a topic in this project. So a scheduled watchdog
# does it instead: it cancels any execution that runs too long, and every
# running execution once the billing export shows the month's cost at or over
# the budget.

locals {
  max_job_runtime_hours = 8
  # GPUs one person's executions may hold at once, and the whole project.
  max_gpus_per_user = 5
  max_gpus          = 10
}

# --- Cancel executions that run too long, overspend or hog GPUs ----------

resource "google_service_account" "job_watchdog" {
  account_id   = "job-watchdog"
  display_name = "Cancels Cloud Run jobs that run too long or overspend"
}

# Just enough to find and cancel executions; nothing else.
resource "google_project_iam_custom_role" "job_canceller" {
  role_id     = "jobCanceller"
  title       = "Cloud Run job canceller"
  permissions = ["run.jobs.list", "run.executions.list", "run.executions.cancel"]
}

resource "google_project_iam_member" "job_watchdog_cancel" {
  project = local.project_id
  role    = google_project_iam_custom_role.job_canceller.id
  member  = google_service_account.job_watchdog.member
}

# To total the month's cost from the billing export. dataset_iam_member only
# adds this account; the billing export's own access is left alone.
resource "google_bigquery_dataset_iam_member" "job_watchdog_billing" {
  dataset_id = google_bigquery_dataset.billing_export.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = google_service_account.job_watchdog.member
}

resource "google_project_iam_member" "job_watchdog_query" {
  project = local.project_id
  role    = "roles/bigquery.jobUser"
  member  = google_service_account.job_watchdog.member
}

# Cloud Scheduler calls the function as the watchdog's own account.
resource "google_cloud_run_service_iam_member" "job_watchdog_invoke" {
  location = local.region
  service  = google_cloudfunctions2_function.job_watchdog.service_config[0].service
  role     = "roles/run.invoker"
  member   = google_service_account.job_watchdog.member
}

resource "google_cloud_scheduler_job" "job_watchdog" {
  name   = "job-watchdog"
  region = local.region
  # Every minute while we test it; back to "*/15 * * * *" afterwards.
  schedule = "* * * * *"

  http_target {
    http_method = "POST"
    uri         = google_cloudfunctions2_function.job_watchdog.service_config[0].uri
    oidc_token {
      service_account_email = google_service_account.job_watchdog.email
      audience              = google_cloudfunctions2_function.job_watchdog.service_config[0].uri
    }
  }

  depends_on = [google_project_service.apis]
}

# Cloud Build packages the function. The project's default compute account has
# no roles here, so builds run as their own account.
resource "google_service_account" "function_builder" {
  account_id   = "function-builder"
  display_name = "Builds Cloud Run functions"
}

resource "google_project_iam_member" "function_builder" {
  for_each = toset([
    "roles/artifactregistry.writer",
    "roles/logging.logWriter",
  ])
  project = local.project_id
  role    = each.value
  member  = google_service_account.function_builder.member
}

resource "google_storage_bucket_iam_member" "function_builder_source" {
  bucket = google_storage_bucket_object.job_watchdog_source.bucket
  role   = "roles/storage.objectViewer"
  member = google_service_account.function_builder.member
  # The admin bucket also holds the roster and Terraform state.
  condition {
    title      = "function-source-only"
    expression = "resource.name.startsWith(\"projects/_/buckets/${google_storage_bucket_object.job_watchdog_source.bucket}/objects/functions/\")"
  }
}

# Cloud Functions copies the source into its own staging bucket before the
# build reads it.
resource "google_storage_bucket_iam_member" "function_builder_staging" {
  bucket = "gcf-v2-sources-${local.project_number}-${local.region}"
  role   = "roles/storage.objectViewer"
  member = google_service_account.function_builder.member
}

data "archive_file" "job_watchdog" {
  type        = "zip"
  source_dir  = "${path.module}/functions/job_watchdog"
  output_path = "${path.module}/.terraform/job_watchdog.zip"
  excludes    = ["__pycache__", ".*.swp"]
}

resource "google_storage_bucket_object" "job_watchdog_source" {
  bucket = "${local.project_id}-admin"
  # A new name for each version of the code, so Terraform redeploys on change.
  name   = "functions/job_watchdog-${data.archive_file.job_watchdog.output_md5}.zip"
  source = data.archive_file.job_watchdog.output_path
}

resource "google_cloudfunctions2_function" "job_watchdog" {
  name     = "job-watchdog"
  location = local.region

  build_config {
    runtime         = "python313"
    entry_point     = "check_jobs"
    service_account = google_service_account.function_builder.id
    source {
      storage_source {
        bucket = google_storage_bucket_object.job_watchdog_source.bucket
        object = google_storage_bucket_object.job_watchdog_source.name
      }
    }
  }

  service_config {
    max_instance_count    = 1
    available_memory      = "256M"
    timeout_seconds       = 120
    service_account_email = google_service_account.job_watchdog.email
    environment_variables = {
      PROJECT           = local.project_id
      REGION            = local.region
      MAX_RUNTIME_HOURS = local.max_job_runtime_hours
      BUDGET_USD        = local.monthly_budget_usd
      MAX_GPUS_PER_USER = local.max_gpus_per_user
      MAX_GPUS          = local.max_gpus
      # The standard export's table, named by Cloud Billing.
      BILLING_TABLE = "${local.project_id}.${google_bigquery_dataset.billing_export.dataset_id}.gcp_billing_export_v1_${replace(local.billing_account, "-", "_")}"
    }
  }

  depends_on = [
    google_project_service.apis,
    google_project_iam_member.function_builder,
    google_storage_bucket_iam_member.function_builder_source,
    google_storage_bucket_iam_member.function_builder_staging,
  ]
}

# --- Container images ------------------------------------------------------

resource "google_artifact_registry_repository" "wildhat" {
  repository_id = "wildhat"
  location      = local.region
  format        = "DOCKER"

  cleanup_policy_dry_run = false
  cleanup_policies {
    id     = "delete-untagged"
    action = "DELETE"
    condition {
      tag_state  = "UNTAGGED"
      older_than = "604800s" # 7 days
    }
  }

  depends_on = [google_project_service.apis]
}

# --- Billing export --------------------------------------------------------

# Cloud Billing writes cost rows here, labels included, so spend can be split
# by who launched what. Turning the export on can only be done in the console:
# Billing > Billing export > BigQuery export, pick this project and dataset.
# Standard and detailed exports are both on (enabled 2026-10-22).
#
# Enabling it gave billing-export-bigquery@system.gserviceaccount.com Owner on
# the dataset. Don't add `access` blocks here: Terraform would replace the
# dataset's whole access list and silently break the export.
resource "google_bigquery_dataset" "billing_export" {
  dataset_id = "billing_export"
  location   = "US"

  depends_on = [google_project_service.apis]
}
