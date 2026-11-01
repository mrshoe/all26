# Where runs happen: the base image, each person's Cloud Run jobs, and the
# Firestore database that records runs (../INFRA.md).
#
# Each person has a CPU job, wh-<handle>, and (once the GPU quota allows) a
# GPU job, wh-<handle>-gpu. `wildhat launch` runs them with overrides: the
# command, environment variables, task count and timeout. Everything else is
# fixed here, including the image, the machine size and how many tasks run at
# once. A job runs as its launcher's own service account, so it can do
# exactly what its launcher can, and mentors' jobs are just like students'.

locals {
  # Off until Google raises the L4 quota above 0 (see guardrails.tf).
  gpu_jobs = local.config.gpu_jobs

  # Overridable per run by `wildhat launch`, up to the watchdog's limit.
  job_timeout = "3600s"
  # Tasks of one execution that may run at once. Overrides can't change it.
  # 5 x 8 GB is as much as a new project's Cloud Run memory quota allows
  # ("Parallelism must be set to 5 or fewer to set the requested total
  # memory").
  cpu_parallelism = local.config.cpu_parallelism

  image_repo = "${local.region}-docker.pkg.dev/${local.project_id}/${google_artifact_registry_repository.wildhat.repository_id}"
  # Tagged by a hash of images/base, so changing any file there builds a new
  # image and moves every job to it.
  base_image_dir = "${path.module}/../images/base"
  base_image_tag = substr(sha256(join("\n", [
    for f in sort(fileset(local.base_image_dir, "**")) :
    "${f} ${filesha256("${local.base_image_dir}/${f}")}" if !strcontains(f, "__pycache__")
  ])), 0, 12)
  base_image = "${local.image_repo}/base:${local.base_image_tag}"
}

# --- The base image ---------------------------------------------------------

# Cloud Build runs as this account: the default compute account has no roles.
resource "google_service_account" "image_builder" {
  account_id   = "image-builder"
  display_name = "Builds wildhat's container images"

  depends_on = [google_project_service.apis]
}

resource "google_project_iam_member" "image_builder" {
  for_each = toset([
    "roles/artifactregistry.writer",
    "roles/logging.logWriter",
  ])
  project = local.project_id
  role    = each.value
  member  = google_service_account.image_builder.member
}

# `gcloud builds submit` stages the build's files under builds/ in the admin
# bucket, which also holds the roster and Terraform state.
resource "google_storage_bucket_iam_member" "image_builder_source" {
  bucket = "${local.project_id}-admin"
  role   = "roles/storage.objectViewer"
  member = google_service_account.image_builder.member
  condition {
    title      = "build-source-only"
    expression = "resource.name.startsWith(\"projects/_/buckets/${local.project_id}-admin/objects/builds/\")"
  }
}

resource "terraform_data" "base_image" {
  triggers_replace = local.base_image
  provisioner "local-exec" {
    # wildhat admin apply runs Terraform, so wildhat is on the PATH, and this
    # works the same in PowerShell and on macOS. It skips the build if the
    # tag already exists.
    command     = "wildhat admin build-image base ${local.base_image_tag}"
    working_dir = path.module
  }
  depends_on = [
    google_project_iam_member.image_builder,
    google_storage_bucket_iam_member.image_builder_source,
  ]
}

# --- Run records --------------------------------------------------------------

# The `runs` and `training` collections. Every wh- account can read and write
# them (students.tf): IAM can't limit an account to its own documents, and
# each record says who launched it.
resource "google_firestore_database" "default" {
  name        = "(default)"
  location_id = local.region
  type        = "FIRESTORE_NATIVE"

  delete_protection_state = "DELETE_PROTECTION_ENABLED"
  # Run records outlive Terraform: never delete the database with it.
  deletion_policy = "ABANDON"

  depends_on = [google_project_service.apis]
}

# For `wildhat runs --mine` and `--user`: one person's runs, newest first.
resource "google_firestore_index" "runs_by_person" {
  database   = google_firestore_database.default.name
  collection = "runs"
  fields {
    field_path = "launched_by"
    order      = "ASCENDING"
  }
  fields {
    field_path = "created_at"
    order      = "DESCENDING"
  }
}

# --- Each person's jobs -------------------------------------------------------

resource "google_cloud_run_v2_job" "cpu" {
  for_each = local.students
  name     = "wh-${each.key}"
  location = local.region
  # Removing someone from the roster removes their jobs.
  deletion_protection = false
  labels              = { launched-by = each.key }

  template {
    labels      = { launched-by = each.key }
    task_count  = 1
    parallelism = local.cpu_parallelism
    template {
      service_account = google_service_account.student[each.key].email
      timeout         = local.job_timeout
      max_retries     = 0
      containers {
        image = local.base_image
        resources {
          limits = { cpu = "2", memory = "8Gi" }
        }
      }
    }
  }

  depends_on = [terraform_data.base_image]
}

resource "google_cloud_run_v2_job" "gpu" {
  for_each            = local.gpu_jobs ? local.students : {}
  name                = "wh-${each.key}-gpu"
  location            = local.region
  deletion_protection = false
  labels              = { launched-by = each.key }

  template {
    labels     = { launched-by = each.key }
    task_count = 1
    # One L4 per task, so this is also the per-execution GPU cap.
    parallelism = local.max_gpus_per_user
    template {
      service_account = google_service_account.student[each.key].email
      timeout         = local.job_timeout
      max_retries     = 0
      # Required for GPU jobs; also cheaper.
      gpu_zonal_redundancy_disabled = true
      node_selector {
        accelerator = "nvidia-l4"
      }
      containers {
        image = local.base_image
        resources {
          # Cloud Run's minimum with an L4.
          limits = { cpu = "4", memory = "16Gi", "nvidia.com/gpu" = "1" }
        }
      }
    }
  }

  depends_on = [terraform_data.base_image]
}

# People may run (and cancel) only their own jobs, changing only what
# overrides allow.
resource "google_cloud_run_v2_job_iam_member" "cpu" {
  for_each = google_cloud_run_v2_job.cpu
  location = each.value.location
  name     = each.value.name
  role     = "roles/run.jobsExecutorWithOverrides"
  member   = google_service_account.student[each.key].member
}

resource "google_cloud_run_v2_job_iam_member" "gpu" {
  for_each = google_cloud_run_v2_job.gpu
  location = each.value.location
  name     = each.value.name
  role     = "roles/run.jobsExecutorWithOverrides"
  member   = google_service_account.student[each.key].member
}
