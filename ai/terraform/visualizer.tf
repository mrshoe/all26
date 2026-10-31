# The visualizer (../visualizer): one Cloud Run service, scaled to zero when
# nobody's looking, serving the browser app and its API (../INFRA.md).
#
# Anyone can reach the URL, as Firebase sign-in needs; the app itself checks
# every API call's Firebase ID token for a roster handle. Cloud Run's own
# invoker check is off rather than granted to allUsers, which the
# organization's policy may forbid.

locals {
  visualizer_dir = "${path.module}/../visualizer"
  # Tagged by a hash of the source, like the base image (jobs.tf), leaving
  # out what's built or installed locally.
  visualizer_tag = substr(sha256(join("\n", [
    for f in sort(fileset(local.visualizer_dir, "**")) :
    "${f} ${filesha256("${local.visualizer_dir}/${f}")}"
    if !anytrue([for skip in ["node_modules/", "dist/", "__pycache__", "tests/", "README"] : strcontains(f, skip)])
  ])), 0, 12)
  visualizer_image = "${local.image_repo}/visualizer:${local.visualizer_tag}"
}

resource "google_service_account" "visualizer" {
  account_id   = "visualizer"
  display_name = "The visualizer: reads runs, never changes them"

  depends_on = [google_project_service.apis]
}

resource "google_project_iam_member" "visualizer_firestore" {
  project = local.project_id
  role    = "roles/datastore.viewer"
  member  = google_service_account.visualizer.member
}

resource "google_storage_bucket_iam_member" "visualizer_read" {
  bucket = google_storage_bucket.wildhat.name
  role   = "roles/storage.objectViewer"
  member = google_service_account.visualizer.member
}

# Signed video links: Cloud Run has no private key, so the service asks IAM
# to sign as itself.
resource "google_service_account_iam_member" "visualizer_signs_as_itself" {
  service_account_id = google_service_account.visualizer.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = google_service_account.visualizer.member
}

resource "terraform_data" "visualizer_image" {
  triggers_replace = local.visualizer_image
  provisioner "local-exec" {
    command     = "wildhat admin build-image visualizer ${local.visualizer_tag}"
    working_dir = path.module
  }
  depends_on = [
    google_project_iam_member.image_builder,
    google_storage_bucket_iam_member.image_builder_source,
  ]
}

resource "google_cloud_run_v2_service" "visualizer" {
  name                 = "visualizer"
  location             = local.region
  deletion_protection  = false
  ingress              = "INGRESS_TRAFFIC_ALL"
  invoker_iam_disabled = true

  template {
    service_account = google_service_account.visualizer.email
    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }
    containers {
      image = local.visualizer_image
      env {
        name  = "PROJECT_ID"
        value = local.project_id
      }
      env {
        name  = "BUCKET"
        value = google_storage_bucket.wildhat.name
      }
      env {
        name  = "FIREBASE_API_KEY"
        value = local.config.firebase_api_key
      }
      resources {
        limits = { cpu = "1", memory = "512Mi" }
        # Billed only while answering requests.
        cpu_idle = true
      }
    }
  }

  depends_on = [
    terraform_data.visualizer_image,
    google_project_iam_member.visualizer_firestore,
    google_storage_bucket_iam_member.visualizer_read,
  ]
}

# Firebase only sends sign-in links that return to an authorized domain.
resource "terraform_data" "visualizer_signin" {
  triggers_replace = google_cloud_run_v2_service.visualizer.uri
  provisioner "local-exec" {
    command     = "wildhat admin authorize-domain ${trimprefix(google_cloud_run_v2_service.visualizer.uri, "https://")}"
    working_dir = path.module
  }
}

output "visualizer_url" {
  value = google_cloud_run_v2_service.visualizer.uri
}
