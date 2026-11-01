# Wildhat AI infrastructure. For how student sign-in works, see
# ../wildhat/README.md.
#
# Mentors normally run this through `wildhat admin edit-roster` or
# `wildhat admin apply`, which pass in their gcloud credentials.

locals {
  # The project's settings, shared with the wildhat CLI. ../SETUP.md says how
  # to fill them in for a new project.
  config         = jsondecode(file("${path.module}/../wildhat/src/wildhat/config.json"))
  project_id     = local.config.project_id
  project_number = local.config.project_number
  region         = local.config.region

  # The roster's only copy is in the private admin bucket. Edit it with
  # `wildhat admin edit-roster`. Keyed by handle: {email, name}.
  students = jsondecode(data.google_storage_bucket_object_content.roster.content).students
}

data "google_storage_bucket_object_content" "roster" {
  bucket = "${local.project_id}-admin"
  name   = "roster.json"
}

resource "google_project_service" "apis" {
  for_each = toset([
    "artifactregistry.googleapis.com",
    "bigquery.googleapis.com",
    "billingbudgets.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudfunctions.googleapis.com",
    "cloudquotas.googleapis.com",
    "cloudscheduler.googleapis.com",
    "firestore.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "identitytoolkit.googleapis.com",
    "monitoring.googleapis.com",
    "run.googleapis.com",
    "storage.googleapis.com",
    "sts.googleapis.com",
  ])
  service = each.value
  # Never switch an API off just because Terraform stopped managing it.
  disable_on_destroy = false
}
