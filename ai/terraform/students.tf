# Student sign-in: Firebase ID tokens -> workload identity federation -> one
# service account per student, stu-<handle>.

resource "google_iam_workload_identity_pool" "students" {
  workload_identity_pool_id = "students"
}

resource "google_iam_workload_identity_pool_provider" "firebase" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.students.workload_identity_pool_id
  workload_identity_pool_provider_id = "firebase"
  attribute_mapping = {
    "google.subject"  = "assertion.sub"
    "attribute.email" = "assertion.email"
  }
  attribute_condition = "assertion.email_verified == true"
  oidc {
    issuer_uri        = "https://securetoken.google.com/${local.project_id}"
    allowed_audiences = [local.project_id]
  }
}

resource "google_service_account" "student" {
  for_each     = local.students
  account_id   = "stu-${each.key}"
  display_name = each.value.name
}

# Only a token for the roster's email can act as the student's account. This
# is the real access check; the handle claim just picks the account's name.
resource "google_service_account_iam_member" "student_signin" {
  for_each           = local.students
  service_account_id = google_service_account.student[each.key].name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.students.name}/attribute.email/${each.value.email}"
}

locals {
  student_project_roles = [
    "roles/logging.viewer",
    "roles/run.viewer",
    "roles/serviceusage.serviceUsageConsumer",
  ]
  student_project_grants = {
    for pair in setproduct(keys(local.students), local.student_project_roles) :
    "${pair[0]} ${pair[1]}" => { handle = pair[0], role = pair[1] }
  }
}

resource "google_project_iam_member" "student" {
  for_each = local.student_project_grants
  project  = local.project_id
  role     = each.value.role
  member   = google_service_account.student[each.value.handle].member
}

# Run source, data and results. Students can read everything in it and write
# only under src/<handle>/.
resource "google_storage_bucket" "wildhat" {
  name                        = "${local.project_id}-wildhat"
  location                    = local.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
}

resource "google_storage_bucket_iam_member" "student_read" {
  for_each = local.students
  bucket   = google_storage_bucket.wildhat.name
  role     = "roles/storage.objectViewer"
  member   = google_service_account.student[each.key].member
}

resource "google_storage_bucket_iam_member" "student_upload" {
  for_each = local.students
  bucket   = google_storage_bucket.wildhat.name
  role     = "roles/storage.objectCreator"
  member   = google_service_account.student[each.key].member
  condition {
    title      = "own-src-prefix"
    expression = "resource.name.startsWith(\"projects/_/buckets/${google_storage_bucket.wildhat.name}/objects/src/${each.key}/\")"
  }
}
