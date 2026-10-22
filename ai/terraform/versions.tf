terraform {
  required_version = ">= 1.7"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 8.0"
    }
  }

  # State holds student emails, so it lives in the private admin bucket next
  # to the roster, never in this public repo. The bucket itself was created by
  # hand; it can't hold its own state.
  backend "gcs" {
    bucket = "vernal-mantra-512116-d0-admin"
    prefix = "terraform"
  }
}

provider "google" {
  project = local.project_id
  region  = local.region
  # Mentors run Terraform with their own user credentials, which need a
  # project to bill API quota to.
  user_project_override = true
  billing_project       = local.project_id
}
