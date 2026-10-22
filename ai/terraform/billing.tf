# Spending guardrail. Alerts email the billing account's admins and the extra
# recipients below; they don't stop anything. The watchdog in guardrails.tf
# does: it cancels running jobs once the month's cost reaches the same amount.

locals {
  billing_account    = "01B1D9-5119C3-F28D8E"
  monthly_budget_usd = 300
  # Extra alert recipients, such as a phone's email-to-text address. They're
  # personal, so they live in the private admin bucket, not in this repo.
  alert_emails = jsondecode(data.google_storage_bucket_object_content.alerts.content).budget_alert_emails
}

data "google_storage_bucket_object_content" "alerts" {
  bucket = "${local.project_id}-admin"
  name   = "alerts.json"
}

resource "google_monitoring_notification_channel" "budget_alert" {
  for_each     = toset(local.alert_emails)
  display_name = "Budget alerts (extra recipient)"
  type         = "email"
  labels = {
    email_address = each.value
  }
  depends_on = [google_project_service.apis]
}

resource "google_billing_budget" "monthly" {
  billing_account = local.billing_account
  display_name    = "wildhat monthly"

  budget_filter {
    projects = ["projects/${local.project_number}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(local.monthly_budget_usd)
    }
  }

  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 0.9
  }
  threshold_rules {
    threshold_percent = 1.0
  }

  # The billing admins still get the threshold emails too.
  all_updates_rule {
    monitoring_notification_channels = [
      for c in google_monitoring_notification_channel.budget_alert : c.id
    ]
  }

  depends_on = [google_project_service.apis["billingbudgets.googleapis.com"]]
}
