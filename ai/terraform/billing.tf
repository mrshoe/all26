# Spending guardrail. Alerts email the billing account's admins; they don't
# stop anything. (The org policy blocks budgets from triggering code, so the
# watchdog in guardrails.tf caps how long any job runs instead.)

locals {
  billing_account = "01B1D9-5119C3-F28D8E"
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
      units         = "300"
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

  depends_on = [google_project_service.apis["billingbudgets.googleapis.com"]]
}
