# Setting up a Google Cloud project for wildhat

How to set up wildhat's Google Cloud project from scratch, for example for
another team using a fork of this repo. You do this once. It takes about an
hour, most of it waiting.

Everything project-specific lives in one file,
[wildhat/src/wildhat/config.json](wildhat/src/wildhat/config.json). The
wildhat CLI and Terraform both read it. Once the project exists, its roster,
service accounts, jobs and guardrails are all managed by Terraform
([terraform/](terraform/)) through `wildhat admin`; this guide covers what
has to exist before Terraform can run.

## What you need

- A Google account that can create projects and link them to a billing
  account (Billing Account User or Administrator on it). This account becomes
  the project's Owner.
- macOS or Linux with the
  [gcloud CLI](https://cloud.google.com/sdk/docs/install),
  [Terraform](https://developer.hashicorp.com/terraform/install) and
  [uv](https://docs.astral.sh/uv/getting-started/installation/), signed in
  with `gcloud auth login`.
- A checkout of this repo.

The commands below are for bash or zsh. Set these first, with your own
values; the project ID must be globally unique:

```sh
PROJECT=my-team-wildhat
REGION=us-central1     # Cheapest region with L4 GPUs. Change it only with a reason.
BILLING=XXXXXX-XXXXXX-XXXXXX   # From `gcloud billing accounts list`
```

## 1. Create the project

```sh
gcloud projects create $PROJECT
gcloud billing projects link $PROJECT --billing-account=$BILLING
```

## 2. Turn on Firebase sign-in

Students and mentors sign in to Firebase Authentication with an email link.
Add Firebase to the project:

```sh
gcloud services enable firebase.googleapis.com identitytoolkit.googleapis.com \
  apikeys.googleapis.com --project=$PROJECT
curl -s -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "x-goog-user-project: $PROJECT" -H "Content-Type: application/json" \
  -d '{}' "https://firebase.googleapis.com/v1beta1/projects/$PROJECT:addFirebase"
```

Then switch on email-link sign-in. There's no API for this step:

1. Open `https://console.firebase.google.com/project/$PROJECT/authentication`
   (with your project ID). If it says the project doesn't exist, use the
   account picker at the top right to switch to the account that created it.
2. Click **Get started**. (If you land on the project's home page instead,
   open **Build → Authentication** in the sidebar first.)
3. On the **Sign-in method** tab, choose **Email/Password**, turn on both
   **Email/Password** and **Email link (passwordless sign-in)**, and click
   **Save**.

## 3. Create the admin bucket

A private bucket holds the roster (everyone's handle, email and name),
Terraform's state, and any extra budget-alert recipients. None of these may go
in the repo, which is public.

```sh
gcloud storage buckets create gs://$PROJECT-admin --project=$PROJECT \
  --location=$REGION --uniform-bucket-level-access --public-access-prevention
gcloud storage buckets update gs://$PROJECT-admin --versioning
echo '{"students": {}}' | gcloud storage cp - gs://$PROJECT-admin/roster.json
echo '{"budget_alert_emails": []}' | gcloud storage cp - gs://$PROJECT-admin/alerts.json
```

Budget alerts go to the billing account's admins. To also send them somewhere
else, such as a phone's email-to-text address, list the addresses in
`alerts.json`.

## 4. Fill in config.json

Print the two values you need:

```sh
gcloud projects describe $PROJECT --format='value(projectNumber)'
gcloud services api-keys get-key-string --project=$PROJECT --format='value(keyString)' \
  $(gcloud services api-keys list --project=$PROJECT --format='value(name)' \
    --filter='displayName="Browser key (auto created by Firebase)"')
```

Then edit `wildhat/src/wildhat/config.json`:

| Setting | What it is |
|---|---|
| `project_id`, `project_number`, `region` | The project, and the region everything runs in. |
| `firebase_api_key` | The key printed above. It only identifies the project to Firebase; it isn't secret. |
| `billing_account` | `$BILLING`. |
| `monthly_budget_usd` | Alerts at 50%, 90% and 100%; at 100% the watchdog cancels every running job. |
| `max_job_runtime_hours` | The watchdog cancels any job running longer, and `wildhat launch --timeout` can't ask for more. |
| `max_gpus_per_user`, `max_gpus` | GPUs one person, and everyone together, may use at once. |
| `cpu_parallelism` | Episodes of one CPU run that run at once. A new project's Cloud Run memory quota allows 5. |
| `gpu_jobs` | Leave `false` until you have GPU quota (step 8). |

Install the CLI from your checkout, or reinstall it if you had it already:

```sh
uv tool install --reinstall ./ai/wildhat
```

## 5. Create everything else

From your checkout:

```sh
wildhat admin apply
```

It shows Terraform's plan (about 40 resources) and asks before applying. The
apply takes about 20 minutes, mostly building the base image
([images/base](images/base/)).

On a brand-new project, the first apply usually stops with an error from the
image build (`(gcloud.builds.submit) PERMISSION_DENIED`): Terraform has only
just switched Cloud Build on, and Google takes a few minutes to finish setting
it up. Other new APIs can fail the same way. Wait five minutes and run
`wildhat admin apply` again; it picks up where it stopped.

## 6. Turn on the billing export

The watchdog reads this month's cost from the billing export to enforce the
budget. Turning the export on is only possible in the console:

1. Open [Billing export](https://console.cloud.google.com/billing/export) and
   pick your billing account.
2. Under **Standard usage cost**, click **Edit settings**, choose your project
   and the `billing_export` dataset, and save.

Rows start appearing within a day. Until then, the watchdog logs that it
can't read the cost, but its runtime limit and GPU caps still work.

## 7. Add yourself and try it

Mentors are on the roster like everyone else, so their jobs behave exactly
like students':

```sh
wildhat admin edit-roster
```

Add yourself, save, and apply when asked. New access takes Google up to about
10 minutes to switch on, so wait that long before signing in. Then sign in and
run something:

```sh
wildhat login
cd ai/src
wildhat launch -- python -c "print('hello from the cloud')"
wildhat logs <run-id> -f
```

The first run on a new image takes a few extra minutes while Cloud Run loads
the image. After that, add students with `wildhat admin edit-roster`, and
point them at [wildhat/README.md](wildhat/README.md).

Commit your `config.json`. Everyone who installs wildhat from your fork then
uses your project.

## 8. GPUs, when you need them

A new project has no L4 GPU quota, and Google may refuse an increase until the
project has some billing history. When you have it:

1. Request `max_gpus` L4s for Cloud Run in your region:
   [Quotas](https://console.cloud.google.com/iam-admin/quotas), the Cloud Run
   Admin API's "Total Nvidia L4 GPU allocation without zonal redundancy, per
   project per region". (wildhat's GPU jobs turn zonal redundancy off, which
   is cheaper.)
2. Set `gpu_jobs` to `true` in `config.json` and run `wildhat admin apply`.
