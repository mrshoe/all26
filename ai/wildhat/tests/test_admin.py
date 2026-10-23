"""Mentor commands: edit-roster, sync, apply and cancel-jobs."""

import json
import re
from pathlib import Path

from harness import ADMIN_BUCKET, MENTOR, MENTOR_ACCESS_TOKEN, WATCHDOG_URL, WildhatTestCase

MAX = {"email": "824767@seq.org", "name": "Max Shoemaker"}
JANE = {"email": "111111@seq.org", "name": "Jane Doe"}


def roster_json(**students) -> str:
    return json.dumps({"students": {h.replace("_", "-"): s for h, s in students.items()}}, indent=2)


class AdminTestCase(WildhatTestCase):
    def setUp(self):
        super().setUp()
        self.gcloud_login(MENTOR)
        # Max is on the roster but was disabled; a former student still has a
        # handle; a mentor uses Firebase without one.
        self.google.add_user(MAX["email"], "max-shoemaker", displayName=MAX["name"], disabled=True)
        self.google.add_user("999999@seq.org", "gone-kid")
        self.google.add_user("mentor@team100.org")
        self.save_roster(roster_json(max_shoemaker=MAX))

    def save_roster(self, text: str) -> None:
        self.google.put_object(ADMIN_BUCKET, "roster.json", text.encode())

    def saved_roster(self) -> dict:
        return json.loads(self.google.object(ADMIN_BUCKET, "roster.json"))

    def edits_left_in(self, run) -> Path:
        """The temp file wildhat says the mentor's edits are in."""
        match = re.search(r"Your edits are in (.+)", run.out + run.err)
        self.assertIsNotNone(match, run)
        path = Path(match.group(1).strip())
        self.addCleanup(path.unlink, missing_ok=True)
        return path


class EditRosterTest(AdminTestCase):
    def test_saves_syncs_and_applies(self):
        self.queue_edits(roster_json(max_shoemaker=MAX, jane_doe=JANE))

        run = self.wildhat("admin", "edit-roster", typed=["y", "y"])

        self.assertSucceeded(run)
        self.assertIn("+ jane-doe (111111@seq.org)", run.out)
        self.assertEqual(self.saved_roster()["students"], {"max-shoemaker": MAX, "jane-doe": JANE})
        # Firebase now matches the roster.
        self.assertEqual(self.google.claims(JANE["email"]), {"handle": "jane-doe"})
        self.assertEqual(self.google.user(JANE["email"])["displayName"], "Jane Doe")
        self.assertFalse(self.google.user(MAX["email"])["disabled"])
        self.assertTrue(self.google.user("999999@seq.org")["disabled"])
        self.assertEqual(self.google.claims("999999@seq.org"), {})
        self.assertEqual(self.google.claims("mentor@team100.org"), {})
        self.assertNotIn("disabled", self.google.user("mentor@team100.org"))
        # Then Terraform, with the mentor's credentials, in ai/terraform.
        self.assertEqual([c[0] for c in self.commands("terraform")], ["init", "plan", "apply"])
        for call in self.calls("terraform"):
            self.assertEqual(call["oauth_token"], MENTOR_ACCESS_TOKEN)
            self.assertIn(f"-chdir={self.terraform_dir}", call["args"])
        plan, apply = self.calls("terraform")[1:]
        self.assertIn(f"-out={apply['args'][-1]}", plan["args"])

    def test_the_editor_opens_the_current_roster(self):
        self.queue_edits(None)

        self.wildhat("admin", "edit-roster")

        (editor,) = self.calls("fake-editor")
        # It's a temp file, outside the repo, and deleted afterwards.
        self.assertFalse(Path(editor["args"][-1]).exists())

    def test_no_changes(self):
        self.queue_edits(None)

        run = self.wildhat("admin", "edit-roster")

        self.assertSucceeded(run)
        self.assertIn("No changes.", run.out)
        self.assertEqual(self.google.requests_to("accounts:update"), [])
        self.assertEqual(self.calls("terraform"), [])

    def test_lists_every_kind_of_change(self):
        self.save_roster(roster_json(max_shoemaker=MAX, jane_doe=JANE))
        self.queue_edits(
            roster_json(
                max_shoemaker={"email": MAX["email"], "name": "Max S"},
                sam_lee={"email": "222222@seq.org", "name": "Sam Lee"},
            )
        )

        run = self.wildhat("admin", "edit-roster", typed=["n"])

        self.assertIn("+ sam-lee (222222@seq.org)", run.out)
        self.assertIn("- jane-doe (111111@seq.org)", run.out)
        self.assertIn("~ max-shoemaker: name 'Max Shoemaker' -> 'Max S'", run.out)
        self.edits_left_in(run)

    def test_declining_to_save_keeps_the_edits(self):
        new = roster_json(max_shoemaker=MAX, jane_doe=JANE)
        self.queue_edits(new)

        run = self.wildhat("admin", "edit-roster", typed=["n"])

        self.assertEqual(run.code, 1)
        self.assertEqual(json.loads(self.edits_left_in(run).read_text(encoding="utf-8")), json.loads(new))
        self.assertEqual(self.saved_roster()["students"], {"max-shoemaker": MAX})
        self.assertEqual(self.google.requests_to("accounts:update"), [])

    def test_declining_to_apply_still_saves_and_syncs(self):
        self.queue_edits(roster_json(max_shoemaker=MAX, jane_doe=JANE))

        run = self.wildhat("admin", "edit-roster", typed=["y", "n"])

        self.assertEqual(run.code, 1)
        self.assertIn("wildhat admin apply", run.out)
        self.assertIn("jane-doe", self.saved_roster()["students"])
        self.assertEqual(self.google.claims(JANE["email"]), {"handle": "jane-doe"})
        self.assertEqual([c[0] for c in self.commands("terraform")], ["init", "plan"])

    def test_mistakes_can_be_fixed_in_the_editor(self):
        self.queue_edits("{oops", roster_json(max_shoemaker=MAX, jane_doe=JANE))

        run = self.wildhat("admin", "edit-roster", typed=["y", "y", "y"])

        self.assertSucceeded(run)
        self.assertIn("Not valid JSON", run.out)
        self.assertEqual(len(self.calls("fake-editor")), 2)
        self.assertIn("jane-doe", self.saved_roster()["students"])

    def test_rejects_bad_rosters(self):
        cases = {
            "Not valid JSON": "{oops",
            'must be an object with a "students" object': "[]",
            "handles are first-last": roster_json(Max=MAX),
            "handles are first-last ": roster_json(jane2_doe=JANE),
            "email must be a lowercase address": roster_json(jane_doe={"email": "Jane@seq.org", "name": "J"}),
            'needs exactly "email" and "name"': roster_json(jane_doe={"email": JANE["email"]}),
            "name is missing": roster_json(jane_doe={"email": JANE["email"], "name": " "}),
            "same email as": roster_json(max_shoemaker=MAX, jane_doe=MAX),
            "Listed more than once: jane-doe": (
                '{"students": {"jane-doe": %s, "jane-doe": %s}}' % (json.dumps(JANE), json.dumps(JANE))
            ),
        }
        for message, text in cases.items():
            with self.subTest(message):
                self.queue_edits(text)

                run = self.wildhat("admin", "edit-roster", typed=["n"])

                self.assertEqual(run.code, 1)
                self.assertIn(message.strip(), run.out)
                self.edits_left_in(run)
        self.assertEqual(self.saved_roster()["students"], {"max-shoemaker": MAX})

    def test_someone_else_saving_first_is_caught(self):
        self.queue_edits(roster_json(max_shoemaker=MAX, jane_doe=JANE))

        def another_mentor_saves_then_yes():
            self.save_roster(roster_json(max_shoemaker=MAX, sam_lee={"email": "2@seq.org", "name": "Sam"}))
            return "y"

        run = self.wildhat("admin", "edit-roster", typed=[another_mentor_saves_then_yes])

        self.assertFailedWith(run, "Someone else changed the roster")
        self.assertIn("jane-doe", self.edits_left_in(run).read_text(encoding="utf-8"))
        self.assertEqual(set(self.saved_roster()["students"]), {"max-shoemaker", "sam-lee"})
        self.assertEqual(self.google.requests_to("accounts:update"), [])

    def test_starts_an_empty_roster_if_there_is_none(self):
        del self.google.objects[(ADMIN_BUCKET, "roster.json")]
        self.queue_edits(roster_json(jane_doe=JANE))

        run = self.wildhat("admin", "edit-roster", typed=["y", "y"])

        self.assertSucceeded(run)
        self.assertEqual(self.saved_roster()["students"], {"jane-doe": JANE})


class SyncTest(AdminTestCase):
    def test_makes_firebase_match_the_roster(self):
        self.save_roster(roster_json(max_shoemaker=MAX, jane_doe=JANE))

        run = self.wildhat("admin", "sync")

        self.assertSucceeded(run)
        self.assertEqual(self.google.claims(JANE["email"]), {"handle": "jane-doe"})
        self.assertFalse(self.google.user(MAX["email"])["disabled"])
        self.assertTrue(self.google.user("999999@seq.org")["disabled"])
        self.assertEqual(self.calls("terraform"), [])

    def test_a_second_sync_changes_nothing(self):
        self.save_roster(roster_json(max_shoemaker=MAX, jane_doe=JANE))
        self.wildhat("admin", "sync")
        updates = len(self.google.requests_to("accounts:update"))

        run = self.wildhat("admin", "sync")

        self.assertIn("2 student(s) already up to date", run.out)
        self.assertEqual(len(self.google.requests_to("accounts:update")), updates)

    def test_keeps_other_custom_claims(self):
        self.google.user("999999@seq.org")["customAttributes"] = json.dumps({"handle": "gone-kid", "admin": True})

        self.wildhat("admin", "sync")

        self.assertEqual(self.google.claims("999999@seq.org"), {"admin": True})

    def test_dry_run_changes_nothing(self):
        self.save_roster(roster_json(max_shoemaker=MAX, jane_doe=JANE))

        run = self.wildhat("admin", "sync", "--dry-run")

        self.assertSucceeded(run)
        self.assertIn("would create jane-doe", run.out)
        self.assertIn("would disable gone-kid", run.out)
        self.assertEqual([r for r in self.google.requests if r["method"] == "POST"], [])

    def test_needs_a_roster(self):
        del self.google.objects[(ADMIN_BUCKET, "roster.json")]

        self.assertFailedWith(self.wildhat("admin", "sync"), "There's no roster")


class MentorCredentialsTest(AdminTestCase):
    def test_uses_the_mentors_account_by_name(self):
        self.wildhat("admin", "sync")

        self.assertIn(["auth", "print-access-token"], self.commands("gcloud"))
        (call,) = [c for c in self.calls("gcloud") if "print-access-token" in c["args"]]
        self.assertIn(f"--account={MENTOR}", call["args"])

    def test_expired_gcloud_credentials(self):
        self.respond("gcloud", ["auth", "print-access-token"], stderr="ERROR: Reauthentication failed.", returncode=1)

        run = self.wildhat("admin", "sync")

        self.assertFailedWith(run, f"Couldn't get gcloud credentials for {MENTOR}. Run: gcloud auth login")
        self.assertIn("Reauthentication failed", run.err)

    def test_no_google_account_in_gcloud(self):
        (self.fake_dir / "gcloud.json").unlink()

        run = self.wildhat("admin", "sync")

        self.assertFailedWith(run, "none is signed in to gcloud. Run: gcloud auth login")

    def test_a_google_account_without_access_is_refused_with_a_hint(self):
        self.gcloud_login("someone@team100.org")

        run = self.wildhat("admin", "sync")

        self.assertFailedWith(run, "HTTP 403")
        self.assertIn("gcloud auth list", run.err)

    def test_the_active_google_account_wins(self):
        self.gcloud_login("someone@team100.org", MENTOR, active=MENTOR)

        self.assertSucceeded(self.wildhat("admin", "sync"))

    def test_several_google_accounts_and_none_active_must_be_picked(self):
        self.gcloud_login("someone@team100.org", MENTOR)
        state = self.gcloud_state()
        state["active"] = None
        (self.fake_dir / "gcloud.json").write_text(json.dumps(state), encoding="utf-8")

        run = self.wildhat("admin", "sync")

        self.assertFailedWith(run, "Several Google accounts are signed in to gcloud")
        self.assertIn("gcloud config set account", run.err)


class ApplyTest(AdminTestCase):
    def test_nothing_to_apply(self):
        self.respond("terraform", ["plan"], returncode=0)

        run = self.wildhat("admin", "apply")

        self.assertSucceeded(run)
        self.assertIn("already match", run.out)
        self.assertEqual([c[0] for c in self.commands("terraform")], ["init", "plan"])

    def test_plan_failure(self):
        self.respond("terraform", ["plan"], returncode=1)

        self.assertFailedWith(self.wildhat("admin", "apply"), "terraform plan failed")

    def test_init_failure_shows_terraforms_output(self):
        self.respond("terraform", ["init"], stderr="Error: backend unreachable", returncode=1)

        run = self.wildhat("admin", "apply")

        self.assertFailedWith(run, "terraform init failed")
        self.assertIn("backend unreachable", run.err)

    def test_without_terraform(self):
        self.remove_launcher("terraform")

        self.assertFailedWith(self.wildhat("admin", "apply"), "Couldn't find terraform")


class CancelJobsTest(AdminTestCase):
    def watchdog_calls(self) -> list[dict]:
        host = WATCHDOG_URL.split("//")[1]
        return [r for r in self.google.requests if r["host"] == host]

    def test_cancels_everything_after_asking(self):
        self.google.watchdog_reply = {
            "month_cost": 42.0,
            "checked": 3,
            "cancelled": ["projects/p/locations/us-central1/jobs/train/executions/train-abc12"],
        }

        run = self.wildhat("admin", "cancel-jobs", typed=["y"])

        self.assertSucceeded(run)
        self.assertIn("Cancel EVERY running job execution?", run.out)
        self.assertIn("$42.00", run.out)
        self.assertIn("Checked 3 execution(s)", run.out)
        self.assertIn("cancelled train/executions/train-abc12", run.out)
        (call,) = self.watchdog_calls()
        self.assertEqual(float(call["query"]["max_minutes"]), 0)

    def test_only_long_runs(self):
        run = self.wildhat("admin", "cancel-jobs", "--running-longer-than", "90", "--yes")

        self.assertSucceeded(run)
        self.assertIn("Nothing to cancel.", run.out)
        (call,) = self.watchdog_calls()
        self.assertEqual(float(call["query"]["max_minutes"]), 90)

    def test_saying_no_cancels_nothing(self):
        run = self.wildhat("admin", "cancel-jobs", typed=["n"])

        self.assertEqual(run.code, 1)
        self.assertEqual(self.google.requests, [])

    def test_refused_by_the_watchdog(self):
        self.respond("gcloud", ["auth", "print-identity-token"], stdout="not-a-mentor\n")

        self.assertFailedWith(self.wildhat("admin", "cancel-jobs", "--yes"), "HTTP 403")
