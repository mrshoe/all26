"""A mentor's computer: their own Google account is signed in to gcloud for the
admin commands, and they're also on the roster and run `wildhat login`, so
their jobs look exactly like a student's."""

import json

from harness import ADMIN_BUCKET, MENTOR, MENTOR_ACCESS_TOKEN, PROJECT, WildhatTestCase

MENTOR_SA = f"wh-david-shoemaker@{PROJECT}.iam.gserviceaccount.com"
ROSTER = {"students": {"david-shoemaker": {"email": MENTOR, "name": "David Shoemaker"}}}


class MentorMachineTest(WildhatTestCase):
    def setUp(self):
        super().setUp()
        self.gcloud_login(MENTOR)
        self.google.add_user(MENTOR, "david-shoemaker")
        self.google.put_object(ADMIN_BUCKET, "roster.json", json.dumps(ROSTER).encode())

    def test_login_leaves_the_mentors_account_active(self):
        run = self.wildhat("login", "--email", MENTOR)

        self.assertSucceeded(run)
        self.assertIn(f"wildhat works as {MENTOR_SA}", run.out)
        self.assertIn(f"gcloud's active account is still {MENTOR}", run.out)
        self.assertIn(f"--account={MENTOR_SA}", run.out)
        state = self.gcloud_state()
        self.assertEqual(state["active"], MENTOR)
        self.assertEqual(set(state["accounts"]), {MENTOR, MENTOR_SA})
        (login,) = [c for c in self.calls("gcloud") if c["args"][:2] == ["auth", "login"]]
        self.assertIn("--no-activate", login["args"])
        # The access check still ran as the service account.
        self.assertEqual(len(self.calls("credential-source")), 1)

    def test_whoami_checks_the_service_account_not_the_active_one(self):
        self.assertSucceeded(self.wildhat("login", "--email", MENTOR))

        run = self.wildhat("whoami")

        self.assertSucceeded(run)
        self.assertIn(MENTOR_SA, run.out)
        self.assertEqual(len(self.calls("credential-source")), 2)

    def test_admin_commands_still_use_the_mentors_account(self):
        self.assertSucceeded(self.wildhat("login", "--email", MENTOR))

        self.assertSucceeded(self.wildhat("admin", "sync"))
        self.assertSucceeded(self.wildhat("admin", "apply", typed=["y"]))
        self.assertSucceeded(self.wildhat("admin", "cancel-jobs", "--yes"))

        for call in self.calls("terraform"):
            self.assertEqual(call["oauth_token"], MENTOR_ACCESS_TOKEN)

    def test_admin_commands_find_the_mentors_account_even_if_the_service_account_is_active(self):
        # E.g. they ran `wildhat login` before ever signing in to gcloud
        # themselves, then `gcloud auth login --no-activate`.
        (self.fake_dir / "gcloud.json").unlink()
        self.assertSucceeded(self.wildhat("login", "--email", MENTOR))
        self.gcloud_login(MENTOR, active=MENTOR_SA)

        self.assertSucceeded(self.wildhat("admin", "sync"))
        self.assertSucceeded(self.wildhat("admin", "cancel-jobs", "--yes"))

    def test_logout_keeps_the_mentors_account(self):
        self.assertSucceeded(self.wildhat("login", "--email", MENTOR))

        self.assertSucceeded(self.wildhat("logout"))

        state = self.gcloud_state()
        self.assertEqual(state["active"], MENTOR)
        self.assertEqual(set(state["accounts"]), {MENTOR})
        self.assertSucceeded(self.wildhat("admin", "sync"))
