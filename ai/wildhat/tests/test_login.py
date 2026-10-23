import json
import os
import shlex
import urllib.parse

from harness import PROJECT, WildhatTestCase, jwt_claims

STUDENT_SA = f"wh-max-shoemaker@{PROJECT}.iam.gserviceaccount.com"


def wrapped(link: str, *wrappers: str) -> str:
    """The link as a mail system might rewrite it, e.g. google.com/url?q=..."""
    for wrapper in wrappers:
        link = wrapper + urllib.parse.quote(link, safe="")
    return link


class LoginTest(WildhatTestCase):
    def test_clicking_the_emailed_link_signs_in_and_sets_up_gcloud(self):
        self.google.add_user("824767@seq.org", "max-shoemaker")

        run = self.wildhat("login", "--email", " 824767@SEQ.org ")

        self.assertSucceeded(run)
        self.assertIn("Signed in as 824767@seq.org (max-shoemaker)", run.out)
        self.assertIn(f"gcloud now works as {STUDENT_SA}", run.out)
        # The link was sent to this computer's one-shot web server.
        (send,) = self.google.requests_to("accounts:sendOobCode")
        continue_url = urllib.parse.urlsplit(json.loads(send["body"])["continueUrl"])
        self.assertEqual(continue_url.hostname, "localhost")
        visits = {urllib.parse.urlsplit(url).path: (status, page) for url, status, page in self.google.browser_visits}
        self.assertEqual(visits["/favicon.ico"][0], 404)
        self.assertEqual(visits["/"][0], 200)
        self.assertIn("You can close this tab", visits["/"][1])

        self.assertEqual(
            [c[:4] for c in self.commands("gcloud")],
            [
                ["auth", "list"],
                ["iam", "workload-identity-pools", "create-cred-config",
                 "projects/735918754010/locations/global/workloadIdentityPools/students/providers/firebase"],
                ["auth", "login"],
                ["config", "set", "project", PROJECT],
                ["config", "set", "run/region", "us-central1"],
                ["run", "jobs", "list"],
            ],
        )
        for call in self.calls("gcloud"):
            self.assertIn("--quiet", call["args"])
            self.assertEqual(call["allow_executables"], "1")
        # On a student's laptop, plain gcloud now works as them.
        self.assertEqual(self.gcloud_state()["active"], STUDENT_SA)
        (check,) = [c for c in self.calls("gcloud") if c["args"][:3] == ["run", "jobs", "list"]]
        self.assertIn(f"--account={STUDENT_SA}", check["args"])

    def test_signing_in_again_keeps_the_service_account_active(self):
        self.log_in()

        self.assertSucceeded(self.wildhat("login", "--email", "824767@seq.org"))

        self.assertEqual(self.gcloud_state()["active"], STUDENT_SA)

    def test_gcloud_can_run_wildhat_id_token_from_the_credential_config(self):
        self.log_in()

        cred = json.loads((self.state_dir / "cred.json").read_text(encoding="utf-8"))
        self.assertIn(STUDENT_SA, cred["service_account_impersonation_url"])
        # google-auth splits the command with shlex; the path has a space in it.
        argv = shlex.split(cred["credential_source"]["executable"]["command"])
        self.assertEqual(argv[1:], ["id-token"])
        self.assertEqual(os.path.normcase(os.path.dirname(argv[0])), os.path.normcase(str(self.bin.resolve())))
        # `gcloud run jobs list` ran it, and it handed over Max's token.
        (source,) = self.calls("credential-source")
        self.assertEqual(source["returncode"], 0, source)
        token = json.loads(source["stdout"])["id_token"]
        self.assertEqual(jwt_claims(token)["handle"], "max-shoemaker")

    def test_saved_credentials_are_private(self):
        if os.name == "nt":
            self.skipTest("%APPDATA% is already private to the user")
        self.log_in()
        self.assertEqual(self.state_dir.stat().st_mode & 0o777, 0o700)
        for path in self.state_dir.iterdir():
            # gcloud writes cred.json; it holds no secrets, just a command.
            if path.name != "cred.json":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600, path)

    def test_pasting_the_link_works_when_the_email_is_opened_elsewhere(self):
        self.google.add_user("824767@seq.org", "max-shoemaker")
        self.google.click_links = False

        for name, wrappers in [
            ("plain", ()),
            ("gmail", ("https://www.google.com/url?sa=D&q=",)),
            ("outlook and gmail", ("https://www.google.com/url?q=", "https://nam12.safelinks.protection.outlook.com/?url=")),
        ]:
            with self.subTest(name):
                run = self.wildhat(
                    "login", "--email", "824767@seq.org",
                    typed=[
                        "",
                        "hello?",
                        lambda: "  <" + wrapped(self.google.sent_links[-1], *wrappers) + ">  ",
                    ],
                )
                self.assertSucceeded(run)
                self.assertIn("Couldn't find a sign-in code in that", run.out)
                self.assertIn("Signed in as 824767@seq.org", run.out)

    def test_students_not_on_the_roster_are_turned_away(self):
        run = self.wildhat("login", "--email", "someone@seq.org")

        self.assertFailedWith(run, "isn't on the team roster yet")
        self.assertEqual(self.calls("gcloud"), [])
        self.assertFalse((self.state_dir / "account.json").exists())

    def test_a_used_or_bad_link_is_explained(self):
        self.google.add_user("824767@seq.org", "max-shoemaker")
        self.google.fail("accounts:signInWithEmailLink", 400, "INVALID_OOB_CODE")

        run = self.wildhat("login", "--email", "824767@seq.org")

        self.assertFailedWith(run, "That sign-in link is invalid or was already used")

    def test_disabled_students_are_told_so(self):
        self.google.add_user("824767@seq.org", "max-shoemaker", disabled=True)

        run = self.wildhat("login", "--email", "824767@seq.org")

        self.assertFailedWith(run, "Your account has been disabled")

    def test_missing_cloud_access_is_reported_with_gclouds_message(self):
        self.google.add_user("824767@seq.org", "max-shoemaker")
        self.respond("gcloud", ["run", "jobs", "list"], stderr="ERROR: PERMISSION_DENIED: nope", returncode=1)

        run = self.wildhat("login", "--email", "824767@seq.org")

        self.assertFailedWith(run, "your cloud access isn't set up yet")
        self.assertIn("PERMISSION_DENIED: nope", run.err)

    def test_a_failing_gcloud_step_stops_login(self):
        self.google.add_user("824767@seq.org", "max-shoemaker")
        self.respond("gcloud", ["auth", "login"], stderr="ERROR: bad cred file", returncode=1)

        run = self.wildhat("login", "--email", "824767@seq.org")

        self.assertFailedWith(run, "gcloud auth login failed")
        self.assertIn("bad cred file", run.err)
        self.assertNotIn(["config", "set", "project", PROJECT], self.commands("gcloud"))

    def test_without_gcloud_no_email_is_sent(self):
        self.remove_launcher("gcloud")

        run = self.wildhat("login", "--email", "824767@seq.org")

        self.assertFailedWith(run, "Couldn't find gcloud")
        self.assertEqual(self.google.requests, [])

    def test_offline(self):
        self.google.offline = True

        run = self.wildhat("login", "--email", "824767@seq.org")

        self.assertFailedWith(run, "Are you online?")

    def test_asks_for_the_email_and_checks_it(self):
        run = self.wildhat("login", typed=["not-an-email"])

        self.assertIn("School email:", run.out)
        self.assertFailedWith(run, "doesn't look like an email address")
        self.assertEqual(self.google.requests, [])
