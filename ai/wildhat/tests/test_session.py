"""After login: `wildhat id-token` (as gcloud runs it), whoami and logout."""

import json
import time
from pathlib import Path
from unittest import mock

from harness import PROJECT, WildhatTestCase, jwt_claims

STUDENT_SA = f"wh-max-shoemaker@{PROJECT}.iam.gserviceaccount.com"
ID_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:id_token"


class IdTokenTest(WildhatTestCase):
    def id_token(self, *, hours_later: float = 0) -> tuple[int, dict]:
        """Run `wildhat id-token`; its output must be exactly one JSON object."""
        later = time.time() + hours_later * 3600
        with mock.patch("time.time", return_value=later):
            run = self.wildhat("id-token")
        self.assertEqual(run.err, "")
        return run.code, json.loads(run.out)

    def test_prints_the_executable_response_gcloud_expects(self):
        self.log_in()

        code, response = self.id_token()

        self.assertEqual(code, 0)
        self.assertEqual(
            set(response), {"version", "success", "token_type", "id_token", "expiration_time"}
        )
        self.assertEqual(response["version"], 1)
        self.assertIs(response["success"], True)
        self.assertEqual(response["token_type"], ID_TOKEN_TYPE)
        self.assertEqual(jwt_claims(response["id_token"])["handle"], "max-shoemaker")
        self.assertAlmostEqual(response["expiration_time"], time.time() + 3600, delta=60)

    def test_reuses_a_fresh_token_without_going_online(self):
        self.log_in()
        before = len(self.google.requests)

        _, first = self.id_token()
        _, second = self.id_token(hours_later=0.5)

        self.assertEqual(first["id_token"], second["id_token"])
        self.assertEqual(len(self.google.requests), before)

    def test_gets_a_new_token_when_the_old_one_is_nearly_expired(self):
        self.log_in()
        _, first = self.id_token()

        code, second = self.id_token(hours_later=0.95)

        self.assertEqual(code, 0)
        self.assertNotEqual(first["id_token"], second["id_token"])
        self.assertEqual(len(self.google.requests_to("/v1/token")), 1)
        # The new token is cached too.
        _, third = self.id_token(hours_later=0.95)
        self.assertEqual(third["id_token"], second["id_token"])
        self.assertEqual(len(self.google.requests_to("/v1/token")), 1)

    def test_a_disabled_student_gets_a_failure_response(self):
        self.log_in()
        self.google.user("824767@seq.org")["disabled"] = True

        code, response = self.id_token(hours_later=2)

        self.assertEqual(code, 1)
        self.assertIs(response["success"], False)
        self.assertEqual(response["code"], "refresh_failed")
        self.assertIn("disabled", response["message"])

    def test_not_logged_in(self):
        code, response = self.id_token()

        self.assertEqual(code, 1)
        self.assertEqual(
            response,
            {"version": 1, "success": False, "code": "not_logged_in", "message": "run wildhat login"},
        )


class WhoamiTest(WildhatTestCase):
    def test_shows_the_account_and_checks_access(self):
        self.log_in()
        sources_before = len(self.calls("credential-source"))

        run = self.wildhat("whoami")

        self.assertSucceeded(run)
        self.assertIn("824767@seq.org", run.out)
        self.assertIn("max-shoemaker", run.out)
        self.assertIn(STUDENT_SA, run.out)
        self.assertRegex(run.out, r"access:\s+working")
        self.assertIn(["auth", "print-access-token"], self.commands("gcloud"))
        # Checking access ran the whole chain, down to `wildhat id-token`.
        self.assertEqual(len(self.calls("credential-source")), sources_before + 1)

    def test_reports_broken_access(self):
        self.log_in()
        self.respond("gcloud", ["auth", "print-access-token"], stderr="ERROR: denied", returncode=1)

        run = self.wildhat("whoami")

        self.assertEqual(run.code, 1)
        self.assertRegex(run.out, r"access:\s+NOT working")
        self.assertIn("ERROR: denied", run.out)

    def test_not_logged_in(self):
        self.assertFailedWith(self.wildhat("whoami"), "You're not signed in")


class LogoutTest(WildhatTestCase):
    def test_revokes_gcloud_credentials_and_deletes_saved_ones(self):
        self.log_in()

        run = self.wildhat("logout")

        self.assertSucceeded(run)
        self.assertIn(["auth", "revoke", STUDENT_SA], self.commands("gcloud"))
        self.assertFalse(self.state_dir.exists())
        self.assertFailedWith(self.wildhat("whoami"), "You're not signed in")
        self.assertEqual(json.loads(self.wildhat("id-token").out)["code"], "not_logged_in")

    def test_works_when_not_logged_in(self):
        run = self.wildhat("logout")

        self.assertSucceeded(run)
        self.assertEqual(self.calls("gcloud"), [])


class CommandLineTest(WildhatTestCase):
    def test_version(self):
        from wildhat import __version__

        run = self.wildhat("--version")

        self.assertSucceeded(run)
        self.assertEqual(run.out.strip(), __version__)

    def test_the_source_tree_is_what_is_tested(self):
        import wildhat
        from harness import SRC

        self.assertEqual(Path(wildhat.__file__).resolve().parent, SRC / "wildhat")
