"""Login sets the variables gcloud and the Python libraries need in new
terminals: by itself on Windows, and in the shell's startup file, if the
person says yes, on macOS and Linux. Logout takes them out again."""

import os
import subprocess
import sys
import unittest
import uuid

from harness import WildhatTestCase, cli

ALLOW = "GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES"
CREDS = "GOOGLE_APPLICATION_CREDENTIALS"
QUESTION = "Add them to ~/.zshrc?"


@unittest.skipUnless(os.name == "nt", "Windows only")
class WindowsTest(WildhatTestCase):
    def test_login_sets_both_variables_without_asking(self):
        run = self.log_in()

        self.assertEqual(self.user_env, {ALLOW: "1", CREDS: str(self.state_dir / "cred.json")})
        self.assertIn("Open a new terminal to use gcloud directly", run.out)
        self.assertNotIn("[Y/n]", run.out)

    def test_someone_elses_credentials_are_kept(self):
        self.user_env[CREDS] = r"C:\keys\mine.json"

        run = self.log_in()

        self.assertEqual(self.user_env, {ALLOW: "1", CREDS: r"C:\keys\mine.json"})
        self.assertIn(r"Leaving GOOGLE_APPLICATION_CREDENTIALS set to C:\keys\mine.json", run.out)

    def test_logout_removes_wildhats_credentials(self):
        self.log_in()

        self.assertSucceeded(self.wildhat("logout"))

        self.assertEqual(self.user_env, {ALLOW: "1"})

    def test_logout_keeps_someone_elses_credentials(self):
        self.log_in()
        self.user_env[CREDS] = r"C:\keys\mine.json"

        self.assertSucceeded(self.wildhat("logout"))

        self.assertEqual(self.user_env[CREDS], r"C:\keys\mine.json")


@unittest.skipUnless(os.name == "nt", "Windows only")
class WindowsRegistryTest(unittest.TestCase):
    """The real registry functions, on a throwaway variable."""

    def test_set_read_and_delete(self):
        name = f"WILDHAT_TEST_{uuid.uuid4().hex[:8].upper()}"
        self.addCleanup(cli.write_user_env, name, None)

        cli.write_user_env(name, r"C:\Users\Jane Doe\x.json")
        self.assertEqual(cli.read_user_env(name), r"C:\Users\Jane Doe\x.json")
        # As setx would see it: a new process reads it from the registry.
        reg = subprocess.run(["reg", "query", r"HKCU\Environment", "/v", name], capture_output=True, text=True)
        self.assertIn(r"C:\Users\Jane Doe\x.json", reg.stdout)

        cli.write_user_env(name, None)
        self.assertIsNone(cli.read_user_env(name))
        cli.write_user_env(name, None)  # Deleting again is fine.


@unittest.skipIf(os.name == "nt", "macOS and Linux only")
class ShellStartupFileTest(WildhatTestCase):
    def setUp(self):
        super().setUp()
        self.zshrc = self.home / ".zshrc"
        self.zshrc.write_text("export PATH=/opt/homebrew/bin:$PATH", encoding="utf-8")
        self.google.add_user("813579@seq.org", "kai-morales")

    def login(self, *typed):
        run = self.wildhat("login", "--email", "813579@seq.org", typed=typed)
        self.assertSucceeded(run)
        return run

    def exported(self, rc) -> dict:
        """The variables a new shell gets from `rc`."""
        script = f'. "{rc}"; echo "$1=${{{ALLOW}}}"; echo "$2=${{{CREDS}}}"'
        out = subprocess.run(
            ["/bin/sh", "-c", script, "sh", ALLOW, CREDS],
            capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}, check=True,
        ).stdout
        return {k: v for k, v in (line.split("=", 1) for line in out.splitlines()) if v}

    def test_yes_adds_the_lines_and_a_new_shell_gets_them(self):
        run = self.login(self.when_asked(QUESTION, "y"))

        self.assertIn("Added them to ~/.zshrc", run.out)
        # The path has a space in it ("home dir"), so this checks the quoting.
        self.assertEqual(self.exported(self.zshrc), {ALLOW: "1", CREDS: str(self.state_dir / "cred.json")})
        self.assertTrue(self.zshrc.read_text(encoding="utf-8").startswith("export PATH=/opt/homebrew/bin:$PATH\n"))

    def test_just_pressing_enter_means_yes(self):
        self.login(self.when_asked(QUESTION, ""))

        self.assertIn(ALLOW, self.exported(self.zshrc))

    def test_logging_in_again_doesnt_ask_or_add_them_twice(self):
        self.login(self.when_asked(QUESTION, "y"))
        before = self.zshrc.read_text(encoding="utf-8")

        run = self.login()

        self.assertNotIn(QUESTION, run.out)
        self.assertEqual(self.zshrc.read_text(encoding="utf-8"), before)

    def test_no_leaves_the_file_alone_and_shows_the_lines(self):
        for answer in ("n", None):
            with self.subTest(answer=answer):
                run = self.login(*([self.when_asked(QUESTION, answer)] if answer else []))

                self.assertEqual(self.zshrc.read_text(encoding="utf-8"), "export PATH=/opt/homebrew/bin:$PATH")
                self.assertIn(f"export {ALLOW}=1", run.out)
                self.assertIn(QUESTION, run.out)

    def test_bash_uses_its_own_startup_file(self):
        rc = self.home / (".bash_profile" if sys.platform == "darwin" else ".bashrc")
        os.environ["SHELL"] = "/bin/bash"

        self.login(self.when_asked(f"Add them to ~/{rc.name}?", "y"))

        self.assertIn(ALLOW, self.exported(rc))

    def test_other_shells_just_get_the_lines(self):
        os.environ["SHELL"] = "/usr/bin/fish"

        run = self.login()

        self.assertIn("your shell's startup file needs", run.out)
        self.assertIn(f"export {ALLOW}=1", run.out)
        self.assertNotIn("[Y/n]", run.out)

    def test_someone_elses_credentials_are_kept(self):
        os.environ[CREDS] = "/Users/me/keys.json"

        run = self.login(self.when_asked(QUESTION, "y"))

        self.assertIn("Leaving GOOGLE_APPLICATION_CREDENTIALS set to /Users/me/keys.json", run.out)
        self.assertEqual(self.exported(self.zshrc), {ALLOW: "1"})

    def test_logout_removes_the_lines(self):
        self.login(self.when_asked(QUESTION, "y"))

        run = self.wildhat("logout")

        self.assertSucceeded(run)
        self.assertIn("Removed wildhat's lines from ~/.zshrc", run.out)
        self.assertEqual(self.zshrc.read_text(encoding="utf-8"), "export PATH=/opt/homebrew/bin:$PATH\n")
        self.assertEqual(self.exported(self.zshrc), {})


if __name__ == "__main__":
    unittest.main()
