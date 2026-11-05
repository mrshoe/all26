"""Cloud's retries: reads and downloads are tried again after failures that
may pass, and nothing else is. No network: urlopen is faked. From ai/src:

    python -m unittest discover -s shared/tests -t .
"""

import io
import shutil
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from shared import cloud
from shared.cloud import Cloud, CloudError, TransientCloudError


class Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Broken(io.RawIOBase):
    """A reply whose connection drops partway through."""

    def readinto(self, b):
        raise ConnectionResetError("reset by peer")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "nope", {}, io.BytesIO(b'{"error": {"message": "nope"}}'))


class RetryTest(unittest.TestCase):
    def setUp(self):
        self.cloud = Cloud()
        self.cloud._token, self.cloud._expires = "token", time.time() + 3600
        self.cloud.project, self.cloud.bucket, self.cloud._handle = "p", "b", "kai-morales"
        self.sleeps: list[float] = []
        patcher = mock.patch.object(cloud.time, "sleep", self.sleeps.append)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def replies(self, *outcomes):
        """urlopen gives these in turn: an exception is raised, anything else
        returned."""
        outcomes = list(outcomes)

        def urlopen(req, timeout):
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return mock.patch.object(cloud.urllib.request, "urlopen", urlopen)

    def test_a_read_survives_a_network_blip(self):
        dns = urllib.error.URLError(OSError(8, "nodename nor servname provided, or not known"))
        with self.replies(dns, http_error(503), Reply(b'{"fields": {"a": {"integerValue": "3"}}}')):
            self.assertEqual(self.cloud.document("runs/x"), {"a": 3})
        self.assertEqual(len(self.sleeps), 2)
        self.assertTrue(1 <= self.sleeps[0] < 2 and 2 <= self.sleeps[1] < 3)

    def test_a_download_survives_a_dropped_connection(self):
        path = self.tmp / "episode.json"
        with self.replies(Broken(), Reply(b"the file")):
            self.cloud.download("runs/x/episode.json", path)
        self.assertEqual(path.read_bytes(), b"the file")
        self.assertEqual(len(self.sleeps), 1)

    def test_gives_up_in_the_end(self):
        down = urllib.error.URLError(OSError(8, "nodename nor servname provided"))
        with self.replies(*[down] * cloud.TRIES), self.assertRaisesRegex(TransientCloudError, "Are you online"):
            self.cloud.document("runs/x")
        self.assertEqual(len(self.sleeps), cloud.TRIES - 1)

    def test_lasting_failures_and_writes_arent_retried(self):
        with self.replies(http_error(403)), self.assertRaisesRegex(CloudError, "HTTP 403"):
            self.cloud.document("runs/x")
        with self.replies(http_error(503)), self.assertRaisesRegex(CloudError, "HTTP 503"):
            self.cloud.update("training/x", {"a": 1})
        self.assertEqual(self.sleeps, [])

    def test_a_missing_record_is_still_none(self):
        with self.replies(http_error(404)):
            self.assertIsNone(self.cloud.document("runs/x"))


if __name__ == "__main__":
    unittest.main()
