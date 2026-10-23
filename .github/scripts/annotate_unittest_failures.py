"""Turn each failing test in `python -m unittest -v` output into a GitHub
annotation, so failures show on the run's summary page.

Usage: python annotate_unittest_failures.py test-output.txt
"""

import re
import sys


def escape(text: str, *extra: str) -> str:
    for char in ("%", "\r", "\n", *extra):
        text = text.replace(char, f"%{ord(char):02X}")
    return text


output = open(sys.argv[1], encoding="utf-8").read()
for block in output.split("=" * 70 + "\n")[1:]:
    title, _, body = block.partition("-" * 70 + "\n")
    body = re.split(r"\n-{70}\n", body)[0]
    print(f"::error title={escape(title.strip(), ':', ',')}::{escape(body.strip())}")
