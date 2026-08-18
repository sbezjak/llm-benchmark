"""Environment loading shared by the test harness and the sweep CLI.

The paid providers read their keys from the environment. In a pytest run conftest
loads `.env`; the standalone sweep job needs the same, so the loader lives here
and both call it. Deliberately tiny instead of the python-dotenv dependency -
keys-only `KEY=value` lines are all this project's `.env` ever holds, and values
already set in the real environment win."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | str = ".env") -> None:
    env = Path(path)
    if not env.is_file():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
