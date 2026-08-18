import datetime as dt
from pathlib import Path

from llm_benchmark.config import load_dotenv


def pytest_configure(config):
    """Rewrite the pytest-html output path to a unique UTC-timestamped file so
    no run ever overwrites another (a structural guarantee). A descriptive path
    passed explicitly (e.g. --html=reports/report-full-live.html) is preserved -
    only the default is auto-renamed - so a report meant to be committed keeps
    its name. Carried verbatim from P0-P4."""
    load_dotenv()
    if not hasattr(config.option, "htmlpath"):
        return
    config.option.self_contained_html = True
    if config.option.htmlpath not in (None, "reports/report.html"):
        return
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    config.option.htmlpath = str(reports / f"report-{ts}.html")
