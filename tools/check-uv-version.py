"""Refuse a lock/pyproject change made with a uv too old to read this project's settings.

Both pyproject files set a *relative* resolution window:

    [tool.uv]
    exclude-newer = "2 days"

A uv older than MINIMUM cannot parse that value. It does not stop — it aborts settings
discovery with a warning and then resolves with no window at all, writing a lock file that
looks entirely normal while ignoring the project's resolution policy and dropping the
[options] block that records it. `uv lock` exits 0, and the damage is only visible to
someone who compares lock headers.

uv's own `required-version` cannot guard this: it lives in the same [tool.uv] table, so the
parse failure discards it too, and any uv able to read it is new enough by definition.
Hence an external check.

Runs from .pre-commit-config.yaml, restricted to the four files whose contents a stale uv
would corrupt. pre-commit also runs in CI, so this covers both.
"""

import re
import shutil
import subprocess
import sys

# First uv release that parses a relative `exclude-newer` and writes
# `exclude-newer-span` into the lock's [options] block. Established by bisection over the
# releases between 0.9.0 and 0.12.3: 0.9.16 warns and emits neither; 0.9.17 does both.
# Do not lower without re-testing that boundary.
MINIMUM = (0, 9, 17)

VERSION_PATTERN = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def installed_version() -> tuple[int, int, int]:
    """Return the running uv's version, or exit with an actionable message."""
    if shutil.which("uv") is None:
        sys.exit(
            "uv is not installed, but this commit changes a file uv owns "
            "(pyproject.toml or uv.lock). Install uv "
            f">={'.'.join(map(str, MINIMUM))} — see https://docs.astral.sh/uv/"
        )

    try:
        output = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        sys.exit(f"Could not run `uv --version`: {error}")

    match = VERSION_PATTERN.search(output)
    if match is None:
        sys.exit(f"Could not read a version out of `uv --version` output: {output.strip()!r}")

    return tuple(int(part) for part in match.groups())


def main() -> None:
    version = installed_version()
    if version < MINIMUM:
        readable = ".".join(map(str, version))
        required = ".".join(map(str, MINIMUM))
        sys.exit(
            f'uv {readable} is too old to read this project\'s `exclude-newer = "2 days"`.\n'
            f"It would resolve with no time window and write a lock file that silently\n"
            f"ignores that setting. Upgrade to {required} or newer:\n"
            f"    uv self update\n"
        )


if __name__ == "__main__":
    main()
