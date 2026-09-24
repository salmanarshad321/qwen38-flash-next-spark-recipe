"""Verify that packaged runtime sources match the inspected live image."""

from hashlib import sha256
from pathlib import Path
import sys

ROOT = Path("/")


def main() -> None:
    manifest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("LIVE_SOURCE_SHA256SUMS")
    for line in manifest.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        expected, path = line.split("  ", 1)
        actual = sha256((ROOT / path.lstrip("/")).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"Source hash mismatch: {path}: {actual} != {expected}")
    print(f"Source matches {manifest.name}")


if __name__ == "__main__":
    main()
