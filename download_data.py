"""Rebuild data/ from the original SNAP ego-Twitter archive.

Downloads https://snap.stanford.edu/data/twitter.tar.gz (~21 MB) and extracts
only the ego networks the app uses.

Usage:  python download_data.py
"""

import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = "https://snap.stanford.edu/data/twitter.tar.gz"
EGOS = ["18836167", "13274152", "14365883", "17767841", "15741636", "629863", "40580577"]
EXTENSIONS = ["edges", "circles", "feat", "featnames", "egofeat"]
OUT = Path(__file__).parent / "data"


def main():
    OUT.mkdir(exist_ok=True)
    wanted = {f"twitter/{ego}.{ext}" for ego in EGOS for ext in EXTENSIONS}
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "twitter.tar.gz"
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, archive)
        with tarfile.open(archive) as tar:
            for member in tar:
                if member.name in wanted:
                    target = OUT / Path(member.name).name
                    target.write_bytes(tar.extractfile(member).read())
                    wanted.discard(member.name)
    if wanted:
        raise SystemExit(f"Missing from archive: {sorted(wanted)}")
    print(f"Wrote {len(EGOS)} ego networks to {OUT}/")


if __name__ == "__main__":
    main()
