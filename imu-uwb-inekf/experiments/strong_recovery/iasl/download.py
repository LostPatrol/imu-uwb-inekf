"""Download the complete, fixed official i-ASL release without selecting scenarios."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request
import time
import hashlib
import json

# Pin the official repository revision before inspecting numerical results.
REVISION = "41e585b47d84e810d7b448bdb21e3d98a8a9423e"
DESTINATION = Path("E:/datasets/iasl-uwb-imu")
MANIFEST = Path(__file__).resolve().parents[4] / "agent/codex/strong_iasl/download_manifest.json"
FILES = ["README.md", "uwb.yaml"] + [
    f"scenario{i}/{sensor}.csv" for i in range(1, 4)
    for sensor in ("gt", "imu", "uwb")
] + [f"rosbag/0612_hw{i}.bag" for i in range(1, 4)]
# Official git blob hashes also detect interrupted downloads left on disk.
TREE_URL = f"https://api.github.com/repos/i-ASL/uwb_imu_dataset/git/trees/{REVISION}?recursive=1"


def download(entry):
    """Store an unmodified official source file and return its byte count."""
    relative = entry["path"]
    target = DESTINATION / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/i-ASL/uwb_imu_dataset/{REVISION}/{relative}"
    if target.exists() and valid_blob(target.read_bytes(), entry):
        return relative, target.stat().st_size
    for attempt in range(4):
        try:
            content = urllib.request.urlopen(url, timeout=20).read()
            if not valid_blob(content, entry):
                raise ValueError(f"Official git blob hash/size mismatch: {relative}")
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1)
    target.write_bytes(content)
    return relative, len(content)


def valid_blob(content, entry):
    """Compare exact bytes with the pinned official Git object."""
    prefix = f"blob {len(content)}\0".encode()
    return len(content) == entry["size"] and hashlib.sha1(prefix+content).hexdigest() == entry["sha"]


if __name__ == "__main__":
    tree = json.loads(urllib.request.urlopen(TREE_URL, timeout=20).read())
    entries = [entry for entry in tree["tree"] if entry["path"] in FILES]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(download, entries))
        for result in results:
            print(result, flush=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(dict(revision=REVISION, files=entries,
        verification="Every local byte count and Git SHA-1 blob matches the pinned official tree."),
        indent=2), encoding="utf-8")
