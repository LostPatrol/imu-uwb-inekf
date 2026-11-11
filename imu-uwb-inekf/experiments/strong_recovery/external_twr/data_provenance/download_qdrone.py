"""Download, verify and extract the complete fixed author-provided QDrone release."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen
import argparse
import hashlib
import json
import zipfile

# This revision was fixed before any estimator results were inspected.
REPO = "QDrone/UWB_IMU_GT_QDrone_Benchmark_Dataset"
REVISION = "cd3774c9db03899afde886a52f06687bbe4a1ed1"


def download(item, root):
    """Verify downloaded files against author-repository LFS or Git blob hashes."""
    path = root / item["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{quote(item['path'])}?download=true"
        temporary = path.with_suffix(path.suffix + ".partial")
        with urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        temporary.replace(path)
    content = path.read_bytes()
    assert len(content) == item["size"], str(path)
    sha256 = hashlib.sha256(content).hexdigest()
    if "lfs" in item:
        assert sha256 == item["lfs"]["oid"], str(path)
    else:
        blob = f"blob {len(content)}\0".encode() + content
        assert hashlib.sha1(blob).hexdigest() == item["oid"], str(path)
    return {"path": item["path"], "bytes": len(content), "sha256": sha256}


def main():
    """Preserve original ZIPs and write a complete source/extraction manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("E:/datasets/qdrone-benchmark"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true&expand=false"
    with urlopen(api, timeout=30) as response:
        tree = json.load(response)
    (args.output / "qdrone_hf_tree.json").write_text(json.dumps(tree, indent=2), encoding="utf-8")
    files = [item for item in tree if item["type"] == "file"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(lambda item: download(item, args.root), files))
    members = []
    for item in files:
        source = args.root / item["path"]
        if source.suffix != ".zip":
            continue
        destination = args.root / "extracted" / source.parent.name / source.stem
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                assert target.is_relative_to(destination.resolve()), member.filename
                if not member.is_dir():
                    members.append({"archive": item["path"], "member": member.filename,
                                    "bytes": member.file_size, "local_path": str(target)})
            archive.extractall(destination)
    result = {"repo": REPO, "revision": REVISION, "files": manifest, "members": members}
    (args.output / "qdrone_download_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"artifacts": len(files), "verified_bytes": sum(x["bytes"] for x in manifest),
                      "csv_members": len(members)}, indent=2))


if __name__ == "__main__":
    main()
