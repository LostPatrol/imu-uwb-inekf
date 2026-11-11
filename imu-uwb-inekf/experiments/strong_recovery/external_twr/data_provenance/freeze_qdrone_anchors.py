"""Freeze QDrone anchor coordinates directly from the publisher's HTML tables."""

from html import unescape
from pathlib import Path
from urllib.request import urlopen
import argparse
import hashlib
import json
import re

# Publisher labels map to the fixed raw archive site names, never to fitted geometry.
SOURCE_URL = "https://benchmark.qdrone.ausmlab.com/datasets/"
LABELS = {"Indoor": "indoor", "Outdoor Field": "field", "Outdoor Structure": "building",
          "Bridge": "bridge", "Tunnel": "tunnel"}
OUT = Path(__file__).resolve().parent


def parse_anchors(content):
    """Extract each labelled group of four UWB coordinates without hand copying."""
    text = unescape(re.sub(r"<[^>]+>", " ", content.decode("utf-8")))
    text = re.sub(r"\s+", " ", text)
    sites, evidence = {}, []
    for label, site in LABELS.items():
        label_pattern = re.escape(label) + r"\s+UWB0\s*\("
        starts = list(re.finditer(label_pattern, text))
        assert len(starts) == 1, (label, len(starts))
        section = text[starts[0].start():starts[0].start() + 350]
        matches = list(re.finditer(r"UWB([0-3])\s*\(([^)]+)\)", section))[:4]
        assert [int(match[1]) for match in matches] == [0, 1, 2, 3], section
        anchors = {}
        for match in matches:
            values = re.findall(r"[-+]?\d+(?:\.\d+)?", match[2])
            assert len(values) == 3, match[0]
            anchors[str(100 + int(match[1]))] = [float(value) for value in values]
        sites[site] = anchors
        evidence.append({"publisher_label": label, "archive_site": site,
                         "verbatim_table_coordinate_text": section[:matches[-1].end()],
                         "anchors": anchors})
    return sites, evidence


def main():
    """Save original response bytes, hash, parsed coordinates and source evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Fetch current publisher page instead of frozen HTML")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = OUT / "qdrone_datasets_source.html"
    if args.refresh or not snapshot.exists():
        with urlopen(SOURCE_URL, timeout=30) as response:
            content = response.read()
            headers = dict(response.headers)
    else:
        content = snapshot.read_bytes()
        previous = json.loads((OUT / "qdrone_anchor_provenance.json").read_text(encoding="utf-8"))
        headers = previous["response_headers"]
    (args.output / snapshot.name).write_bytes(content)
    sites, evidence = parse_anchors(content)
    output = args.output / "qdrone_anchors_final.json"
    # Serialize exactly as in the verified Windows replay, independent of the current OS.
    output.write_bytes((json.dumps(sites, indent=2) + "\n").replace("\n", "\r\n").encode("utf-8"))
    manifest = {"source_url": SOURCE_URL, "source_sha256": hashlib.sha256(content).hexdigest(),
                "response_headers": headers, "units": "meters",
                "published_labels": "UWB0..UWB3", "raw_numeric_ids": "100..103",
                "id_mapping": "numeric raw ID 100+i maps to publisher UWB i; fixed before range/GT audit",
                "source_evidence": evidence,
                "final_json_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "supersedes": "Both earlier strong_data/qdrone_anchors JSONs were erroneous hand transcriptions; all resulting evaluations invalidated."}
    (args.output / "qdrone_anchor_provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
