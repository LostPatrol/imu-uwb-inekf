"""Preserve source snapshots and check the independently published indoor geometry."""

from pathlib import Path
from urllib.request import urlopen
import argparse
import hashlib
import json
import re
import shutil
import subprocess

# Source files are retained so rechecking does not depend on changing web pages.
SOURCES = {
    "qdrone_documentation_source.html": "https://benchmark.qdrone.ausmlab.com/documentation/",
    "park2020_same_indoor_source.pdf": "https://isprs-annals.copernicus.org/articles/V-1-2020/317/2020/isprs-annals-V-1-2020-317-2020.pdf",
}


def main():
    """Check all indoor coordinates and record the limits of source verification."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--pdftotext", type=Path, help="Optional Poppler executable to regenerate retained paper text")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = []
    for name, url in SOURCES.items():
        path = args.output / name
        if not path.exists():
            with urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        sources.append({"file": name, "url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    text_file = args.output / "park2020_same_indoor_source.txt"
    converter = args.pdftotext or (shutil.which("pdftotext") if not text_file.exists() else None)
    if converter:
        subprocess.run([str(converter), "-raw", str(args.output / "park2020_same_indoor_source.pdf"),
                        str(text_file)], check=True)
    if not text_file.exists():
        parser.error("The paper text is a local cache. Supply --pdftotext PATH for the first source check.")
    text = text_file.read_text(encoding="utf-8")
    paragraph = text.split("anchors 1, 2, 3 and 4 were placed at coordinates", 1)[1].split("respectively", 1)[0]
    triples = re.findall(r"\(([-.\d]+) m,\s*([-.\d]+) m,\s*([-.\d]+) m\)", paragraph)
    paper = {str(100 + i): list(map(float, xyz)) for i, xyz in enumerate(triples)}
    final = json.loads((args.output / "qdrone_anchors_final.json").read_text(encoding="utf-8"))
    assert len(paper) == 4 and paper == final["indoor"]
    result = {
        "sources": sources,
        "paper_text_sha256": hashlib.sha256(text_file.read_bytes()).hexdigest(),
        "indoor_12_coordinates_exactly_match": True,
        "paper_coordinates_m": paper,
        "paper_section": "Park et al., Multilateration under Flip Ambiguity for UAV Positioning Using Ultra-wideband, ISPRS Annals V-1-2020, section3.2",
        "other_sites_independent_pdf_crosscheck": "not established; their tables are preserved from the official dataset website",
        "numeric_id_mapping_limit": "Website labels UWB0..3 and paper labels anchors1..4; raw IDs100..103 are mapped in the same order, fixed before residual inspection. An explicit numeric-ID correspondence statement was not located.",
        "frame_evidence": "Official documentation Data Format says uncalibrated raw data are in the UWB anchor coordinate system; per-event XYZ are supplied as prism ground truth. Outdoor range-vs-prism audit is consistent with this common-frame reading, without alignment fitting.",
        "time_evidence_limit": "Related indoor paper section3.2 states MultiStation/UAV synchronization used comparison of MultiStation and IMU height. Original RTS clock samples and interpolation procedure are not included here; do not claim independently audited hardware synchronization.",
        "lever_limit": "Prism underneath the UAV and UWB antenna on the UAV body are distinct points; no measured prism/tag/IMU extrinsic vector is supplied.",
        "calibration_policy": "Official optional per-record a*range+b is minimized against GT range and is not used by this audit or the frozen estimator input.",
        "indoor_source_anomaly": {"record": "2018-05-30-13-13-21_exp3_big_circle", "raw_csv_line": 21910,
                                  "gt_copy_csv_line": 32865, "timestamp_s": 62.827735, "anchor_id": 101,
                                  "raw_range_m": 4294961.75, "self_range_error_metadata": 65.535,
                                  "policy": "kept exactly; no reinterpretation of unsigned encoding or removal"},
    }
    (args.output / "qdrone_source_crosscheck.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"indoor_12_coordinates_exactly_match": True, "sources": sources}, indent=2))


if __name__ == "__main__":
    main()
