"""Extract the pinned author's numerical implementation while removing ROS I/O."""

from pathlib import Path
import argparse
import hashlib
import json
import subprocess

# The last KIT-ISAS upstream revision is fixed for reproducibility.
COMMIT = "75bf5a32f1a8e5c3046a5bd1a1ddf659fa996f7d"


def remove_function(text, signature):
    """Remove one complete C++ function with balanced braces."""
    start = text.index(signature)
    opening = text.index("{", start)
    depth = 1
    closing = opening + 1
    while depth:
        depth += (text[closing] == "{") - (text[closing] == "}")
        closing += 1
    return text[:start] + text[closing:]


def prepare(upstream, destination, strict_author_startup=False):
    """Copy untouched numerical headers and mechanically strip ROS functions."""
    source = upstream / "sfuise"
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        raise ValueError(f"Expected {COMMIT}, got {revision}")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"commit": COMMIT, "source_url": "https://github.com/KIT-ISAS/SFUISE", "files": {}}
    for original in (source / "include").rglob("*.h"):
        relative = original.relative_to(source / "include")
        if relative.name == "PoseVisualization.h":
            continue
        content = original.read_text(encoding="utf-8")
        changes = []
        if relative.name == "common_utils.h":
            content = content.replace("#include <ros/ros.h>", "")
            content = content.replace('#include "geometry_msgs/PoseStamped.h"', "")
            content = content[:content.index("class CommonUtils")]
            changes.append("Removed ROS includes and unused CommonUtils I/O class")
        if relative.name == "SplineState.h":
            content = content.replace('#include "sfuise_msgs/Spline.h"', "")
            content = content.replace('#include "sfuise_msgs/Knot.h"', "")
            content = remove_function(content, "    void getSplineMsg(")
            changes.append("Removed ROS serialization method and message includes")
            content = content.replace("Eigen::Matrix4d Q_l[size_J - 1];", "Eigen::Matrix4d Q_l[3];")
            content = content.replace("Eigen::Matrix4d Q_r[size_J - 1];", "Eigen::Matrix4d Q_r[3];")
            changes.append("MSVC: allocate bounded length-3 arrays instead of GCC variable-length arrays; same indexing")
            if not strict_author_startup:
                content = content.replace("cps[i + idx_window] = knots[idx0 + i];", """// Author startup integrates with one knot before adding the second.
            // Repeat that sole endpoint instead of reading an absent deque element.
            cps[i + idx_window] = (knots.size() == 1 && idx0 + i == 1)
                ? knots.front() : knots.at(idx0 + i);""")
                changes.append("Startup portability correction: repeat the sole endpoint when first integration requests absent knot[1]; checked indexing otherwise")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        manifest["files"][str(relative)] = {
            "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            "generated_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "changes": changes,
        }
    original = source / "src" / "SplineFusion.cpp"
    content = original.read_text(encoding="utf-8")
    content = content[content.index("class SplineFusion"):content.index("int main(")]
    for signature in ["    SplineFusion(", "    void run()", "    void readParameters(",
                      "    void getImuCallback(", "    void getTdoaCallback(",
                      "    void getToaCallback(", "    void getAnchorCallback(",
                      "    void displayControlPoints()", "    geometry_msgs::Point32 getPointMsg("]:
        content = remove_function(content, signature)
    content = "\n".join(line for line in content.splitlines()
                        if "ros::Subscriber " not in line and "ros::Publisher " not in line
                        and "std_msgs::Int64 start_time;" not in line
                        and "start_time.data =" not in line
                        and "pub_start_time.publish" not in line)
    content = content.replace("  private:", "  public:")
    content = content.replace("size_t idx[num_data];", "std::vector<size_t> idx(num_data);")
    content = content.replace("findClosestNWithOrderedID(t_s, num_data, idx);",
                              "findClosestNWithOrderedID(t_s, num_data, idx.data());")
    content = content.replace("ROS_ERROR_STREAM(", "std::cerr << (")
    content = '#pragma once\n#include "Linearizer.h"\n#include "utils/tic_toc.h"\n' + content
    (destination / "AuthorFusion.h").write_text(content, encoding="utf-8")
    manifest["files"]["SplineFusion.cpp"] = {
        "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "generated_sha256": hashlib.sha256((destination / "AuthorFusion.h").read_bytes()).hexdigest(),
        "changes": ["Removed ROS constructor, callbacks, configuration, display, main and run publication wrapper",
                    "Exposed members to standalone wrapper; removed start-time ROS publication",
                    "Replaced TDoA-only variable-length array with std::vector for MSVC; stderr logging macro"],
    }
    (destination / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    """Parse the source and generated-header directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-author-startup", action="store_true")
    args = parser.parse_args()
    prepare(args.upstream, args.output, args.strict_author_startup)


if __name__ == "__main__":
    main()
