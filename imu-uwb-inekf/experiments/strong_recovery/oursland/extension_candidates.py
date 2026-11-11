"""Apply the unchanged output-only candidates to the supplementary four flights.

This cohort is reported separately from the balanced original eight flights and
the later single-write feedback extension.
"""

import argparse
import json

import feedback_runner as extension
import independent as experiment
import score_independent


def main():
    """Generate all frozen supplementary candidates before scoring this cohort."""
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("candidates","score"),required=True)
    stage=parser.parse_args().stage
    experiment.OUTPUT=extension.OUTPUT
    if stage=="candidates":
        records=[]
        for name in extension.NAMES:
            for gap in experiment.GAPS:
                path=extension.OUTPUT/"K"/f"{name}__gap{gap:g}.json"
                records.append(json.loads(path.read_text(encoding="utf8")))
        experiment.save_json(extension.OUTPUT/"K_records.json",records)
        experiment.candidates()
    else:
        score_independent.main(expected_cases=16)


if __name__=="__main__":
    main()
