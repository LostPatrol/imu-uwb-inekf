"""Replay one prelisted recording without changing aggregate protocols or scores."""

import argparse
import json
import feedback_runner as extension
import independent as experiment


def main():
    """Generate all four fixed masks for the requested cohort recording."""
    parser=argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--supplementary",action="store_true")
    args=parser.parse_args()
    if args.supplementary:
        if args.name not in extension.NAMES:
            raise ValueError("Recording is not in the registered supplementary cohort")
        result=extension.replay_record(args.name)
    else:
        names=experiment.pd.read_csv(experiment.OUTPUT/"inventory.csv").name.tolist()
        if args.name not in names:
            raise ValueError("Recording is not in the registered balanced cohort")
        result=experiment.generate_record(args.name)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
