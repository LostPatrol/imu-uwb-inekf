"""Check all predetermined trial-2 sensor timestamp coverage without reading poses."""

import json
from pathlib import Path
import pandas as pd

# Frozen independent cohort: every existing geometry/protocol combination.
HERE=Path(__file__).resolve().parent
DATA=Path("E:/datasets/dataset/flight-dataset/csv-data")
TRIAL=2
RESUME=25.
GAPS=(2.,5.,10.,20.)
REQUIRED_END=36.


def main():
    """Retain missing or short recordings explicitly instead of replacing them."""
    rows=[]
    for geometry in (1,2,3,4):
        for protocol in (2,3):
            name=f"const{geometry}-trial{TRIAL}-tdoa{protocol}"+("-traj1" if geometry==4 else "")
            path=DATA/f"const{geometry}"/(name+".csv")
            row=dict(name=name,geometry=geometry,tdoa_protocol=protocol,exists=path.exists(),sufficient_length=False)
            if path.exists():
                frame=pd.read_csv(path,usecols=["t_acc","t_gyro","t_tdoa","t_pose"])
                streams={key:frame[key].dropna().to_numpy(float) for key in frame}
                origin=max(streams[key][0] for key in ("t_acc","t_gyro","t_tdoa"))
                row.update(origin=float(origin),required_end_absolute_s=float(origin+REQUIRED_END))
                for key,values in streams.items():
                    row[key+"_count"]=len(values)
                    row[key+"_last_relative_s"]=float(values[-1]-origin)
                row["sufficient_length"]=all(values[-1]>=origin+REQUIRED_END for values in streams.values())
            rows.append(row)
    output=HERE/"independent"
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output/"inventory.csv",index=False)
    protocol=dict(trial=TRIAL,names=[row["name"] for row in rows],nominal_yaw_offset_deg=0,
        resume_s=RESUME,gap_lengths_s=GAPS,required_coverage_end_s=REQUIRED_END,
        score_windows_after_resume_s=[[0,2],[2,5],[6,11]],
        initial="sensor-only0.6s, exact original main invariant P, same native streams",
        backend="unchanged author class or full-trajectory validated numerical author mirror; no retuning",
        adaptation="native Toronto TDoA2 ring/TDoA3 complete pairs; no truth alignment",
        variants=["K","always_C","guarded"],truth="only sensor/pose timestamps read for this inventory, no evaluation pose coordinates",
        cohort="independent trial2; mechanism trial4 scores must remain separate",workers_max=6)
    (output/"protocol_pending_shadow.json").write_text(json.dumps(protocol,indent=2),encoding="utf8")
    print(json.dumps(rows,indent=2))


if __name__=="__main__":
    main()
