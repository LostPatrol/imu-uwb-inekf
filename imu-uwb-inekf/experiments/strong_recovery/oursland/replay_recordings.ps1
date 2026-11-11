# Reproduce fixed per-record four-mask runs from the workspace root; reuse complete K caches.
$py = './.venv/Scripts/python.exe'
$runner = 'imu-uwb-inekf/experiments/strong_recovery/oursland/replay_one.py'
& $py $runner const1-trial2-tdoa2
& $py $runner const1-trial2-tdoa3
& $py $runner const2-trial2-tdoa2
& $py $runner const2-trial2-tdoa3
& $py $runner const3-trial2-tdoa2
& $py $runner const3-trial2-tdoa3
& $py $runner const4-trial2-tdoa2-traj1
& $py $runner const4-trial2-tdoa3-traj1
& $py $runner const4-trial2-tdoa2-traj2 --supplementary
& $py $runner const4-trial2-tdoa2-traj3 --supplementary
& $py $runner const4-trial2-tdoa3-traj2 --supplementary
& $py $runner const4-trial2-tdoa3-traj3 --supplementary
