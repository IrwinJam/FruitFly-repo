# Runs the remaining Phase 1 GPU experiments in sequence (one GPU job at a time).
# Usage (repo root): powershell -File flyseek\brain\run_phase1.ps1
$ErrorActionPreference = "Continue"
$py = ".\.venv\Scripts\python.exe"
$shufs = (0..9 | ForEach-Object { "navcore_shuf$_" })

Write-Output "=== navcore vs 10 shuffles ==="
& $py -W ignore -m flyseek.brain.sanity_checks --tags navcore @shufs --replicates 10 --out phase1_sanity_navcore
& $py -W ignore -m flyseek.brain.shuffle_compare --file phase1_sanity_navcore.json --real navcore

Write-Output "=== dt check: pruned5 at dt=0.1 ms ==="
& $py -W ignore -m flyseek.brain.sanity_checks --tags pruned5 --rates 25 50 --replicates 5 --dt 0.1 --out phase1_sanity_dt01

Write-Output "=== stability, calibrated ==="
& $py -W ignore -m flyseek.brain.stability --tags navcore pruned5 pruned5_shuf0 full --rates 5 20 50 --seconds 5

Write-Output "=== active-network benchmark ==="
& $py -W ignore -m flyseek.brain.bench

Write-Output "=== full graph vs full shuffle ==="
& $py -W ignore -m flyseek.brain.sanity_checks --tags full full_shuf0 --replicates 10 --out phase1_sanity_full

Write-Output "=== PHASE 1 BATCH DONE ==="
