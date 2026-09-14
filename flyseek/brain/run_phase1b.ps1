# Phase 1, second batch: calibrated stability + 100-shuffle null on navcore.
$ErrorActionPreference = "Continue"
$py = ".\.venv\Scripts\python.exe"

Write-Output "=== stability, calibrated ==="
& $py -W ignore -m flyseek.brain.stability --tags navcore pruned5 pruned5_shuf0 full --rates 5 20 50 --seconds 5

Write-Output "=== build navcore shuffles 10-99 ==="
& $py -W ignore -m flyseek.brain.shuffle --tag navcore --seeds (10..99) | Select-Object -Last 1

Write-Output "=== navcore vs 100 shuffles ==="
$shufs = (0..99 | ForEach-Object { "navcore_shuf$_" })
& $py -W ignore -m flyseek.brain.sanity_checks --tags navcore @shufs --replicates 10 --out phase1_sanity_navcore100
& $py -W ignore -m flyseek.brain.shuffle_compare --file phase1_sanity_navcore100.json --real navcore

Write-Output "=== PHASE 1B DONE ==="
