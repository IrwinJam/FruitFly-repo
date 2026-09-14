# Phase 4.5 exploration training on The Skeld, then same-budget controls.
# Resumable: re-running continues each run from its checkpoint.
$ErrorActionPreference = "Continue"
$py = ".\.venv\Scripts\python.exe"
$common = @("--generations", "30", "--pop", "12", "--seeds-per-candidate", "2", "--seconds", "45")

Write-Output "=== explore_navcore (real wiring) ==="
& $py -W ignore -m flyseek.train.es --run explore_navcore --graph navcore @common

Write-Output "=== explore_shuf0 (shuffled wiring control) ==="
& $py -W ignore -m flyseek.train.es --run explore_shuf0 --graph navcore_shuf0 @common

Write-Output "=== explore_pfl3off (PFL3 silenced control) ==="
& $py -W ignore -m flyseek.train.es --run explore_pfl3off --graph navcore --silence PFL3 @common

Write-Output "=== PHASE 4 TRAINING DONE ==="
