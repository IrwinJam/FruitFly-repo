# Rebuild all graphs with consensus neurotransmitter labels, then re-run the key
# Phase 1 checks to confirm conclusions don't change (0.21% of output weight changed sign).
$ErrorActionPreference = "Continue"
$py = ".\.venv\Scripts\python.exe"

Write-Output "=== archive pre-rebuild results ==="
New-Item -ItemType Directory -Force docs\phase1_pre_consensus_nt | Out-Null
Copy-Item docs\phase1_sanity_*.json, docs\phase1_sanity_*.md, docs\phase1_stability.* docs\phase1_pre_consensus_nt\ -Force

Write-Output "=== rebuild graphs (consensus NT) ==="
& $py -W ignore -m flyseek.connectome.build_graph
& $py -W ignore -m flyseek.connectome.celltypes
& $py -W ignore -c "from flyseek.brain.navcore import build; print(build('pruned5', 6, 3, 0.01, out_tag='navcore'))"
& $py -W ignore -m flyseek.brain.shuffle --tag pruned5 --seeds 0 1 2 | Select-Object -Last 1
& $py -W ignore -m flyseek.brain.shuffle --tag full --seeds 0 | Select-Object -Last 1
& $py -W ignore -m flyseek.brain.shuffle --tag navcore --seeds (0..99) | Select-Object -Last 1

Write-Output "=== unit tests ==="
& $py -m pytest tests -q -p no:warnings

Write-Output "=== confirm: navcore vs 100 shuffles ==="
$shufs = (0..99 | ForEach-Object { "navcore_shuf$_" })
& $py -W ignore -m flyseek.brain.sanity_checks --tags navcore @shufs --replicates 10 --out phase1_sanity_navcore100
& $py -W ignore -m flyseek.brain.shuffle_compare --file phase1_sanity_navcore100.json --real navcore
& $py -W ignore -m flyseek.brain.dna01_escape --file phase1_sanity_navcore100.json --real navcore

Write-Output "=== confirm: full vs shuffle, pruned5 vs 3 shuffles ==="
& $py -W ignore -m flyseek.brain.sanity_checks --tags full full_shuf0 --replicates 10 --out phase1_sanity_full
& $py -W ignore -m flyseek.brain.sanity_checks --tags pruned5 pruned5_shuf0 pruned5_shuf1 pruned5_shuf2 --replicates 10 --out phase1_sanity

Write-Output "=== confirm: stability + MB ignition ==="
& $py -W ignore -m flyseek.brain.stability --tags navcore pruned5 pruned5_shuf0 full --rates 5 20 50 --seconds 5
& $py -W ignore -m flyseek.brain.ignition_calibrated
& $py -W ignore -m flyseek.brain.kc_test

Write-Output "=== NT REBUILD BATCH DONE ==="
