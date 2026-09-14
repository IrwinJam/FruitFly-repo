# Phase 3 vertical-slice matches (short preset, navcore brains, 1 seeker + 3 hiders).
# Pass a comma-separated list of match names to run a subset, e.g.:
#   powershell -File flyseek\world\run_phase3_matches.ps1 -OnlyList "match02_brain_seeker_vs_scripted,match03_scripted_seeker_vs_brain_hiders"
# (powershell -File binds only the first bare argument to an array parameter, which
#  silently skipped match03 once, hence the single comma-separated string.)
param([string]$OnlyList = "")
$Only = if ($OnlyList) { $OnlyList.Split(",") } else { @() }
$ErrorActionPreference = "Continue"
$py = ".\.venv\Scripts\python.exe"
$runs = @(
  @{ brains = "all";    name = "match01_all_brains" },
  @{ brains = "seeker"; name = "match02_brain_seeker_vs_scripted" },
  @{ brains = "hiders"; name = "match03_scripted_seeker_vs_brain_hiders" }
)
foreach ($r in $runs) {
  if ($Only -and ($Only -notcontains $r.name)) { continue }
  Write-Output "=== $($r.name) ==="
  & $py -W ignore -m flyseek.world.match --brains $r.brains --preset short --seed 0 --name $r.name
  & $py -W ignore -m flyseek.world.export_replay $r.name
}
Write-Output "=== PHASE 3 MATCHES DONE ==="
