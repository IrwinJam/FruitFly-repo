# Phase 5 overnight status

_Generated 2026-09-15 09:18. Regenerated after every pipeline stage; see `C:\flyseek-data\phase5_overnight.log`._

## Pipeline stages

```
[stage] 2026-09-14 22:13 0 waiting for running seeker eval (pid 27560) and hider_navcore training (pid 28256)
[stage] 2026-09-15 00:54 1 hider_navcore (resume/finish)
[stage] 2026-09-15 00:54 2 held-out hider evaluation
[stage] 2026-09-15 01:54 3 seeker_navcore did NOT beat the explorer (decision=retrain) -> seeker_navcore_v2 (8 matches per candidate)
[stage] 2026-09-15 04:02 3b held-out evaluation of seeker_navcore_v2
[stage] 2026-09-15 09:17 RESTART after Windows Update reboot at 04:22 (pipeline killed during 3b)
[stage] 2026-09-15 09:17 0 waiting for running seeker eval (pid none) and hider_navcore training (pid none)
[stage] 2026-09-15 09:17 1 hider_navcore (resume/finish)
[stage] 2026-09-15 09:18 2 held-out hider evaluation
[stage] 2026-09-15 09:18 3 seeker_navcore did NOT beat the explorer (decision=retrain) -> seeker_navcore_v2 (8 matches per candidate)
```

## Training runs

| Run | Generations | Fitness mean, first 5 gens | Fitness mean, last 5 gens | Best single | Seeker win rate, last 5 | Stuck s, last 5 |
|---|---|---|---|---|---|---|
| seeker_navcore | 25 | 0.68 | 0.54 | 1.46 | 0.06 | 47.5 |
| hider_navcore | 25 | 0.27 | 0.30 | 0.73 | 0.95 | 19.5 |
| seeker_navcore_v2 | 20 | 0.68 | 0.75 | 1.25 | 0.16 | 45.0 |

![training curves](phase5_training_curves.png)

Round-to-round swings mostly reflect which matches were drawn that round (every candidate in a round plays the same seeds). Only the held-out evaluations below decide whether training helped.

## Held-out: `phase5_eval_seeker` (seeker, 40 matches, preset short)

| Condition | Fitness (95% CI) | Seeker wins | Stuck s | vs first condition (diff, paired t p) |
|---|---|---|---|---|
| explorer | 0.664 [0.506, 0.823] | 3/40 | 49.1 | reference |
| trained_best | 0.481 [0.329, 0.633] | 1/40 | 54.7 | -0.184, p=0.034 |
| trained_mean | 0.774 [0.590, 0.958] | 7/40 | 42.3 | +0.110, p=0.22 |
| random | 0.052 [-0.000, 0.104] | 0/40 | 4.0 | -0.612, p=4.2e-09 |
| scripted | 1.486 [1.413, 1.558] | 35/40 | – | +0.821, p=2.9e-14 |
| lc10a_off | 0.519 [0.352, 0.687] | 5/40 | 52.6 | -0.145, p=0.13 |
| pfl3_off | 0.493 [0.337, 0.649] | 1/40 | 65.4 | -0.171, p=0.037 |

## Held-out: `phase5_eval_hider` (hider, 40 matches, preset short)

| Condition | Fitness (95% CI) | Seeker wins | Hider survival s (95% CI) | Stuck s | vs first condition (diff, paired t p) |
|---|---|---|---|---|---|
| explorer | 0.244 [0.180, 0.307] | 39/40 | 21.5 [16.1, 27.0] | 12.8 | reference |
| trained_best | 0.352 [0.282, 0.421] | 39/40 | 31.3 [25.4, 37.2] | 24.7 | +0.108, p=0.015 |
| trained_mean | 0.251 [0.209, 0.293] | 39/40 | 22.2 [18.6, 25.8] | 16.6 | +0.007, p=0.85 |
| random | 0.202 [0.161, 0.243] | 40/40 | 18.2 [14.5, 21.8] | 0.8 | -0.042, p=0.2 |
| scripted | 0.494 [0.431, 0.556] | 35/40 | 42.5 [37.4, 47.7] | – | +0.250, p=5.5e-08 |
| loom_off_trained | 0.292 [0.232, 0.352] | 39/40 | 25.9 [20.7, 31.1] | 20.8 | +0.048, p=0.25 |
| dnp01_off_trained | 0.297 [0.226, 0.368] | 39/40 | 26.3 [20.2, 32.4] | 20.6 | +0.053, p=0.26 |
| pfl3_off_trained | 0.146 [0.122, 0.170] | 40/40 | 13.1 [11.0, 15.3] | 7.2 | -0.098, p=0.0096 |
