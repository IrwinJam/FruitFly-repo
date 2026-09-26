# AMongus Fly v1.0: connectome flies play Hide n Seek

Every fly is a spiking model of the fruit-fly central nervous system built from the MaleCNS v1.0 connectome
(22,686 simulated neurons). The wiring is never trained; training tunes only a small set of engineered
parameters around it, all listed in `docs/PARAMETERS.md`.

## Assets
- **`amongusfly_showcase.webm`** (12.8 MB, 47 s): the showcase video, real unedited game time, one clip per story
  beat.
- **`amongusfly_showcase_replays.zip`** (72.4 MB): the nine recorded games with every fly's spikes, for the
  viewer. Unzip into `viewer/public/replays/`; see the README.

## Main results (`docs/SHOWCASE.md`)
- **Real wiring matters.** With degree-preserving shuffled wiring and the same training budget, flies explore
  2.1 rooms instead of 8.2, the seeker wins 0 of 40 games against scripted hiders, and hiders survive half as
  long.
- **Two small cell types carry the seeker.** Silencing 24 PFL3 or 12 PFL2 neurons in the seeker alone cuts its
  wins in full-length rounds from 13/20 to 1/20 and 0/20, and both hold with the readout rule that reads PFL2
  switched off; silencing visual pursuit (LC10a) does not significantly.
- **The whole connectome.** On all 165,122 neurons the compass circuit still steers flies to an unseen goal
  (83% vs 18% without the goal); the walker's trained settings do not transfer.
- **At scale.** Over 300 further games on fresh seeds the seeker wins 91% against three hiders (95% CI 86–94%)
  and 76% against five (67–83%).
- **Measured limit, and a way around it.** The frozen model's steering ignores heading errors of about −25° to
  40° at every setting that still steers, so flies touch walls 36–42% of the time. Route, readout and
  learned-sense fixes failed; driving the compass input at 0.75× halves wall contact and, on fresh seeds,
  lets the seeker win 20/20 games against scripted hiders instead of 14/20 and 99% of the 300-game
  verification against three hiders instead of 91%.
- **Known gap.** The odour channels (danger meter, pings) reach no neurons; the flies react to them through the
  engineered policies only. The fix is left to follow-up work.

## Reproduce
`amongusfly/train/run_pipeline.sh` (training, every evaluation and control, the showcase and its checks), then
`run_obstacle_sense.sh`. One RTX 2060 SUPER; no paid compute. See the README.

Code: MIT. Connectome data: MaleCNS v1.0, CC-BY 4.0 (not redistributed). Vent coordinates: SkeldJS (MIT).
Among Us and The Skeld are © Innersloth; this is a non-commercial fan research project, not affiliated with
Innersloth, and contains no game art.
