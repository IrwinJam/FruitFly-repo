# AMongus Fly design

Simulated fruit-fly brains, built from the MaleCNS v1.0 connectome, control the players in *Among Us*
Hide n Seek on The Skeld: one seeker and three or five hiders, each with a live panel of its own brain
activity. This page summarises the design and the prior work it builds on; results are in
[SHOWCASE.md](SHOWCASE.md).

## Architecture

```
game world -> senses (encoders) -> connectome (LIF, frozen) -> motor readout -> body -> game world
                                        ^
               route planner / role policy: goal direction (FC2) + compass heading (EPG)
```

| Part | Source | What it does |
|---|---|---|
| Neurons and synapses | Connectome | MaleCNS v1.0; synapse counts become weights, the neurotransmitter sets the sign |
| Neuron model | Shiu et al. (2024) | Leaky integrate-and-fire, one global weight scale derived from dataset statistics |
| Simulated subgraph | Chosen | `navcore`: 22,686 neurons between the sensory inputs and the descending neurons |
| Senses | Encoders into identified neurons | Seeing a hider (LC10a), a looming seeker (LC4, LPLC2), optional wall proximity (LLPC1) |
| Compass and goal | Encoders into identified neurons | Heading bump on EPG, goal bump on FC2; the network computes the turn (PFL3, PFL2) |
| Where to go | Route planner and role policies | Destination choice, chasing, hiding; uses only what a player of that role knows |
| Readout | Trained, bounded | Descending-neuron rates (DNa02, DNa03, DNg13, DNp01, MDN) to turn, dash and back |
| Body | Kinematic | 2D unicycle with wall sliding; Among Us is top-down, so no legs are needed |
| Training | CMA-ES | Tunes only the parameters around the network; no connectome weight changes |

Every parameter outside the connectome is listed with its range in [PARAMETERS.md](PARAMETERS.md).

## Design decisions

- **MaleCNS v1.0, not FlyWire.** It includes the brain and the ventral nerve cord, is public (CC-BY 4.0)
  and needs no account for bulk download.
- **A frozen connectome with a small trained adapter around it.** The question is what the wiring does
  as it is, so training may not touch it; the controls (below) test whether the wiring matters.
- **A kinematic body.** The game is 2D and top-down; a leg-level biomechanical body would cost far more
  compute without changing play.
- **Side-level senses, not a compound eye.** Each sense drives one cell type per side at a bounded rate,
  within the range where its effect on the descending neurons is wiring-specific.
- **Navigation through the central complex.** Heading and goal enter as bumps on EPG and FC2, the
  pathway the fly itself uses to turn toward a goal.
- **Resolve every cell type by name, and fail loudly.** Names differ between datasets (for example, P1 is
  annotated as pIP1, and oDN1 is a female neuron with no male counterpart).
- **Controls at every level.** Degree-preserving shuffled wiring trained with the same budget, silencing
  of named cell types, random and scripted players, held-out seeds, and decision rules written down
  before the runs they decide.

## Prior work

| Project | Relation to AMongus Fly |
|---|---|
| Shiu et al. (2024), [Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) | The LIF reference model and its parameters |
| [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain) | PyTorch implementation of the same model; activation and silencing interface |
| [fly-brain-minecraft](https://github.com/blendi-remade/fly-brain-minecraft) | MaleCNS in a game; neurons shown at their soma positions |
| [Fly64](https://github.com/ornata/fly) | MaleCNS playing Mario; DNg100 forward, DNa02/DNg13 steering |
| [DOOMFLY](https://github.com/nftechie/doomfly) | Connectome playing DOOM with plasticity in the mushroom body |
| [fly-craftax](https://github.com/liuzihe02/fly-craftax) | Frozen connectome with a trained linear readout |
| [nfly](https://github.com/zhengxuyu/nfly) | MaleCNS as a rate network with trained per-edge gains on single-agent games |
| Jin et al. (2026), arXiv:2602.17997 | Connectome graph model trained with deep RL to control a biomechanical body |
| [NeuroMechFly v2](https://www.nature.com/articles/s41592-024-02497-y), [flybody](https://www.nature.com/articles/s41586-025-09029-4) | Biomechanical fly bodies (not used here) |

## References

**Data**
- [MaleCNS connectome](https://male-cns.janelia.org/) ([download](https://male-cns.janelia.org/download/)) ·
  Berg et al. (2026), [Sexual dimorphism in the complete connectome of the *Drosophila* male CNS](https://www.sciencedirect.com/science/article/pii/S0092867426009426), *Cell*
- [SkeldJS](https://github.com/SkeldJS/SkeldJS) generated map data (vent positions and links, MIT)

**Circuits**
- Shiu et al. (2024), [A *Drosophila* computational brain model reveals sensorimotor processing](https://pubmed.ncbi.nlm.nih.gov/37205514/), *Nature*
- Westeinde et al. (2024), [Transforming a head direction signal into a goal-oriented steering command](https://www.nature.com/articles/s41586-024-07039-2), *Nature*
- Mussells Pires et al. (2024), [Converting an allocentric goal into an egocentric steering signal](https://www.nature.com/articles/s41586-023-07006-3), *Nature*
- Hulse et al. (2021), [A connectome of the *Drosophila* central complex](https://elifesciences.org/articles/66039), *eLife*
- Wolff, Iyer & Rubin (2015), [Neuroarchitecture of the *Drosophila* central complex: protocerebral bridge](https://onlinelibrary.wiley.com/doi/full/10.1002/cne.23705), *J. Comp. Neurol.*
- [Fine-grained descending control of steering (DNa01/DNa02)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10614758/) ·
  [Neural circuit mechanisms for steering control](https://elifesciences.org/articles/102230)
- [Descending-to-motor organisation in the nerve cord (MANC)](https://elifesciences.org/articles/96084) ·
  [Moonwalker neurons and backward walking](https://pubmed.ncbi.nlm.nih.gov/35139358/)
- [Sexual arousal gates visual processing during courtship (P1, LC10a)](https://www.nature.com/articles/s41586-021-03714-w) ·
  [Parallel pathways for visual object pursuit](https://www.cell.com/neuron/fulltext/S0896-6273(26)00001-2)
- [LC4/LPLC2 looming responses and the giant fibre](https://www.sciencedirect.com/science/article/pii/S0960982219301381) ·
  [Visual projection neurons link features to behaviours](https://pmc.ncbi.nlm.nih.gov/articles/PMC5293491/)

**Game**
- [Hide n Seek](https://among-us.fandom.com/wiki/Hide_n_Seek) and [The Skeld](https://among-us.fandom.com/wiki/The_Skeld) (Among Us wiki) ·
  [Innersloth: Hide n Seek announcement](https://www.innersloth.com/new-game-mode-hide-n-seek-is-here-emergency-meeting-35/)
