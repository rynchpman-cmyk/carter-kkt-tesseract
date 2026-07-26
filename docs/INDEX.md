# Documentation index

This documentation follows the project in the order it was created and then
explains the final system from the inside out.

1. [Origin story](ORIGIN_STORY.md) — six stages from scaled addition to the
   Carter/KKT recurrence.
2. [Mathematical model](MATHEMATICAL_MODEL.md) — the literal map, KKT sector,
   Carter closure, and boundary certificates.
3. [Architecture](ARCHITECTURE.md) — source-of-truth files, runtime dataflow,
   kernels, and artifacts.
4. [Differentiation](DIFFERENTIATION.md) — selected-system implicit VJP,
   mixed-direction neural reverse, and piecewise BPTT.
5. [Lyapunov conditioning](LYAPUNOV_CONDITIONING.md) — why the baseline
   explodes and how the scale-1 attractor is trained.
6. [Visualization](VISUALIZATION.md) — the basin atlas, cobweb, KKT phase map,
   and 4D Carter projection.
7. [Reproducibility](REPRODUCIBILITY.md) — environments, commands, generated
   artifacts, and validation tiers.
8. [Publishing](PUBLISHING.md) — recommended GitHub metadata and publication
   commands.
9. [Theory 3.3 physical hybrid](THEORY33_HYBRID.md) — future-timelike
   currents, analytic M1 closure, hard physical gates, and characteristic
   validation.
10. [Theory 3.3 experiment report](../THEORY33_EXPERIMENT_REPORT.md) —
    literature-motivated nonsmooth differentiation and neural constitutive
    experiments.
11. [Advanced frozen-constitutive report](../THEORY33_ADVANCED_REPORT.md) —
    independent kinetic data, phase uncertainty, PSD mobility, event-aware
    BPTT, margin continuation, replay, and Vulkan export.

12. [Numerical-relativity PDE solver](NR_PDE_SOLVER.md) — the 52-component
    CCZ4, conservative two-current, and mixed-vector CPU reference backend.
13. [NR PDE transplant report](../NR_PDE_TRANSPLANT_REPORT.md) — source
    boundary, state contract, first coupled run, regression evidence, and
    promotion gates.
14. [Frozen neural PDE promotion](../NEURAL_PDE_PROMOTION_REPORT.md) —
    active learned primitive recovery/RHS and the first analytic/frozen
    continuum ladder.
15. [Constraint-converged neural spacetime](../CONSTRAINED_NEURAL_SPACETIME_REPORT.md)
    — compatible CMC data, full evolution, and joint CCZ4 convergence.
16. [Neural spacetime frontier](../NEURAL_SPACETIME_FRONTIER_REPORT.md) —
    WENO5-Z fluxes, Strang splitting, a 128-step constrained horizon, 2D
    continuum convergence, and the complete 3D neural path.

Measured outputs are preserved separately:

- [Progressive full-physics report](../FULL_PHYSICS_UNROLL_REPORT.md)
- [Literal Lyapunov-conditioning report](../LYAPUNOV_CONDITIONING_REPORT.md)
- [Theory 3.3 experiment report](../THEORY33_EXPERIMENT_REPORT.md)
- [Advanced frozen-constitutive report](../THEORY33_ADVANCED_REPORT.md)
- [NR PDE transplant report](../NR_PDE_TRANSPLANT_REPORT.md)
- [Frozen neural PDE promotion](../NEURAL_PDE_PROMOTION_REPORT.md)
- [Constraint-converged neural spacetime](../CONSTRAINED_NEURAL_SPACETIME_REPORT.md)
- [Neural spacetime frontier](../NEURAL_SPACETIME_FRONTIER_REPORT.md)
