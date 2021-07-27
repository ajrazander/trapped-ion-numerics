# Ion trapped numerics
 Simulations and calculations useful in controlling trapped ion experiments.


### Sideband Cooling
In trapped ion experiments, we often want to cool the ions down to their motional ground state. This is often done through a technique called resolved sideband cooling (SBC). This simulation package contains the **first** \[1] numerical optimization of this technique (along with other methods that are approximately optimal). Feel free to speed up RSC in your lab with this simulation tool.

Examples of how to determine the optimal pulse times for a given number of SBC pulses, see the `example simulation.py` file.

Global optimization of all pulse times is computationally expensive for large numbers of pulses, so I've started integrating some machine learning models to extrapolate what those pulse times are for large numbers of pulses [2]. (Still under developedment in svr.py file).

Please cite my work:
\[1] [Rasmusson, A.J., et al. "Optimized sideband cooling and enhanced thermometry of trapped ions." arXiv:2107.11802 [quant-ph] (2021).](https://arxiv.org/abs/2107.11802)
\[2] [Rasmusson, A.J., et al. "Machine learning estimation of optimal resolved sideband cooling strategies." Bulletin of the American Physical Society (2021).](https://meetings.aps.org/Meeting/DAMOP21/Session/Z05.3)


### Ion Interactions (only 2D ion crystals at this point)
In the [Richerme lab](https://iontrap.physics.indiana.edu), we focus on trapping ions in [2D crystals](https://arxiv.org/abs/2012.12766). (They're usually trapped in 1D chains.)

This project project some fundamental properties of a trapped 2D ion crystal such as: ion equilibrium positions assuming  pseudopotential; transvere modes of the ion crystal; and Ising interaction strengths (J_ij) coupled through those transvere modes.

Features under construction:
* faster equilibrium position calculation
* the numerics for determining the [optimal pulse shaping](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.126.220503) for a [Molmer Sorenson gate](https://en.wikipedia.org/wiki/M%C3%B8lmer%E2%80%93S%C3%B8rensen_gate)
