# Ion trapped numerics
 Simulations and calculations useful in controlling trapped ion experiments.


### Sideband Cooling
In trapped ion experiments, we often want to cool the trapped ions to the ground state. This is often done through a technique called resolved sideband cooling (RSC). The simulation package contains the **first** numerical optimization of this technique (along with other methods that are approximately optimal). Feel free to speed up RSC in your lab with this simulation tool.

Examples of how to determine the optimal pulse times for a give number of RSC pulses, see the `example simulation.py` file.

Global optimization of all pulse times is computationally expensive for large numbers of pulses, so I've started integrating some machine learning models to extrapolate what those pulse times are for large numbers of pulses. (Still under developed ment in svr.py file).

Please cite as:

[Rasmusson, A.J., et al. "Machine learning estimation of optimal resolved sideband cooling strategies." Bulletin of the American Physical Society (2021).](https://meetings.aps.org/Meeting/DAMOP21/Session/Z05.3)

### Ion Interactions (only 2D ion crystals at this point)
In the [Richerme lab](https://iontrap.physics.indiana.edu), we focus on trapping ions in [2D crystals](https://arxiv.org/abs/2012.12766). (They're most commonly trapped in 1D chains.) So, that's what I'm doing.

This project currently computes: 2D ion equilibrium positions assuming psuedopotential; 2D ion crystal traverse modes; and 2D ion crystal Ising interaction strengths (J_ij).

Features under construction: faster equilibrium position calculation; and the numerics for determining the [optimal pulse shaping](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.126.220503) for a [Molmer Sorenson gate](https://en.wikipedia.org/wiki/M%C3%B8lmer%E2%80%93S%C3%B8rensen_gate).
