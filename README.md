# Ion trapped numerics
 Simulations and calculations useful in controlling trapped ion experiments.


### Sideband Cooling
In trapped ion experiments, we often want to cool the trapped ions to the ground state. This is often done through a technique called resolved sideband cooling (RSC). The simulation package contains the **first** numerical optimization of this technique (along with other methods that are approximately optimal). Feel free to speed up RSC in your lab with this simulation tool.

Examples of how to determine the optimal pulse times for a give number of RSC pulses, see the `example simulation.py` file.

Global optimization of all pulse times is computationally expensive for large numbers of pulses, so I've started integrating some machine learning models to extrapolate what those pulse times are for large numbers of pulses. (Still under developed ment in svr.py file).

Please cite as:

[Rasmusson, A.J., et al. "Machine learning estimation of optimal resolved sideband cooling strategies." Bulletin of the American Physical Society (2021).](https://meetings.aps.org/Meeting/DAMOP21/Session/Z05.3)

### 2D Ion Crystal Formation
In the [Richerme lab](https://iontrap.physics.indiana.edu), we focus on trapping ions in 2D crystals (they're most commonly trapped in 1D chains). This project will eventually compute the normal transverse modes of a 2D crystal. Currently, I'm optimizing how the ion's equilibrium positions are determined.
