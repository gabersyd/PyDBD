# PyDBD
A software based on one dimensional fluid modelling of argon Dielectric Barrier Discharge.


The code solves particle and energy continuity equation using drift-diffusion approximation.
Drift and diffusion equations are decoupled and solved separately.
The drift part is solved using explicit second order upwind scheme.
Whereas the diffusion equation is solved using second order implicit scheme. 

Multi-regions(plasma and dielectrics) are coupled strongly (monolithic solver) while solving the poisson equation.
Whereas the continuity (transport) equation is solved for the plasma region only. 


For any publication that involves PyDBD, citation have to be made to the following paper. 
https://iopscience.iop.org/article/10.1088/2058-6272/ac241f




<a href="https://mybinder.org/v2/gh/gabersyd/PyDBD/master">
<img src=https://mybinder.org/badge_logo.svg target: https://mybinder.org/v2/gh/gabersyd/PyDBD/master
</a>
