# midpoint-policy-iteration
 ## Exact and approximate midpoint policy iteration for linear quadratic control
 
 The code in this repository implements the algorithms and ideas from our paper:
 
 "Approximate Midpoint Policy Iteration for Linear Quadratic Control"
 * [Proceedings of Machine Learning Research (PMLR) / Learning for Dynamics & Control (L4DC) 2021](http://proceedings.mlr.press/v144/gravell21a.html)
 * [arXiv](https://arxiv.org/abs/2011.14212)
 * [Tech report](https://personal.utdallas.edu/~tyler.summers/papers/AMPI_extended.pdf)
  
 ## Dependencies
* Python 3.5+ (tested with 3.7.6)
* NumPy
* SciPy
* Matplotlib

## Installing
Currently there is no formal package installation procedure; simply download this repository and run the Python files.

## Examples

### experiments.py
This file contains the code necessary to reproduce the experiment results and plots reported in the paper.

### midpoint_policy_iteration.py
This file contains the midpoint policy iteration and vanilla policy iteration algorithms which can be applied to generic linear quadratic regulator (LQR) problems. Both the exact and approximate versions are implemented and be specified via the appropriate function arguments.


## Authors
* **Ben Gravell** - [UTDallas](http://www.utdallas.edu/~tyler.summers/)
* **Iman Shames** - [University of Melbourne](https://findanexpert.unimelb.edu.au/profile/537214-iman-shames)
* **Tyler Summers** - [UTDallas](http://www.utdallas.edu/~tyler.summers/)
