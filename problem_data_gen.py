import numpy as np
import numpy.linalg as la
import numpy.random as npr
import scipy.linalg as sla
from midpoint_policy_iteration import get_initial_gains, policy_iteration, midpoint_policy_iteration, verify_are

import sys
sys.path.insert(0, '../utility')
from matrixmath import mdot, specrad, solveb, dare_gain, is_pos_def, svec2, smat2, kron, vec


def gen_rand_AB(n=4, m=3, rho=None, seed=1, round_places=1):
    npr.seed(seed)
    if rho is None:
        rho = 0.9
    A = npr.randn(n, n)
    B = npr.rand(n, m)
    if round_places is not None:
        A = A.round(round_places)
        B = B.round(round_places)
    A = A * (rho / specrad(A))
    return A, B


def gen_rand_problem_data(n=4, m=3, rho=None, seed=1):
    npr.seed(seed)
    A, B = gen_rand_AB(n, m, rho, seed)
    Q = np.eye(n)
    R = np.eye(m)
    S = sla.block_diag(Q, R)
    problem_data_keys = ['A', 'B', 'S']
    problem_data_values = [A, B, S]
    problem_data = dict(zip(problem_data_keys, problem_data_values))
    return problem_data