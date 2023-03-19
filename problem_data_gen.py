import numpy as np
import numpy.linalg as la
import numpy.random as npr
import scipy.linalg as sla

from utility.matrixmath import mdot, specrad


def gen_rand_pd(n):
    Qdiag = np.diag(npr.rand(n))
    Qvecs = la.qr(npr.randn(n, n))[0]
    Q = mdot(Qvecs, Qdiag, Qvecs.T)
    return Q


def gen_rand_AB(n=4, m=3, rho=None, seed=1, round_places=None):
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


def gen_rand_problem_data(n=4, m=3, rho=None, seed=1, penalty_rand=True):
    npr.seed(seed)
    A, B = gen_rand_AB(n, m, rho, seed)
    if penalty_rand:
        S = gen_rand_pd(n+m)
    else:
        Q = np.eye(n)
        R = np.eye(m)
        S = sla.block_diag(Q, R)
    problem_data_keys = ['A', 'B', 'S']
    problem_data_values = [A, B, S]
    problem_data = dict(zip(problem_data_keys, problem_data_values))
    return problem_data
