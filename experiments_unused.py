import numpy as np
import numpy.linalg as la
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
from utility.matrixmath import dare_gain, specrad
from problem_data_gen import gen_rand_problem_data
from midpoint_policy_iteration import get_initial_gains, policy_iteration, verify_are, rollout
from experiments import plot_example


def find_hard_problem(num_systems=1000, convergence_threshold=1e-6):
    worst_iters = 0

    for i in range(num_systems):
        npr.seed(i)
        rho = 2*npr.rand()
        problem_data = gen_rand_problem_data(n=3, m=1, rho=rho, seed=i)
        problem_data_keys = ['A', 'B', 'S']
        A, B, S = [problem_data[key] for key in problem_data_keys]
        n, m = [M.shape[1] for M in [A, B]]

        # Initial gains
        K0 = get_initial_gains(problem_data, initial_gain_method='dare_perturb')

        # Baseline solution
        Q = S[0:n, 0:n]
        R = S[n:n+m, n:n+m]
        V = S[0:n, n:n+m]
        Pare, Kare = dare_gain(A, B, Q, R, E=None, S=V)

        num_iterations = 20
        results_dict = policy_iteration(problem_data, problem_data_known=True,
                                        K0=K0, num_iterations=num_iterations, solver='policy_iteration')

        P_history = results_dict['P_history']
        frac_history = np.array([la.norm(P_history[t]-Pare, ord=2) / np.trace(Pare) for t in range(num_iterations)])

        iters = np.argmin(frac_history > convergence_threshold)

        if iters > worst_iters:
            worst_iters = iters
            problem_data_worst = problem_data
            results_dict_worst = results_dict
            Pare_worst = Pare
        print('Problem %d / %d solved' % (i+1, num_systems))

    return problem_data_worst, results_dict_worst, worst_iters, Pare_worst, num_iterations


def example_handpicked():
    npr.seed(1)

    # Pathological example
    # n = 10
    # m = 4
    # a1 = 1e-3
    # a2 = 0.05
    # a3 = 0.0
    # b1 = 1.0
    # b2 = 0.01
    # s1 = 1.0
    # s2 = 1.0

    # n = 8
    # m = 4
    # a1 = 0.1
    # a2 = 0.02
    # a3 = 0.0
    # b1 = 1.0
    # b2 = 0.5
    # s1 = 1.0
    # s2 = 1.0
    #
    # A = np.zeros([n, n])
    # B = np.zeros([n, m])
    # S = np.zeros([n+m, n+m])
    # for i in range(n):
    #     # A matrix
    #     # A[i, i] = ((-1)**i) * (1-a1)
    #     A[i, i] = (1-a1)
    #     if i < n-1:
    #         A[i, i+1] = a2
    #     if i > 0:
    #         A[i, i-1] = a3
    #     # B matrix
    #     for j in range(m):
    #         if i == j:
    #             B[i, j] = b1
    #         else:
    #             B[i, j] = b2
    #             # B[i, j] = (i+j) * b2
    #     # S matrix
    #     S[i, i] = s1
    # for j in range(m):
    #     S[j+n, j+n] = s2

    # Spring-mass serial chain
    # def tridiag(mainval, supval, subval, n):
    #     return np.diag(mainval * np.ones(n), 0)+np.diag(supval * np.ones(n-1), 1)+np.diag(subval * np.ones(n-1), -1)
    #
    # # Problem data
    # # Number of springs
    # N = 8
    # # Number of states
    # n = 2*N
    # # Number of inputs
    # m = 8
    # # m = np.copy(N)
    # # Time horizon
    # T = 100
    # # Initial state
    # x0 = np.zeros(n)
    # # Dynamics
    # k = 0.01
    # c = 0.001
    # A11 = np.zeros([N, N])
    # A12 = np.eye(N)
    # A21 = tridiag(-2*k, k, k, N)
    # A22 = tridiag(-2*c, c, c, N)
    # A = np.block([[A11, A12],
    #               [A21, A22]])
    # B1 = np.zeros([N, N])
    # # B2 = np.eye(N)
    # B2 = np.zeros([N, N])
    # B2[0:m, 0:m] = np.diag(np.arange(m)+1)
    # B = np.vstack([B1, B2])
    #
    # S = np.zeros([n+m, n+m])
    # s1 = 1.0
    # s2 = 1.0
    # for i in range(n):
    #     # S matrix
    #     S[i, i] = s1
    # for j in range(m):
    #     S[j+n, j+n] = s2

    # Well-behaved random example
    n = 6
    m = 2
    rho = 0.99
    problem_data = gen_rand_problem_data(n=n, m=m, rho=rho, seed=1)
    problem_data['rho'] = rho
    problem_data_keys = ['A', 'B', 'S']
    A, B, S = [problem_data[key] for key in problem_data_keys]
    n, m = [M.shape[1] for M in [A, B]]

    # if specrad(A) > 1:
    #     raise ValueError('specrad(A) > 1')

    # System theoretic quantities
    from utility.lti import ctrb, obsv, dgram_ctrb, dgram_obsv
    C = ctrb(A, B)
    print(la.matrix_rank(C))
    Q = S[0:n, 0:n]
    E = obsv(A, Q)
    print(la.matrix_rank(E))

    # Check Gramians if A is Schur stable
    if specrad(A) < 1:
        D = dgram_ctrb(A, B)
        print(la.svd(D)[1])
        F = dgram_obsv(A, Q)
        print(la.svd(F)[1])

    problem_data = {'A': A, 'B': B, 'S': S}
    n, m = [M.shape[1] for M in [A, B]]
    K0 = get_initial_gains(problem_data, initial_gain_method='dare_perturb')
    # K0 = get_initial_gains(problem_data, initial_gain_method='zero')

    # Baseline solution
    Q = S[0:n, 0:n]
    R = S[n:n+m, n:n+m]
    V = S[0:n, n:n+m]
    Pare, Kare = dare_gain(A, B, Q, R, E=None, S=V)

    num_iterations = 15

    # Simulation options
    # Std deviation for initial state, control inputs, and additive noise
    # xstd, ustd, wstd = 1.0, 1.0, 1e-3
    xstd, ustd, wstd = 1.0, 1.0, 1e-5

    # problem_data['W'] = wstd*np.eye(n)

    # Rollout length
    nt = 1000  # Should be > (n+m)*(n+m+1)/2

    # Number of rollouts
    nr = 1

    # Q-function estimation scheme
    # qfun_estimator = 'lsadp'
    qfun_estimator = 'lstdq'

    sim_options_keys = ['xstd', 'ustd', 'wstd', 'nt', 'nr', 'qfun_estimator']
    sim_options_values = [xstd, ustd, wstd, nt, nr, qfun_estimator]
    sim_options = dict(zip(sim_options_keys, sim_options_values))

    # Common offline training data
    offline_training_data = rollout(problem_data, K0, sim_options)
    # offline_training_data = None

    # # Plot an open-loop trajectory
    # T = 100000
    # x_hist = np.zeros([T+1, n])
    # x_hist[0] = np.ones(n)
    # for t in range(T):
    #     x_hist[t+1] = A.dot(x_hist[t])
    # print(specrad(A))
    # plt.plot(x_hist[:, 0])

    # settings_dict = {'Exact policy iteration':
    #                      {'problem_data_known': True,
    #                       'solver': 'policy_iteration',
    #                       'use_half_data': False},
    #                  'Exact midpoint policy iteration':
    #                      {'problem_data_known': True,
    #                       'solver': 'midpoint_policy_iteration',
    #                       'use_half_data': False}}

    settings_dict = {'Approximate policy iteration':
                         {'problem_data_known': False,
                          'solver': 'policy_iteration',
                          'use_half_data': False},
                     'Approximate midpoint policy iteration':
                         {'problem_data_known': False,
                          'solver': 'midpoint_policy_iteration',
                          'use_half_data': False}}

    methods = list(settings_dict.keys())

    results_dict = {}
    for method in methods:
        settings = settings_dict[method]
        results_dict[method] = policy_iteration(problem_data, K0=K0,
                                                sim_options=sim_options,
                                                num_iterations=num_iterations,
                                                offline_training_data=offline_training_data,
                                                use_increasing_rollout_length=False,
                                                **settings)
        # verify_are(problem_data, results_dict[method]['P'], algo_str=method)

    plot_example(results_dict, num_iterations, Pare, show_value_matrix=True, show_error=True)
    return results_dict, settings_dict, problem_data, num_iterations, Pare
