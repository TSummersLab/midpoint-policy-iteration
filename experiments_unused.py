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
        problem_data = gen_rand_problem_data(n=8, m=4, rho=rho, seed=i)
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
    npr.seed(8)
    #
    # # Pathological example
    # # n = 10
    # # m = 4
    # # a1 = 1e-3
    # # a2 = 0.05
    # # a3 = 0.0
    # # b1 = 1.0
    # # b2 = 0.01
    # # s1 = 1.0
    # # s2 = 1.0
    #
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
    def tridiag(mainval, supval, subval, n):
        return np.diag(mainval * np.ones(n), 0)+np.diag(supval * np.ones(n-1), 1)+np.diag(subval * np.ones(n-1), -1)

    def tridiagv(mainval, supval, subval):
        return np.diag(mainval, 0)+np.diag(supval, 1)+np.diag(subval, -1)

    # Problem data

    # # Number of springs
    # N = 6
    # # Number of states
    # n = 2*N
    # # Number of inputs
    # m = 3
    # # m = np.copy(N)

    # Number of springs
    N = 40
    # Number of states
    n = 2*N
    # Number of inputs
    m = 5
    # m = np.copy(N)

    # Dynamics
    # k = 0.1*np.ones(N+1)  # Spring constant
    # c = 0.1*np.ones(N+1)  # Damping constant

    k = 0.06 + 0.2*npr.rand(N+1)  # Spring constants
    c = 0.04 + 0.2*npr.rand(N+1)  # Damping constants

    # k = 0.05 - 0.2*npr.rand(N+1)  # Spring constants
    # c = 0.05 + 0.2*npr.rand(N+1)  # Damping constants

    A11 = np.zeros([N, N])
    A12 = np.eye(N)
    A21 = tridiagv(-(k[1:] + k[0:-1]), k[1:-1], k[1:-1])
    A22 = tridiagv(-(c[1:] + c[0:-1]), c[1:-1], c[1:-1])

    Ac = np.block([[A11, A12],
                  [A21, A22]])
    B1 = np.zeros([N, m])
    # B2 = np.eye(N)
    B2 = np.zeros([N, m])
    # Active actuator selection
    # u_active = np.arange(m)  # Apply force to the first m masses
    u_active = np.sort(npr.permutation(np.arange(N))[0:m])  # Apply force to m masses randomly uniformly distributed across the length
    # Actuator strengths
    # b = np.arange(m)+1
    b = 5*npr.rand(m)
    B2[u_active, np.arange(m)] = b
    Bc = np.vstack([B1, B2])
    # Discretize, forward Euler
    dt = 0.000001
    A = np.eye(n) + Ac*dt
    B = dt*Bc

    # Penalty weights
    S = np.zeros([n+m, n+m])
    sx1 = 0.1  # Position penalty
    sx2 = 0.01  # Velocity penalty
    su = 10.0  # Control effort penalty
    for i in range(N):
        S[i, i] = sx1
    for i in range(N, 2*N):
        S[i, i] = sx2
    for j in range(m):
        S[j+n, j+n] = su

    print(A)
    print(B)



    # # Well-behaved random example
    # # n = 6
    # # m = 2
    # n = 10
    # m = 2
    # rho = 1.10
    # problem_data = gen_rand_problem_data(n=n, m=m, rho=rho, seed=1)
    # problem_data['rho'] = rho
    # problem_data_keys = ['A', 'B', 'S']
    # A, B, S = [problem_data[key] for key in problem_data_keys]
    # n, m = [M.shape[1] for M in [A, B]]


    # # Single inertial mass with force control
    # n = 2
    # m = 1
    # Ts = 0.01
    # mass = 1.0
    # damp = 0.0
    # A = np.array([[1, Ts],
    #               [0, 1-damp*Ts]])
    # B = np.array([[0], [Ts/mass]])
    # S = np.eye(n+m)


    # Form problem
    problem_data = {'A': A, 'B': B, 'S': S}
    n, m = [M.shape[1] for M in [A, B]]
    K0 = get_initial_gains(problem_data, initial_gain_method='dare_perturb', frac_tgt=1e100, bisection_epsilon=1e-15)
    # K0 = get_initial_gains(problem_data, initial_gain_method='dare_perturb', frac_tgt=10, bisection_epsilon=1e-6)
    # K0 = get_initial_gains(problem_data, initial_gain_method='zero')

    # # Basic system quantities
    # # Open-loop eigenvalues / stability
    # print(la.eig(A)[0])
    #
    # print(A)
    # # plt.figure()
    # # plt.imshow(A)
    # # Show the continuous-time matrix for spring-mass-damper system
    # # plt.figure()
    # # plt.spy(Ac, marker='s', markersize=5)
    # # plt.axvline(N-0.5, linestyle='--', color='tab:grey')
    # # plt.axhline(N-0.5, linestyle='--', color='tab:grey')
    #
    # print(B)
    # # plt.figure()
    # # plt.imshow(A)
    # # Show the continuous-time matrix for spring-mass-damper system
    # # plt.figure()
    # # plt.spy(Bc, marker='s', markersize=5)
    # # plt.axhline(N-0.5, linestyle='--', color='tab:grey')
    #
    # # System theoretic quantities
    # from utility.lti import ctrb, obsv, dgram_ctrb, dgram_obsv
    # C = ctrb(A, B)
    # print('controllability rank = %d of %d' % (la.matrix_rank(C), n))
    # Q = S[0:n, 0:n]
    # E = obsv(A, Q)
    # print('observability rank = %d of %d' % (la.matrix_rank(E), n))
    #
    # # Check Gramians (well posed only if A is Schur stable)
    # if specrad(A) < 1:
    #     D = dgram_ctrb(A, B)  # Controllability Gramian
    #     print(la.svd(D)[1])
    #     F = dgram_obsv(A, Q)  # Observability Gramian with respect to penalty
    #     print(la.svd(F)[1])
    # else:
    #     print('skipping ctrb, obsv Gramians since specrad(A) >= 1')
    #
    # AK0 = A + np.dot(B, K0)
    # if not specrad(AK0) < 1:
    #     raise ValueError('Initial gain is not stabilizing!')
    #






    # Baseline solution
    Q = S[0:n, 0:n]
    R = S[n:n+m, n:n+m]
    V = S[0:n, n:n+m]
    Pare, Kare = dare_gain(A, B, Q, R, E=None, S=V)

    num_iterations = 30

    # Simulation options
    # Std deviation for initial state, control inputs, and additive noise
    # xstd, ustd, wstd = 1.0, 1.0, Ts*1e-2
    xstd, ustd, wstd = 1.0, 1.0, 1e-2

    # Uncomment this if using the unbiased LSTDQ estimator which requires knowledge of the process noise covariance
    problem_data['W'] = wstd*np.eye(n)

    # Rollout length
    # nt = 300
    # nt = 4000  # Should be > (n+m)*(n+m+1)/2
    nt = 100000000
    nt_min = int((n+m)*(n+m+1)/2)
    if nt < nt_min:
        raise ValueError('Rollout length too short! Is %d, should be > %d' % (nt, nt_min))

    # Number of rollouts
    nr = 1



    # Q-function estimation scheme
    qfun_estimator = 'lsadp'
    # qfun_estimator = 'lstdq'  # TODO for large n, m the estimates do not converge to the true values as the noise goes to zero. Seems to be a sensitivity of the least-squares solution in LSTDQ to small noise.

    sim_options_keys = ['xstd', 'ustd', 'wstd', 'nt', 'nr', 'qfun_estimator']
    sim_options_values = [xstd, ustd, wstd, nt, nr, qfun_estimator]
    sim_options = dict(zip(sim_options_keys, sim_options_values))

    # Common offline training data
    # offline_training_data = rollout(problem_data, K0, sim_options)
    offline_training_data = None

    # # Plot an open-loop trajectory
    # T = 100000
    # x_hist = np.zeros([T+1, n])
    # x_hist[0] = np.ones(n)
    # for t in range(T):
    #     x_hist[t+1] = A.dot(x_hist[t])
    # print(specrad(A))
    # plt.plot(x_hist[:, 0])

    # Model-based exact
    settings_dict = {'Exact policy iteration':
                         {'problem_data_known': True,
                          'solver': 'policy_iteration',
                          'use_half_data': False},
                     'Exact midpoint policy iteration':
                         {'problem_data_known': True,
                          'solver': 'midpoint_policy_iteration',
                          'use_half_data': False}}

    # Model-free

    # # Situation 1: Offline, midpoint PI gets precisely the same data as PI
    # offline_training_data = rollout(problem_data, K0, sim_options)
    # settings_dict = {'Approximate policy iteration':
    #                      {'problem_data_known': False,
    #                       'solver': 'policy_iteration',
    #                       'share_data_KL': False,
    #                       'use_half_data': False},
    #                  'Approximate midpoint policy iteration':
    #                      {'problem_data_known': False,
    #                       'solver': 'midpoint_policy_iteration',
    #                       'share_data_KL': False,
    #                       'use_half_data': False}}

    # # Situation 2: Online, midpoint PI gets double the amount of data at each iteration as PI
    # offline_training_data = None
    # settings_dict = {'Approximate policy iteration':
    #                      {'problem_data_known': False,
    #                       'solver': 'policy_iteration',
    #                       'share_data_KL': False,
    #                       'use_half_data': False},
    #                  'Approximate midpoint policy iteration':
    #                      {'problem_data_known': False,
    #                       'solver': 'midpoint_policy_iteration',
    #                       'share_data_KL': False,
    #                       'use_half_data': False}}

    # # Situation 3: Online, midpoint PI gets the same amount of data at each iteration as PI,
    # # using entire budget on K and sharing with L gain
    # offline_training_data = None
    # settings_dict = {'Approximate policy iteration':
    #                      {'problem_data_known': False,
    #                       'solver': 'policy_iteration',
    #                       'share_data_KL': True,
    #                       'use_half_data': False},
    #                  'Approximate midpoint policy iteration':
    #                      {'problem_data_known': False,
    #                       'solver': 'midpoint_policy_iteration',
    #                       'share_data_KL': True,
    #                       'use_half_data': False}}

    # Situation 4: Online, midpoint PI gets the same amount of data at each iteration as PI,
    # using half of budget for K and half of budget for L
    # offline_training_data = None
    # settings_dict = {'Approximate policy iteration':
    #                      {'problem_data_known': False,
    #                       'solver': 'policy_iteration',
    #                       'share_data_KL': False,
    #                       'use_half_data': False},
    #                  'Approximate midpoint policy iteration':
    #                      {'problem_data_known': False,
    #                       'solver': 'midpoint_policy_iteration',
    #                       'share_data_KL': False,
    #                       'use_half_data': True}}

    methods = list(settings_dict.keys())
    results_dict = {}
    for method in methods:
        settings = settings_dict[method]
        results_dict[method] = policy_iteration(problem_data, K0=K0,
                                                sim_options=sim_options,
                                                num_iterations=num_iterations,
                                                offline_training_data=offline_training_data,
                                                use_increasing_rollout_length=False,
                                                print_iterates=True,
                                                **settings)
        # verify_are(problem_data, results_dict[method]['P'], algo_str=method)


    plot_example(results_dict, num_iterations, Pare, show_value_matrix=True, show_error=True)

    # # Plot a sample trajectory
    # K = results_dict[list(results_dict.keys())[0]]['K']
    # x0 = npr.randn(n)
    # T = 40000
    # x_hist_ol = np.zeros([T+1, n])
    # x_hist_cl = np.zeros([T+1, n])
    # x_hist_ol[0] = np.copy(x0)
    # x_hist_cl[0] = np.copy(x0)
    # for i in range(T):
    #     w = wstd*npr.randn(n)
    #     x_hist_ol[i+1] = A @ x_hist_ol[i] + w
    #     x_hist_cl[i+1] = (A+B @ K) @ x_hist_cl[i] + w
    #
    # fig, ax = plt.subplots(nrows=2, sharex=True)
    # ax[0].plot(x_hist_ol[:, 0:N])
    # ax[1].plot(x_hist_ol[:, N:])
    # ax[0].set_ylabel('Position')
    # ax[1].set_ylabel('Velocity')
    # ax[-1].set_xlabel('Time')
    # ax[0].set_title('Open loop')
    # fig.tight_layout()
    #
    # fig, ax = plt.subplots(nrows=2, sharex=True)
    # ax[0].plot(x_hist_cl[:, 0:N])
    # ax[1].plot(x_hist_cl[:, N:])
    # ax[0].set_ylabel('Position')
    # ax[1].set_ylabel('Velocity')
    # ax[-1].set_xlabel('Time')
    # ax[0].set_title('Closed loop')
    # fig.tight_layout()

    return results_dict, settings_dict, problem_data, num_iterations, Pare


def example_simple(seed=None):
    npr.seed(seed)

    # Well-behaved random example
    n = 3
    m = 2
    rho = 1.10
    problem_data = gen_rand_problem_data(n=n, m=m, rho=rho, seed=seed)
    problem_data_keys = ['A', 'B', 'S']
    A, B, S = [problem_data[key] for key in problem_data_keys]

    K0 = get_initial_gains(problem_data, initial_gain_method='dare_perturb', frac_tgt=1e3, bisection_epsilon=1e-3)

    # Model-based exact
    # settings_dict = {'Exact policy iteration':
    #                      {'solver': 'policy_iteration'},
    #                  'Exact midpoint policy iteration':
    #                      {'solver': 'midpoint_policy_iteration'}}

    settings_dict = {'Exact midpoint policy iteration':
                         {'solver': 'midpoint_policy_iteration'}}

    methods = list(settings_dict.keys())

    results_dict = {}
    for method in methods:
        settings = settings_dict[method]
        results_dict[method] = policy_iteration(problem_data,
                                                K0=K0,
                                                num_iterations=20,
                                                use_increasing_rollout_length=False,
                                                print_iterates=True,
                                                known_solve_method='direct',
                                                problem_data_known=True,
                                                **settings)
        verify_are(problem_data, results_dict[method]['P'], algo_str=method)


# This is a counterexample to the proposition that the midpoint penalty matrices QL are positive definite
# Indeed, QL can become indefinite for some transient number of iterations
# It seems that eventually QL should become positive definite as # iterations -> inf
def example_counterexample_QL():
    seed = 853860
    npr.seed(seed)

    A = np.array([[-0.26,  1.34, -0.85],
                  [-0.80, 1.14, 0.22],
                  [-0.22, 0.40, -1.00]])

    B = np.array([[0.12, 0.07],
                  [0.73, 0.13],
                  [0.02, 0.58]])

    S = np.diag([1, 1, 1, 1, 1])

    problem_data = {'A': A, 'B': B, 'S': S}

    K0 = np.array([[-0.56, -1.25, 0.16],
                   [-0.36, -0.10, 2.48]])

    AK0 = A+B@K0
    if specrad(AK0 > 1):
        raise ValueError('K0 not stabilizing!')
    else:
        print(specrad(AK0))
        print('')


    # Exact solution
    n, m = B.shape
    Q = S[0:n, 0:n]
    R = S[n:n+m, n:n+m]
    V = S[0:n, n:n+m]
    Pare, Kare = dare_gain(A, B, Q, R, E=None, S=V)


    # Model-based exact
    settings_dict = {'Exact policy iteration':
                         {'solver': 'policy_iteration'},
                     'Exact midpoint policy iteration':
                         {'solver': 'midpoint_policy_iteration'}}

    methods = list(settings_dict.keys())

    results_dict = {}
    for method in methods:
        settings = settings_dict[method]
        results_dict[method] = policy_iteration(problem_data,
                                                K0=K0,
                                                num_iterations=8,
                                                use_increasing_rollout_length=False,
                                                known_solve_method='direct',
                                                problem_data_known=True,
                                                print_iterates=True,
                                                print_diagnostic=True,
                                                **settings)
        # verify_are(problem_data, results_dict[method]['P'], algo_str=method)

    return


if __name__ == '__main__':
    plt.close('all')

    # problem_data_worst, results_dict_worst, worst_iters, Pare_worst, num_iterations = find_hard_problem()

    # example_simple()

    results_dict, settings_dict, problem_data, num_iterations, Pare = example_handpicked()

    # example_counterexample_QL()
