import numpy as np
import numpy.linalg as la
import numpy.random as npr
import scipy.linalg as sla
from midpoint_policy_iteration import get_initial_gains, policy_iteration, midpoint_policy_iteration, verify_are

import sys
sys.path.insert(0, '../utility')
from matrixmath import mdot, specrad, solveb, dare_gain, is_pos_def, svec2, smat2, kron, vec


def set_numpy_decimal_places(places, width=0):
    set_np = '{0:'+str(width)+'.'+str(places)+'f}'
    np.set_printoptions(formatter={'float': lambda x: set_np.format(x)})




if __name__ == "__main__":
    # Model-based or model-free solution to the linear-quadratic regulator using midpoint policy iteration
    import matplotlib.pyplot as plt

    seed = 10
    npr.seed(seed)

    problem_data = gen_rand_problem_data(n=3, m=1, rho=1.1, seed=seed)

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

    # Simulation options
    # Std deviation for initial state, defender inputs, attacker inputs, and additive noise
    xstd, ustd, wstd = 1.0, 1.0, 1e-3
    # npr.seed(890)

    # Rollout length
    nt = 100

    # Number of rollouts
    nr = 1

    # Q-function estimation scheme
    # qfun_estimator = 'lsadp'
    qfun_estimator = 'lstdq'

    sim_options_keys = ['xstd', 'ustd', 'wstd', 'nt', 'nr', 'qfun_estimator']
    sim_options_values = [xstd, ustd, wstd, nt, nr, qfun_estimator]
    sim_options = dict(zip(sim_options_keys, sim_options_values))

    num_iterations = 20

    methods = ['Exact policy iteration',
               'Approximate policy iteration',
               'Exact accelerated policy iteration',
               'Approximate accelerated policy iteration']
    # methods = ['Exact policy iteration',
    #            'Approximate policy iteration']
    num_methods = len(methods)
    settings_dict = {'Exact policy iteration':
                         {'problem_data_known': True,
                          'solver': 'policy_iteration'},
                     'Approximate policy iteration':
                         {'problem_data_known': False,
                          'solver': 'policy_iteration'},
                     'Exact accelerated policy iteration':
                         {'problem_data_known': True,
                          'solver': 'accelerated_policy_iteration'},
                     'Approximate accelerated policy iteration':
                         {'problem_data_known': False,
                          'solver': 'accelerated_policy_iteration'}}
    results_dict = {}
    for method in methods:
        problem_data_known = settings_dict[method]['problem_data_known']
        if settings_dict[method]['solver'] == 'policy_iteration':
            solver_fun = policy_iteration
        elif settings_dict[method]['solver'] == 'accelerated_policy_iteration':
            solver_fun = midpoint_policy_iteration
        else:
            raise ValueError('Invalid solver function chosen.')
        results_dict[method] = solver_fun(problem_data, problem_data_known, K0, sim_options, num_iterations)
        verify_are(problem_data, results_dict[method]['P'], algo_str=method)

    # Plotting
    plt.close('all')
    t_history = np.arange(num_iterations)+1
    # Cost-to-go matrix
    fig1, ax1 = plt.subplots(ncols=num_methods)
    plt.suptitle('Value matrix (P)')

    # Cost over iterations
    fig2, ax2 = plt.subplots()

    for i, method in enumerate(methods):
        P = results_dict[method]['P']
        P_history = results_dict[method]['P_history']
        ax1[i].imshow(P)
        ax1[i].set_title(method)

        ax2.plot(t_history, np.array([la.norm(P_history[t]-Pare, ord=2) for t in range(num_iterations)]))

    fig2.legend(methods)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Error')
    ax2.set_yscale('log')
    if num_iterations <= 20:
        plt.xticks(np.arange(num_iterations)+1)

    plt.show()
