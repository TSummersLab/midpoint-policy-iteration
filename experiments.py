import numpy as np
import numpy.linalg as la
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import multiprocessing as mp

from utility.matrixmath import dare_gain, specrad
from problem_data_gen import gen_rand_problem_data
from midpoint_policy_iteration import get_initial_gains, policy_iteration, verify_are, rollout


def set_numpy_decimal_places(places, width=0):
    set_np = '{0:'+str(width)+'.'+str(places)+'f}'
    np.set_printoptions(formatter={'float': lambda x: set_np.format(x)})


def example_base(seed=None, rho_lim=(0.0, 2.0), methods=None, mc_idx=0):
    # Model-based and model-free solution to the linear-quadratic regulator using midpoint policy iteration
    npr.seed(seed)

    rho = rho_lim[0] + (rho_lim[1]-rho_lim[0])*npr.rand()

    problem_data = gen_rand_problem_data(n=4, m=2, rho=rho, seed=seed)
    problem_data['rho'] = rho
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
    # Std deviation for initial state, control inputs, and additive noise
    xstd, ustd, wstd = 1.0, 1.0, 1e-6

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

    # Common offline training data
    offline_training_data = rollout(problem_data, K0, sim_options)
    # offline_training_data = None

    # Number of policy iterations
    num_iterations = 8

    # Choose the methods to use
    if methods == 'all':
        methods = ['Exact policy iteration',
                   'Exact midpoint policy iteration',
                   'Approximate policy iteration (offline)',
                   'Approximate midpoint policy iteration (offline)',
                   'Approximate policy iteration (online)',
                   'Approximate midpoint policy iteration (online)']
    settings_dict = {}

    for method in methods:
        if method == 'Exact policy iteration':
            setting_dict = {'problem_data_known': True,
                            'offline_training_data': None,
                            'solver': 'policy_iteration',
                            'use_half_data': False}
        elif method == 'Exact midpoint policy iteration':
            setting_dict = {'problem_data_known': True,
                            'offline_training_data': None,
                            'solver': 'midpoint_policy_iteration',
                            'use_half_data': False}
        elif method == 'Approximate policy iteration (offline)':
            setting_dict = {'problem_data_known': False,
                            'offline_training_data': offline_training_data,
                            'solver': 'policy_iteration',
                            'use_half_data': False}
        elif method == 'Approximate midpoint policy iteration (offline)':
            setting_dict = {'problem_data_known': False,
                            'offline_training_data': offline_training_data,
                            'solver': 'midpoint_policy_iteration',
                            'use_half_data': False}
        elif method == 'Approximate policy iteration (online)':
            setting_dict = {'problem_data_known': False,
                            'offline_training_data': None,
                            'solver': 'policy_iteration',
                            'use_half_data': False}
        elif method == 'Approximate midpoint policy iteration (online)':
            setting_dict = {'problem_data_known': False,
                            'offline_training_data': None,
                            'solver': 'midpoint_policy_iteration',
                            'use_half_data': True}
        else:
            raise ValueError
        settings_dict[method] = setting_dict

    results_dict = {}
    for method in methods:
        settings = settings_dict[method]
        results_dict[method] = policy_iteration(problem_data, K0=K0,
                                                sim_options=sim_options,
                                                num_iterations=num_iterations,
                                                use_increasing_rollout_length=False,
                                                **settings)
        # verify_are(problem_data, results_dict[method]['P'], algo_str=method)
    return results_dict, settings_dict, problem_data, num_iterations, Pare, mc_idx


def method_abbrev(method):
    if method == 'Exact policy iteration':
        return 'Exact PI'
    elif method == 'Exact midpoint policy iteration':
        return 'Exact MPI'
    elif method == 'Approximate policy iteration':
        return 'Approx. PI'
    elif method == 'Approximate midpoint policy iteration':
        return 'Approx. MPI'


def example(seed=None):
    results_dict, settings_dict, problem_data, num_iterations, Pare, mc_idx = example_base(seed)
    plot_example(results_dict, num_iterations, Pare, show_value_matrix=False)
    return


def example_many_systems(num_systems=10, rho_lim=(0.0, 2.0), methods=None, seed_offset=0, parallel_option='parallel'):
    # results_list_all = []
    # for i in range(num_systems):
    #     seed = i + seed_offset
    #     results_list_all.append(example_base(seed, rho_lim, methods))
    #     print('Problem %4d / %4d solved' % (i+1, num_systems))

    results_list_all = []

    def collect_result(result):
        results_list_all.append(result)
        mc_idx = result[-1]
        print("Completed Monte Carlo sample %6d / %6d" % (mc_idx+1, num_systems))

    # Serial single-threaded processing
    if parallel_option == 'serial':
        for mc_idx in range(num_systems):
            seed = mc_idx + seed_offset
            result = example_base(seed, rho_lim, methods, mc_idx)
            collect_result(result)

    # Asynchronous parallel CPU processing
    elif parallel_option == 'parallel':
        num_cpus_to_use = mp.cpu_count() - 1
        pool = mp.Pool(num_cpus_to_use)
        for mc_idx in range(num_systems):
            seed = mc_idx+seed_offset
            pool.apply_async(example_base,
                             args=(seed, rho_lim, methods, mc_idx),
                             callback=collect_result)
        pool.close()
        pool.join()

    return results_list_all


def plot_example(results_dict, num_iterations, Pare, show_value_matrix=True, show_error=True):
    methods = list(results_dict.keys())
    num_methods = len(methods)
    t_history = np.arange(num_iterations)+1
    fig_ax_list = []

    if show_value_matrix:
        fig1, ax1 = plt.subplots(ncols=num_methods)
        fig1.suptitle('Value matrix (P)')
        for i, method in enumerate(methods):
            # Cost-to-go matrix
            P = results_dict[method]['P']
            ax1[i].imshow(P)
            ax1[i].set_title(method_abbrev(method))
        fig_ax_list.append((fig1, ax1))

    if show_error:
        fig2, ax2 = plt.subplots()
        for i, method in enumerate(methods):
            P_history = results_dict[method]['P_history']
            x_data = t_history
            y_data = np.array([la.norm(P_history[t]-Pare, ord=2)/la.norm(Pare, ord=2) for t in range(num_iterations)])
            ax2.plot(x_data, y_data, label=method_abbrev(method))
        ax2.legend()
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Relative error')
        ax2.set_yscale('log')
        if num_iterations <= 20:
            ax2.set_xticks(np.arange(num_iterations)+1)
        fig_ax_list.append((fig2, ax2))

    return fig_ax_list


def plot_example_many_systems(results_list_all, method_pairs, rho_lim=(0.0, 2.0), rat_lim=None, plot_type='scatter'):
    # Preprocess the data
    num_systems = len(results_list_all)
    rho_dict = {}
    err_dict = {}
    # Get the list of methods and number of iterations from the first results list
    results = results_list_all[0]
    methods = list(results[0].keys())
    num_iterations = results[3]
    for method in methods:
        rho_dict[method] = np.zeros(num_systems)
        err_dict[method] = np.zeros([num_systems, num_iterations])
    for i, results in enumerate(results_list_all):
        results_dict, settings_dict, problem_data, num_iterations, Pare, mc_idx = results
        methods = list(results_dict.keys())
        for method in methods:
            rho_dict[method][i] = problem_data['rho']
            P_history = results_dict[method]['P_history']
            frac_history = np.array([la.norm(P_history[t]-Pare, ord=2)/la.norm(Pare, ord=2) for t in range(num_iterations)])
            err_dict[method][i] = frac_history

    def method_str_shortener(method):
        if method == 'Exact policy iteration':
            return 'exact_pi'
        elif method == 'Exact midpoint policy iteration':
            return 'exact_mpi'
        elif method == 'Approximate policy iteration (offline)':
            return 'approx_pi_offline'
        elif method == 'Approximate midpoint policy iteration (offline)':
            return 'approx_mpi_offline'
        elif method == 'Approximate policy iteration (online)':
            return 'approx_pi_online'
        elif method == 'Approximate midpoint policy iteration (online)':
            return 'approx_mpi_online'

    # Do the plotting
    nrows = 2
    ncols = 4

    if plot_type == 'scatter':
        scatter_alpha = min(10/num_systems**0.5, 1.0)
    elif plot_type == 'hexbin':
        gridsize = 20

    def relative_error_plot(xdata, ydata1_all, ydata2_all, method1=None, method2=None, plot_type='scatter', pct_lwr=1, pct_upr=99):
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.5*ncols, 2.5*nrows))
        ax_idxs = []
        for i in range(nrows):
            for j in range(ncols):
                ax_idxs.append((i, j))

        for k in range(nrows*ncols):
            ax_idx = ax_idxs[k]
            ydata1 = ydata1_all[:, k]
            ydata2 = ydata2_all[:, k]
            ydata = (ydata2 / ydata1)

            # Axes limits
            xlim = rho_lim
            if rat_lim is None:
                # Capture at least 98% of observations, snap outwards to next largest/smallest power of 10, and make symmetric
                yhardmin = 0.01
                yhardmax = 100.0
                ylogmin = min(np.log10(np.percentile(ydata, pct_lwr)), np.log10(yhardmin))
                ylogmax = max(np.log10(np.percentile(ydata, pct_upr)), np.log10(yhardmax))
                ylogabsmax = math.ceil(max(abs(ylogmin), abs(ylogmax)))
                ylim = (10**(-ylogabsmax*1.05), 10**(ylogabsmax*1.05))
            else:
                ylim = rat_lim

            # Colors
            if plot_type == 'scatter':
                color_good = 'C0'
                color_bad = 'C1'
                cdata = []
                for i in range(num_systems):
                    if ydata[i] > 1.1:
                        c = color_bad
                    elif ydata[i] < 0.9:
                        c = color_good
                    else:
                        c = 'k'
                    cdata.append(c)
            elif plot_type == 'hexbin':
                cmap = 'Blues'
            else:
                raise ValueError

            if plot_type == 'scatter':
                ax[ax_idx].scatter(xdata, ydata, c=cdata, edgecolors='none', alpha=scatter_alpha)
                ax[ax_idx].axhline(y=1, color='k', linestyle='--', alpha=0.5)

                rect_upr = Rectangle((xlim[0], 1.0), xlim[1]-xlim[0], ylim[1]-1.0, linewidth=1, edgecolor='none', facecolor=color_bad, alpha=0.2)
                rect_lwr = Rectangle((xlim[0], ylim[0]), xlim[1]-xlim[0], 1.0-ylim[0], linewidth=1, edgecolor='none', facecolor=color_good, alpha=0.2)
                ax[ax_idx].add_patch(rect_upr)
                ax[ax_idx].add_patch(rect_lwr)
            elif plot_type == 'hexbin':
                ax[ax_idx].hexbin(xdata, ydata, yscale='log', bins='log', gridsize=gridsize, cmap=cmap,
                                  extent=(xlim[0], xlim[1], np.log10(ylim[0]), np.log10(ylim[1])))

            ax[ax_idx].set_yscale('log')
            ax[ax_idx].set_xlim(xlim)
            ax[ax_idx].set_ylim(ylim)
            if rat_lim is None:
                ax[ax_idx].set_yticks([10**(-ylogabsmax), 1.0, 10**(ylogabsmax)])
            ax[ax_idx].set_xlabel(r'$\rho(A)$')
            ax[ax_idx].set_title('k = %d' % (k))

        for i in range(nrows):
            ax[i, 0].set_ylabel('Error ratio')
        fig.tight_layout()
        delimiter = '_'
        method1s = method_str_shortener(method1)
        method2s = method_str_shortener(method2)
        filename = delimiter.join(['many_systems_relative_error', method1s, method2s, plot_type]) + '.pdf'
        plt.savefig('./figures/'+filename, dpi=600, format='pdf', bbox_inches='tight')
        return fig, ax

    def absolute_error_plot(xdata, ydata_all, method=None, plot_type='scatter'):
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.5 * ncols, 2.5 * nrows))
        ax_idxs = []
        for i in range(nrows):
            for j in range(ncols):
                ax_idxs.append((i, j))

        for k in range(nrows * ncols):
            ax_idx = ax_idxs[k]
            ydata = ydata_all[:, k]

            xlim = rho_lim
            ylim = [np.min(ydata)*0.95, np.max(ydata)*1.05]
            if k == 0:
                ylim = [1.0, 100.0]

            if plot_type == 'scatter':
                c = 'k'
            elif plot_type == 'hexbin':
                cmap = 'gray_r'
            else:
                raise ValueError

            if plot_type == 'scatter':
                ax[ax_idx].scatter(xdata, ydata, c=c, edgecolors='none', alpha=scatter_alpha)
            elif plot_type == 'hexbin':
                ax[ax_idx].hexbin(xdata, ydata, yscale='log', bins='log', gridsize=gridsize, cmap=cmap,
                                  extent=(xlim[0], xlim[1], np.log10(ylim[0]), np.log10(ylim[1])))

            ax[ax_idx].set_yscale('log')
            ax[ax_idx].set_xlim(xlim)
            ax[ax_idx].set_ylim(ylim)
            if k == 0:
                ax[ax_idx].set_yticks([1.0, 10.0, 100.0])
            ax[ax_idx].set_xlabel(r'$\rho(A)$')
            ax[ax_idx].set_title('k = %d' % k)

        for i in range(nrows):
            ax[i, 0].set_ylabel('Error')
        fig.tight_layout()
        delimiter = '_'
        method1s = method_str_shortener(method)
        filename = delimiter.join(['many_systems_absolute_error', method1s, plot_type]) + '.pdf'
        plt.savefig('./figures/'+filename, dpi=600, format='pdf', bbox_inches='tight')
        return fig, ax

    for (method1, method2) in method_pairs:
        xdata = rho_dict[method1]
        ydata1_all = err_dict[method1]
        ydata2_all = err_dict[method2]

        relative_error_plot(xdata, ydata1_all, ydata2_all, method1, method2, plot_type)
        absolute_error_plot(xdata, ydata2_all, method2, plot_type)
    return


def plot_example_many_systems_sample(results_list_all, show_num_systems=4):
    for i, results in enumerate(results_list_all):
        if i < show_num_systems:
            results_dict, settings_dict, problem_data, num_iterations, Pare = results
            plot_example(results_dict, num_iterations, Pare, show_value_matrix=False)
        else:
            break
    return


def example_inertial_mass(seed=1):
    # Single inertial mass with force control
    npr.seed(seed)
    n = 2
    m = 1
    Ts = 0.01
    mass = 1.0
    damp = 0.0
    A = np.array([[1, Ts],
                  [0, 1-damp*Ts]])
    B = np.array([[0], [Ts/mass]])
    S = np.eye(n+m)
    problem_data = {'A': A, 'B': B, 'S': S}

    # Baseline solution
    Q = S[0:n, 0:n]
    R = S[n:n+m, n:n+m]
    V = S[0:n, n:n+m]
    Pare, Kare = dare_gain(A, B, Q, R, E=None, S=V)

    num_iterations = 12

    # Simulation options
    # Std deviation for initial state, control inputs, and additive noise
    xstd, ustd, wstd = 1.0, 1.0, Ts*1e-2

    # Rollout length
    nt = int(50*(n+m)*(n+m+1)/2)  # Should be > (n+m)*(n+m+1)/2
    # Number of rollouts
    nr = 1

    # Q-function estimation scheme
    qfun_estimator = 'lstdq'

    sim_options_keys = ['xstd', 'ustd', 'wstd', 'nt', 'nr', 'qfun_estimator']
    sim_options_values = [xstd, ustd, wstd, nt, nr, qfun_estimator]
    sim_options = dict(zip(sim_options_keys, sim_options_values))

    # Get initial gains
    K0 = get_initial_gains(problem_data, initial_gain_method='dare_perturb', frac_tgt=10)
    print(K0)

    # Common offline training data
    offline_training_data = rollout(problem_data, K0, sim_options)

    settings_dict = {'Exact policy iteration':
                         {'problem_data_known': True,
                          'offline_training_data': offline_training_data,
                          'solver': 'policy_iteration',
                          'use_half_data': False},
                     'Exact midpoint policy iteration':
                         {'problem_data_known': True,
                          'offline_training_data': offline_training_data,
                          'solver': 'midpoint_policy_iteration',
                          'use_half_data': False},
                     'Approximate policy iteration':
                         {'problem_data_known': False,
                          'offline_training_data': offline_training_data,
                          'solver': 'policy_iteration',
                          'use_half_data': False},
                     'Approximate midpoint policy iteration':
                         {'problem_data_known': False,
                          'offline_training_data': offline_training_data,
                          'solver': 'midpoint_policy_iteration',
                          'use_half_data': False}}

    methods = list(settings_dict.keys())

    results_dict = {}
    for k, method in enumerate(methods):
        settings = settings_dict[method]
        results_dict[method] = policy_iteration(problem_data, K0=K0,
                                                sim_options=sim_options,
                                                num_iterations=num_iterations,
                                                use_increasing_rollout_length=False,
                                                **settings)

        # # Plot the iteration paths in gain-space
        # K_hist = results_dict[method]['K_history']
        # ax.plot(K_hist[:, 0, 0], K_hist[:, 0, 1], color='C%d'%k, alpha=0.8)
        # ax.scatter(K_hist[:, 0, 0], K_hist[:, 0, 1], color='C%d' % k, alpha=0.8)
    return results_dict, num_iterations, Pare


def plot_example_inertial_mass(results_dict, num_iterations, Pare):
    methods = list(results_dict.keys())
    colors = ['C1', 'C0', 'C1', 'C0']
    linestyles = ['-', '-', ':', ':']
    markers = ['x', 'o', '|', '^']
    markersizes = [16, 8, 20, 8]
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for i, (method, color, linestyle, marker, markersize) in enumerate(
            zip(methods, colors, linestyles, markers, markersizes)):
        P_history = results_dict[method]['P_history']
        x_data = np.arange(num_iterations)+1
        y_data = np.array([la.norm(P_history[t]-Pare, ord=2) / la.norm(Pare, ord=2) for t in range(num_iterations)])
        ax.plot(x_data, y_data, color=color, linestyle=linestyle, linewidth=3,
                marker=marker, markersize=markersize, markeredgewidth=3,
                label=method_abbrev(method))
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative error')
    # ax.set_ylabel(r'$\frac{P_k - P^*}{{tr}(P^*)}$', rotation=0)
    ax.set_yscale('log')
    if num_iterations <= 20:
        ax.set_xticks(np.arange(num_iterations)+1)
    fig.tight_layout()
    filename = 'inertial_mass_convergence.pdf'
    plt.savefig('./figures/'+filename, dpi=600, format='pdf', bbox_inches='tight')
    return fig, ax


def experiments_initial_submission():
    # Experiments used in initial submission of paper
    plt.close('all')
    plt.style.use('./utility/conlab.mplstyle')

    # 1
    results_dict, num_iterations, Pare = example_inertial_mass()
    plot_example_inertial_mass(results_dict, num_iterations, Pare)
    plt.show()

    # 2
    rho_lim = (0.0, 2.0)
    results_list_all = example_many_systems(num_systems=1000, rho_lim=rho_lim, methods='all')
    method_pairs = [['Exact policy iteration', 'Exact midpoint policy iteration'],
                    ['Approximate policy iteration (offline)', 'Approximate midpoint policy iteration (offline)'],
                    ['Approximate policy iteration (online)', 'Approximate midpoint policy iteration (online)']]

    plot_example_many_systems(results_list_all, method_pairs=method_pairs, rho_lim=rho_lim, plot_type='scatter')
    plt.show()
    return


def experiments_extended():
    # Experiments used in arXiv extended version of paper
    plt.close('all')
    plt.style.use('./utility/conlab.mplstyle')

    # 1
    results_dict, num_iterations, Pare = example_inertial_mass()
    plot_example_inertial_mass(results_dict, num_iterations, Pare)
    plt.show()

    # 2
    rho_lim = (0.0, 2.0)
    results_list_all = example_many_systems(num_systems=1000, rho_lim=rho_lim, methods='all')
    method_pairs = [['Exact policy iteration', 'Exact midpoint policy iteration'],
                    ['Approximate policy iteration (offline)', 'Approximate midpoint policy iteration (offline)'],
                    ['Approximate policy iteration (online)', 'Approximate midpoint policy iteration (online)']]
    plot_example_many_systems(results_list_all, method_pairs=method_pairs, rho_lim=rho_lim, plot_type='scatter')
    plot_example_many_systems(results_list_all, method_pairs=method_pairs, rho_lim=rho_lim, plot_type='hexbin')
    # plot_example_many_systems_sample(results_list_all, show_num_systems=4)
    plt.show()
    return


if __name__ == "__main__":
    # experiments_initial_submission()
    experiments_extended()
