import numpy as np
import numpy.linalg as la
import numpy.random as npr
from copy import copy

from utility.matrixmath import mdot, specrad, dare_gain, is_pos_def, svec2, smat2, vec


def groupdot(A, x):
    """
    Perform dot product over groups of matrices,
    suitable for performing many LTI state transitions in a vectorized fashion
    """
    return np.einsum('...ik,...k', A, x)


def groupquadform(A, x):
    """
    Perform quadratic form over many vectors stacked in matrix x with respect to cost matrix A
    Equivalent to np.array([mdot(x[i].T, A, x[i]) for i in range(x.shape[0])])
    """
    return np.sum(np.dot(x, A)*x, axis=1)


def dlyap_direct(A, Q):
    """
    Solve the discrete-time Lyapunov equation
    P = A.T @ P @ A + Q
    via direct method"""
    vQ = vec(Q)
    A2 = np.kron(A.T, A.T)
    vP = la.solve(np.eye(A.size)-A2, vQ)
    return np.reshape(vP, A.shape)


def rollout(problem_data, K, sim_options):
    """Simulate closed-loop state response"""
    problem_data_keys = ['A', 'B', 'S']
    A, B, S = [problem_data[key] for key in problem_data_keys]
    n, m = [M.shape[1] for M in [A, B]]

    sim_options_keys = ['xstd', 'ustd', 'wstd', 'nt', 'nr']
    xstd, ustd, wstd, nt, nr = [sim_options[key] for key in sim_options_keys]

    # Sample initial states, control inputs
    x0 = xstd * npr.randn(nr, n)
    u_explore_hist = ustd * npr.randn(nr, nt, m)

    # Initialize history data arrays
    x_hist = np.zeros([nr, nt, n])
    u_hist = np.zeros([nr, nt, m])
    c_hist = np.zeros([nr, nt])

    # Randomly sample additive noise
    w_all = npr.randn(nt, nr, n)*wstd

    # Initialize
    x = np.copy(x0)

    # Iterate over timesteps
    for i in range(nt):
        # Compute controls
        u = groupdot(K, x) + u_explore_hist[:, i]

        # Compute cost
        z = np.hstack([x, u])
        c = groupquadform(S, z)

        # Record history
        x_hist[:, i] = x
        u_hist[:, i] = u
        c_hist[:, i] = c

        # Look up additive noise
        w = w_all[i]

        # Transition the state using additive noise
        x = groupdot(A, x) + groupdot(B, u) + w

    return x_hist, u_hist, c_hist


def qfun(problem_data, problem_data_known=None, P=None, K=None, sim_options=None):
    """Compute or estimate Q-function matrix"""
    if problem_data_known is None:
        problem_data_known = True
    problem_data_keys = ['A', 'B', 'S']
    A, B, S = [problem_data[key] for key in problem_data_keys]
    n, m = [M.shape[1] for M in [A, B]]

    if problem_data_known:
        if P is None:
            IK = np.vstack([np.eye(n), K])
            P = dlyap_direct(A+B.dot(K), mdot(IK.T, S, IK))
        AB = np.hstack([A, B])
        Q = S + mdot(AB.T, P, AB)
    else:
        nr = sim_options['nr']
        nt = sim_options['nt']
        qfun_estimator = sim_options['qfun_estimator']

        # Simulation data_files collection
        x_hist, u_hist, c_hist = rollout(problem_data, K, sim_options)

        # Form the data_files matrices
        ns = nr * (nt-1)
        nz = int(((n+m+1) * (n+m)) / 2)
        mu_hist = np.zeros([nr, nt, nz])
        nu_hist = np.zeros([nr, nt, nz])

        def phi(x):
            return svec2(np.outer(x, x))

        for i in range(nr):
            for j in range(nt):
                z = np.concatenate([x_hist[i, j], u_hist[i, j]])
                w = np.concatenate([x_hist[i, j], np.dot(K, x_hist[i, j])])
                mu_hist[i, j] = phi(z)
                nu_hist[i, j] = phi(w)

        if qfun_estimator == 'lsadp':
            Y = np.zeros(ns)
            Z = np.zeros([ns, nz])
            for i in range(nr):
                lwr = i * (nt-1)
                upr = (i+1) * (nt-1)
                Y[lwr:upr] = c_hist[i, 0:-1]
                Z[lwr:upr] = mu_hist[i, 0:-1]-nu_hist[i, 1:]
            # Solve the least squares problem
            Q_svec2 = la.lstsq(Z, Y, rcond=None)[0]
            Q = smat2(Q_svec2)

        elif qfun_estimator == 'lstdq':
            Y = np.zeros(nr*nz)
            Z = np.zeros([nr*nz, nz])
            for i in range(nr):
                lwr = i*nz
                upr = (i+1)*nz
                for j in range(nt-1):
                    Y[lwr:upr] += mu_hist[i, j]*c_hist[i, j]
                    Z[lwr:upr] += np.outer(mu_hist[i, j], mu_hist[i, j]-nu_hist[i, j+1])
            # Solve the least squares problem
            Q_svec2 = la.lstsq(Z, Y, rcond=None)[0]
            Q = smat2(Q_svec2)
        else:
            raise ValueError('Invalid Q-function estimator chosen.')
    return Q


def policy_iteration(problem_data, problem_data_known, K0, sim_options=None, num_iterations=100, print_iterates=False):
    problem_data_keys = ['A', 'B', 'S']
    A, B, S = [problem_data[key] for key in problem_data_keys]
    n, m = [M.shape[1] for M in [A, B]]
    K = np.copy(K0)

    # Check initial policies are stabilizing
    if specrad(A+B.dot(K0)) > 1:
        raise Exception("Initial policy is not stabilizing!")

    P_history, K_history = [np.zeros([num_iterations, dim, n]) for dim in [n, m]]
    H_history = np.zeros([num_iterations, n+m, n+m])
    c_history = np.zeros(num_iterations)

    print('Policy iteration')
    for i in range(num_iterations):
        # Record history
        K_history[i] = K

        # Policy evaluation
        IK = np.vstack([np.eye(n), K])
        P = dlyap_direct(A+B.dot(K), mdot(IK.T, S, IK))
        H = qfun(problem_data, problem_data_known, P, K, sim_options)
        Hxx = H[0:n, 0:n]
        Huu = H[n:n+m, n:n+m]
        Hux = H[n:n+m, 0:n]

        # Policy improvement
        K = -la.solve(Huu, Hux)

        # Record history
        P_history[i] = P
        H_history[i] = H
        c_history[i] = np.trace(P)
        if print_iterates:
            print('iteration %3d / %3d' % (i+1, num_iterations))
            print(P)
    print('')
    results_dict = {'P': P,
                    'K': K,
                    'H': H,
                    'P_history': P_history,
                    'K_history': K_history,
                    'c_history': c_history,
                    'H_history': H_history}
    return results_dict


def midpoint_policy_iteration(problem_data, problem_data_known, K0, sim_options=None, num_iterations=100,
                                 print_iterates=False, known_solve_method='match_approx', use_half_data=False):
    problem_data_keys = ['A', 'B', 'S']
    A, B, S = [problem_data[key] for key in problem_data_keys]
    n, m = [M.shape[1] for M in [A, B]]
    K = np.copy(K0)

    # Check initial policies are stabilizing
    if specrad(A+B.dot(K0)) > 1:
        raise Exception("Initial policy is not stabilizing!")

    P_history, K_history = [np.zeros([num_iterations, dim, n]) for dim in [n, m]]
    H_history = np.zeros([num_iterations, n+m, n+m])
    c_history = np.zeros(num_iterations)

    if use_half_data:
        sim_options['nt'] = int(sim_options['nt']/2)

    # Initial policy evaluation
    IK = np.vstack([np.eye(n), K])
    P = dlyap_direct(A+B.dot(K), mdot(IK.T, S, IK))
    H = qfun(problem_data, problem_data_known, P, K, sim_options)

    print('Accelerated policy iteration')

    # If the problem data is not known, then force use of the approximate accelerated PI solve method.
    if not problem_data_known:
        known_solve_method = 'match_approx'

    # If the problem data is known, then the accelerated PI solve method may either use
    # 'direct' updates which use the value functions directly (simpler updates),
    # or may use the same update equations as used in the approximate case 'match_approx'
    # - both should give identical gains and value function matrices.
    if known_solve_method == 'direct':
        def gain(A, B, S, P):
            AB = np.hstack([A, B])
            H = S + mdot(AB.T, P, AB)
            # Hxx = H[0:n, 0:n]
            Huu = H[n:n+m, n:n+m]
            Hux = H[n:n+m, 0:n]
            return -la.solve(Huu, Hux)

        for i in range(num_iterations):
            K_history[i] = K

            # Policy evaluation (reference only)
            IK = np.vstack([np.eye(n), K])
            Peval = dlyap_direct(A+B.dot(K), mdot(IK.T, S, IK))
            P_history[i] = Peval
            c_history[i] = np.trace(Peval)

            # Compute the optimal gain matrix associated with the current cost matrix
            K = gain(A, B, S, P)

            # Compute the Newton step next-cost matrix
            AK = A+np.dot(B, K)
            IK = np.vstack([np.eye(n), K])
            QK = mdot(IK.T, S, IK)
            N = dlyap_direct(AK, QK)

            # Compute the mid-point cost matrix and associated gain matrix
            M = (P+N)/2
            L = gain(A, B, S, M)
            AL = A+np.dot(B, L)

            # Compute the mid-point Newton step next-cost matrix
            QL = mdot(IK.T, S, IK) + mdot(AK.T, P, AK) - mdot(AL.T, P, AL)
            P = dlyap_direct(AL, QL)
    elif known_solve_method == 'match_approx':
        for i in range(num_iterations):
            # Record history
            K_history[i] = K

            # Policy evaluation (reference only)
            IK = np.vstack([np.eye(n), K])
            P = dlyap_direct(A+B.dot(K), mdot(IK.T, S, IK))

            # Newton calculations
            G = qfun(problem_data, problem_data_known, None, K, sim_options)

            # Midpoint calculations
            F = (G+H)/2
            Fux = F[n:n+m, 0:n]
            Fuu = F[n:n+m, n:n+m]
            L = -la.solve(Fuu, Fux)

            # Midpoint step
            Y = np.block([[mdot(IK.T, H, IK), np.zeros([n, m])],
                          [np.zeros([m, n]), np.zeros([m, m])]]) - (H-S)
            problem_data_mid = copy(problem_data)
            problem_data_mid['S'] = Y
            V = qfun(problem_data_mid, problem_data_known, None, L, sim_options)

            # Final
            H = V + S - Y
            Hux = H[n:n+m, 0:n]
            Huu = H[n:n+m, n:n+m]
            K = -la.solve(Huu, Hux)

            # Record history
            P_history[i] = P
            H_history[i] = H
            c_history[i] = np.trace(P)
            if print_iterates:
                print('iteration %3d / %3d' % (i+1, num_iterations))
                print(P)
    print('')
    results_dict = {'P': P,
                    'K': K,
                    'H': H,
                    'P_history': P_history,
                    'K_history': K_history,
                    'c_history': c_history,
                    'H_history': H_history}
    return results_dict


def get_initial_gains(problem_data, initial_gain_method=None, r_min=0.95):
    problem_data_keys = ['A', 'B', 'S']
    A, B, S = [problem_data[key] for key in problem_data_keys]
    n, m = [M.shape[1] for M in [A, B]]
    if initial_gain_method is None:
        initial_gain_method = 'zero'
    if initial_gain_method == 'zero':
        K0 = np.zeros([m, n])
    elif initial_gain_method == 'dare':
        Q = S[0:n, 0:n]
        R = S[n:n+m, n:n+m]
        V = S[0:n, n:n+m]
        Pare, Kare = dare_gain(A, B, Q, R, E=None, S=V)
        K0 = Kare
    elif initial_gain_method == 'dare_perturb':
        K0 = get_initial_gains(problem_data, initial_gain_method='dare')
        # TODO - change this to make the LQ cost np.trace(P) high, not the specrad
        r = specrad(A+B.dot(K0))
        while not r_min < r < 1:
            K0 = K0 + 0.01*npr.randn(m, n)
            r = specrad(A+B.dot(K0))
    else:
        raise ValueError('Invalid gain initialization method chosen.')
    return K0


def verify_are(problem_data, P, algo_str=None, verbose=True):
    """Verify that the ARE is solved by the solution P"""
    if algo_str is None:
        algo_str = ''
    problem_data_keys = ['A', 'B', 'S']
    A, B, S = [problem_data[key] for key in problem_data_keys]
    n, m = [M.shape[1] for M in [A, B]]
    Q = qfun(problem_data, P=P)
    Qxx = Q[0:n, 0:n]
    Quu = Q[n:n+m, n:n+m]
    Qux = Q[n:n+m, 0:n]
    LHS = P
    RHS = Qxx - np.dot(Qux.T, la.solve(Quu, Qux))
    diff = la.norm(LHS-RHS)
    if verbose:
        print(algo_str)
        print('-' * len(algo_str))
        print(' Left-hand side of the ARE: Positive definite = %s' % is_pos_def(LHS))
        print(LHS)
        print('')
        print('Right-hand side of the ARE: Positive definite = %s' % is_pos_def(RHS))
        print(RHS)
        print('')
        print('Difference')
        print(LHS-RHS)
        print('\n')
    return diff




