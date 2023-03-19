import numpy as np
import numpy.linalg as la
import numpy.random as npr
import scipy.linalg as sla
from copy import copy
from warnings import warn

from utility.matrixmath import mdot, specrad, dare_gain, dlyap, is_pos_def, svec2, smat2, vec


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


def dlyap_wrap(A, Q, direct=False):
    if direct:
        vQ = vec(Q)
        A2 = np.kron(A, A)
        vP = la.solve(np.eye(A.size)-A2, vQ)
        return np.reshape(vP, A.shape)
    else:
        return dlyap(A, Q)


def rollout(problem_data, K, sim_options):
    """Simulate closed-loop state response"""
    problem_data_keys = ['A', 'B', 'S']
    A, B, S = [problem_data[key] for key in problem_data_keys]
    n, m = [M.shape[1] for M in [A, B]]

    # Check for stability and issue warning if closed-loop unstable
    if specrad(A + np.dot(B, K)) > 1:
        warn('Closed-loop unstable in rollout, states will blow up')

    sim_options_keys = ['xstd', 'ustd', 'wstd', 'nt', 'nr']
    xstd, ustd, wstd, nt, nr = [sim_options[key] for key in sim_options_keys]

    # Sample initial states, control inputs
    x0 = xstd * npr.randn(nr, n)
    u_explore_hist = ustd * npr.randn(nr, nt, m)

    # Initialize history data arrays
    x_hist = np.zeros([nr, nt, n])
    u_hist = np.zeros([nr, nt, m])

    # Randomly sample an additive noise sequence
    w_hist = wstd * npr.randn(nt, nr, n)

    # Initialize
    x = np.copy(x0)

    # Iterate over timesteps
    for i in range(nt):
        # Compute controls
        u = groupdot(K, x) + u_explore_hist[:, i]

        # Record history
        x_hist[:, i] = x
        u_hist[:, i] = u

        # Look up additive noise
        w = w_hist[i]

        # Transition the state using additive noise
        x = groupdot(A, x) + groupdot(B, u) + w

    return x_hist, u_hist


def rollout_cost(x_hist, u_hist, S):
    """Compute stage costs associated to a state-input trajectory"""
    nr, nt = x_hist.shape[0], x_hist.shape[1]
    c_hist = np.zeros([nr, nt])
    # Iterate over timesteps
    for i in range(nt):
        x = x_hist[:, i] 
        u = u_hist[:, i]
        z = np.hstack([x, u])
        c_hist[:, i] = groupquadform(S, z)
    return c_hist


def qfun(problem_data, problem_data_known=None, P=None, K=None, sim_options=None, sim_data=None):
    """Compute or estimate Q-function matrix"""
    if problem_data_known is None:
        problem_data_known = True
    problem_data_keys = ['A', 'B', 'S']
    A, B, S = [problem_data[key] for key in problem_data_keys]
    n, m = [M.shape[1] for M in [A, B]]

    if problem_data_known:
        if P is None:
            IK = np.vstack([np.eye(n), K])
            AK = A+B.dot(K)
            P = dlyap_wrap(AK.T, mdot(IK.T, S, IK))
        AB = np.hstack([A, B])
        Q = S + mdot(AB.T, P, AB)
    else:
        nr = sim_options['nr']
        nt = sim_options['nt']

        # Check if there is enough data to form a unique least-squares estimate
        if nt < (n+m)*(n+m+1)/2:
            warn('Rollout length is %d, should be > %d for unique LS estimate' % (nt, int((n+m)*(n+m+1)/2)))

        qfun_estimator = sim_options['qfun_estimator']
        x_hist, u_hist, c_hist = sim_data

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
            # Form the correction term to account for process noise
            # This requires the process noise covariance W to be known
            # IK = np.vstack([np.eye(n), K])
            # W = problem_data['W']
            # f = svec2(mdot(IK, W, IK.T))
            for i in range(nr):
                lwr = i*nz
                upr = (i+1)*nz
                for j in range(nt-1):
                    Y[lwr:upr] += mu_hist[i, j]*c_hist[i, j]
                    Z[lwr:upr] += np.outer(mu_hist[i, j], mu_hist[i, j] - nu_hist[i, j+1])
                    # Use this line if correcting bias due to process noise
                    # Z[lwr:upr] += np.outer(mu_hist[i, j], mu_hist[i, j] - nu_hist[i, j+1] + f)
            # Solve the least squares problem
            try:
                # Q_svec2 = la.lstsq(Z, Y, rcond=None)[0]
                Q_svec2 = sla.lstsq(Z, Y)[0]
            except:
                print('Something went wrong when solving least-squares problem in the LSTDQ estimator!')
                print(lwr)
                print(upr)
                print(mu_hist)
                print(Y)
                print(Z)
            Q = smat2(Q_svec2)
        else:
            raise ValueError('Invalid Q-function estimator chosen.')
    return Q


def policy_iteration(problem_data, problem_data_known, K0, sim_options=None, num_iterations=10,
                     solver=None, known_solve_method='match_approx', use_half_data=False, use_half_compute=False,
                     use_increasing_rollout_length=False,
                     offline_training_data=None,
                     share_data_KL=False,
                     print_iterates=False,
                     print_diagnostic=False):
    problem_data_keys = ['A', 'B', 'S']
    A, B, S = [problem_data[key] for key in problem_data_keys]
    n, m = [M.shape[1] for M in [A, B]]
    K = np.copy(K0)

    # Check initial policies are stabilizing
    if specrad(A+B.dot(K0)) > 1:
        raise Exception("Initial policy is not stabilizing!")

    P_history = np.zeros([num_iterations, n, n])
    H_history = np.zeros([num_iterations, n+m, n+m])
    c_history = np.zeros(num_iterations)
    K_history = np.zeros([num_iterations+1, m, n])

    if solver is None:
        solver = 'policy_iteration'

    sim_options = copy(sim_options)

    if use_half_data:
        sim_options['nt'] = int(sim_options['nt']/2)

    if use_half_compute:
        num_iterations = int(num_iterations/2)

    if use_increasing_rollout_length:
        nt_0 = copy(sim_options['nt'])

    def get_sim_data(K, problem_data, offline_training_data, problem_data_known, sim_options):
        if problem_data_known:
            sim_data = None
        else:
            if offline_training_data is None:
                x_hist, u_hist = rollout(problem_data, K, sim_options)
            else:
                x_hist = offline_training_data[0][0:sim_options['nt']]
                u_hist = offline_training_data[1][0:sim_options['nt']]
            c_hist = rollout_cost(x_hist, u_hist, problem_data['S'])
            sim_data = [x_hist, u_hist, c_hist]
        return sim_data

    # Initial policy evaluation
    K_history[0] = K0
    IK = np.vstack([np.eye(n), K])
    AK = A+B.dot(K)
    P = dlyap_wrap(AK.T, mdot(IK.T, S, IK))
    sim_data_K = get_sim_data(K, problem_data, offline_training_data, problem_data_known, sim_options)
    H = qfun(problem_data, problem_data_known, P, K, sim_options, sim_data_K)

    # If the problem data is not known, then force use of the approximate policy iteration solve method.
    if not problem_data_known:
        known_solve_method = 'match_approx'

    # If the problem data is known, then the midpoint PI solve method may either use
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
            if print_iterates:
                print('iteration %3d / %3d' % (i+1, num_iterations))

            # Check for stability and issue warning if closed-loop unstable
            sABK = specrad(A+np.dot(B, K))
            if sABK > 1:
                raise ValueError('Closed-loop went unstable in policy iteration! Specrad(A+BK)=%f' % sABK)
            # Policy evaluation (reference only)
            IK = np.vstack([np.eye(n), K])
            AK = A+B.dot(K)
            Peval = dlyap_wrap(AK.T, mdot(IK.T, S, IK))
            P_history[i] = Peval
            c_history[i] = np.trace(Peval)

            # Compute the optimal gain matrix associated with the current cost matrix
            K = gain(A, B, S, P)
            K_history[i+1] = K

            # Compute the Newton step next-cost matrix
            AK = A+np.dot(B, K)
            IK = np.vstack([np.eye(n), K])
            QK = mdot(IK.T, S, IK)
            N = dlyap_wrap(AK.T, QK)

            # Conditionally execute the midpoint calculations
            if solver == 'policy_iteration':
                P = np.copy(N)
            elif solver == 'midpoint_policy_iteration':
                # Compute the mid-point cost matrix and associated gain matrix
                M = (P+N)/2
                L = gain(A, B, S, M)
                AL = A+np.dot(B, L)

                # Compute the mid-point Newton step next-cost matrix
                QL = mdot(IK.T, S, IK) + mdot(AK.T, P, AK) - mdot(AL.T, P, AL)
                if print_diagnostic:
                    print("eigenvalues of midpoint closed-loop penalty matrix QL % s" % str(np.sort(la.eig(QL)[0])))
                P = dlyap_wrap(AL.T, QL)

    elif known_solve_method == 'match_approx':
        for i in range(num_iterations):
            if print_iterates:
                print('iteration %3d / %3d' % (i+1, num_iterations))

            # Check for stability and issue warning if closed-loop unstable
            sABK = specrad(A+np.dot(B, K))
            if sABK > 1:
                raise ValueError('Closed-loop went unstable in policy iteration! Specrad(A+BK)=%f' % sABK)

            # Policy evaluation (reference only)
            IK = np.vstack([np.eye(n), K])
            AK = A+B.dot(K)
            P = dlyap_wrap(AK.T, mdot(IK.T, S, IK))

            if use_increasing_rollout_length:
                sim_options['nt'] = int(nt_0 * ((i+1)**1))

            # Newton calculations
            sim_data_K = get_sim_data(K, problem_data, offline_training_data, problem_data_known, sim_options)
            G = qfun(problem_data, problem_data_known, P=None, K=K, sim_options=sim_options, sim_data=sim_data_K)

            # Conditionally execute the midpoint calculations
            if solver == 'policy_iteration':
                H = np.copy(G)
            elif solver == 'midpoint_policy_iteration':
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
                if share_data_KL:
                    sim_data_L = get_sim_data(L, problem_data_mid, sim_data_K, problem_data_known, sim_options)
                else:
                    sim_data_L = get_sim_data(L, problem_data_mid, offline_training_data, problem_data_known, sim_options)
                V = qfun(problem_data_mid, problem_data_known, P=None, K=L, sim_options=sim_options, sim_data=sim_data_L)
                H = V + S - Y

            # Policy improvement
            Hux = H[n:n+m, 0:n]
            Huu = H[n:n+m, n:n+m]
            K = -la.solve(Huu, Hux)

            # Record history
            K_history[i+1] = K
            P_history[i] = P
            H_history[i] = H
            c_history[i] = np.trace(P)

    if print_iterates:
        print('')
    results_dict = {'P': P,
                    'K': K,
                    'H': H,
                    'P_history': P_history,
                    'K_history': K_history,
                    'c_history': c_history,
                    'H_history': H_history}
    return results_dict


def get_initial_gains(problem_data, initial_gain_method=None, frac_tgt=10, bisection_epsilon=1e-6, return_Pare=False):
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
        if return_Pare:
            return Kare, Pare
    elif initial_gain_method == 'dare_perturb':
        Kare, Pare = get_initial_gains(problem_data, initial_gain_method='dare', return_Pare=True)
        are_cost = la.norm(Pare, ord=2)
        c_lwr = 0.0
        c_upr = 1.0
        Kdel = npr.randn(m, n)
        Kdel = Kdel/la.norm(Kdel)

        def eval_frac(c):
            K = Kare + c*Kdel
            AK = A+B.dot(K)
            if specrad(AK) < 1:
                IK = np.vstack([np.eye(n), K])
                P = dlyap_wrap(AK.T, mdot(IK.T, S, IK))
                frac = la.norm(P-Pare, ord=2) / are_cost
            else:
                frac = np.inf
            return frac

        # Start by making K0 de-stabilizing
        while np.isfinite(eval_frac(c_upr)):
            c_upr *= 2

        # Do bisection to find K0 that makes initial cost high
        c_mid = (c_upr+c_lwr) / 2
        frac = eval_frac(c_mid)
        while abs(1-(frac/frac_tgt)) > bisection_epsilon and (c_upr-c_lwr) > bisection_epsilon:
            c_mid = (c_upr+c_lwr) / 2
            frac = eval_frac(c_mid)
            if frac > frac_tgt:
                c_upr = c_mid
            else:
                c_lwr = c_mid
        K0 = Kare + c_lwr*Kdel
        # print(eval_frac(c_lwr))
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
