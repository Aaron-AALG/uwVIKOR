import numpy as np
import scipy.optimize as opt

def requirements(data, directions, L, U, v, w0, display):
    '''
    UW-VIKOR requirements
    ---------------------

    Checks whether the input parameters satisfy the UW-VIKOR hypothesis.
    '''
    # data requirements
    if len(data.shape) < 2:
        raise ValueError('[!] data must be matrix-shaped')
    elif data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError('[!] data must be matrix-shaped')
    
    # directions requirements
    if len(directions) != data.shape[1]:
        raise ValueError('[!] Number of optimal directions must be equal to the number of criteria')
    if not all([d == 'min' or d == 'max' for d in directions]):
        raise ValueError('[!] Optimal directions must be either "max" or "min"')
    
    # L requirements
    if len(L) != data.shape[1]:
        raise ValueError('[!] Number of lower bounds must be equal to the number of criteria')
    if any([l < 0 or l > 1 for l in L]):
        raise ValueError('[!] Lower bounds must belong to [0,1]')
    if sum(L) > 1:
        raise ValueError('[!] The sum of lower bounds must be less than 1')
    
    # U requirements
    if len(U) != data.shape[1]:
        raise ValueError('[!] Number of upper bounds must be equal to the number of criteria')
    if any([u < 0 or u > 1 for u in U]):
        raise ValueError('[!] Upper bounds must belong to [0,1]')
    if sum(U) < 1:
        raise ValueError('[!] The sum of upper bounds must be greater than 1.')
    
    # v requirements
    if any([v < 0, v > 1]):
        raise ValueError('[!] Te utility parameter "v" must belong to [0,1]')
    
    # w0 requirements
    if len(w0) > 0:
        if len(w0) != data.shape[1]:
            raise ValueError('[!] Length of initial weight must be equal to the number of criteria')
        if all([w < 0 for w in w0]) and all([w > 1 for w in w0]):
            raise ValueError('[!] Initial weights must belong to [0,1]')
        if abs(sum(w0) - 1) > 10**(-6):
            raise ValueError('[!] Initial weights must sum 1')

    # makefigure requirements
    if int(display) not in [0,1]:
        raise ValueError('[!] "display" must be boolean')
    return

def VIKOR_normalization(data, w, PIS, NIS):
    '''
    VIKOR normalization
    -------------------

    Normalize data as the original VIKOR method.
    '''
    data_normalized = w * (PIS - data) / (PIS - NIS)
    return data_normalized

def VIKOR_ideals(data, directions):
    '''
    VIKOR ideal solutions
    ---------------------

    Extraction of the positive/negative solutions of the decision matrix.
    '''
    # Binary arrays with the indexes
    idx_max = np.array([int(i == 'max') for i in directions])
    idx_min = np.array([int(i == 'min') for i in directions])
    # Max/Min elements per each criteria, i.e. per column.
    col_max = data.max(axis=0)
    col_min = data.min(axis=0)
    # Compute ideal solution
    PIS = col_max * idx_max + col_min * idx_min
    NIS = col_max * (1 - idx_max) + col_min * (1 - idx_min)
    return PIS, NIS

def initial_weights(w0, L, U, J):
    '''
    Build initial wegihts for the optimization problem
    -----
    '''
    if len(w0) == 0:
        w = 1 / (J-2) * (1 - L - U)
    else:
        w = w0
    return w

def S_score(data, w, P, N):
    '''
    VIKOR S-score
    -------------

    S-score according to the Manhattan metric for the decision matrix.
    '''
    s_score = data.sum(axis=1)
    return s_score

def R_score(data, w, P, N):
    '''
    VIKOR R-score
    -------------

    R-score according to the Chebyshev metric for the decision matrix.
    '''
    r_score = data.max(axis=1)
    return r_score

def Q_score(data, w, P, N, v):
    '''
    VIKOR Q-score
    -------------

    Q-score of the VIKOR method.
    '''
    # S score
    s = S_score(data, w, P, N)
    s_min = np.min(s)
    s_max = np.max(s)
    # R score
    r = R_score(data, w, P, N)
    r_min = np.min(r)
    r_max = np.max(r)
    # Q score as the convex max-min combination of S and R
    q = v * (s - s_min) / (s_max - s_min) + (1-v) * (r - r_min) / (r_max - r_min)
    return q

def UWVIKOR_objective_function(w, data, P, N, v, i, mode):
    '''
    Objective function VIKOR Q-score
    -------------

    Q-score for the ith alternative in order to optimize it for the UW-VIKOR method.
    '''
    # Normalization
    norm_data = VIKOR_normalization(data, w, P, N)
    # Objective function
    q = Q_score(norm_data, w, P, N, v)
    if mode == 'min':
        q_i = q[i]
    else:
        q_i = -q[i]
    return q_i

def optimize_VIKOR(data, data_norm, w0, P, N, v, L, U, I, J, display, optimal_mode):
    '''
    Optimize the Q-scores of the VIKOR method
    -------------

    Optimization (min/max) of theQ-score per each alternative.
    '''
    # Define bounds and constraints of the optimization problem
    bounds = [(l,u) for l, u in zip(L, U)]
    constraints = ({'type': 'ineq', 'fun': lambda w: 1-sum(w)},
                   {'type': 'ineq', 'fun': lambda w: sum(w)-1},)
    # Optimizing the Q-score according to the optimal_mode and per each alternative of the problem
    optimal_solution = []
    for i in range(I):
        # Determine an "appropriate" initial weights
        id_max = data_norm[i].argmax()
        id_min = data_norm[i].argmin()
        if optimal_mode == 'max':
            w0[id_max] = U[id_max]
            w0[id_min] = L[id_min]
        elif optimal_mode == 'min':
            w0[id_max] = L[id_max]
            w0[id_min] = U[id_min]
        # Optimize the Q[i] score
        opt_i = opt.minimize(fun = UWVIKOR_objective_function,
                            x0 = w0,
                            args = (data, P, N, v, i, optimal_mode),
                            method = 'SLSQP',
                            bounds = bounds,
                            constraints =  constraints,
                            tol = 10**(-9),
                            options = {'disp': display},
                            )
        if optimal_mode == 'max':
            opt_i.fun = - opt_i.fun
        optimal_solution.append(opt_i)
    # Store the scores and the optimal weights
    s = []
    r = []
    q = []
    w = []
    for i, sol in enumerate(optimal_solution):
        data_norm = VIKOR_normalization(data, sol.x, P, N)
        s.append(S_score(data_norm, sol.x, P, N)[i])
        r.append(R_score(data_norm, sol.x, P, N)[i])
        q.append(sol.fun)
        w.append(sol.x)
    return s, r, q, w

def uwVIKOR(data, directions, L, U, v = 0.5, w0 = [], display = False):
    """
    Unweighted VIKOR method (UW-VIKOR)
    ==================================

    Input:
    ------
        data: matrix which contains the alternatives and the criteria.
        directions: array with the optimal direction of the criteria.
        L: array with the lower bounds of the weigths.
        U: array with the upper bounds of the weigths.
        v: value of the utility parameter (By default v = 0.5).
        w0: array with the initial guess of the weights (By default w0 is empty).
        display: logical argument to indicate whether to show print convergence messages or not (By default display = False).

    Output:
    -------
        List which contains three keys.
            Ranking: List with S, R and Q scores in regard of the optimal weights.
            Weights_min: List with the weights that minimizes the Q score.
            Weights_max: List with the weights that maximizes the Q score.
    """
    # Check whether the data input verifies the basic requirements
    data = np.array(data)
    I, J = data.shape
    requirements(data, directions, L, U, v, w0, display)

    # 1st step: Get PIS and NIS elements
    PIS, NIS = VIKOR_ideals(data, directions)
    w0 = initial_weights(w0, L, U, J)

    # 2nd step: Normalize data as VIKOR manner
    data_norm = VIKOR_normalization(data, w0, PIS, NIS)

    # 3rd step: Optimize Q score and return their associate S and R scores with the weights
    s_min, r_min, q_min, w_min = optimize_VIKOR(data, data_norm, w0, PIS, NIS, v, L, U, I, J, display, optimal_mode = 'min')
    s_max, r_max, q_max, w_max = optimize_VIKOR(data, data_norm, w0, PIS, NIS, v, L, U, I, J, display, optimal_mode = 'max')
    
    # Output preparation
    scores = {
        'S_min': s_min, 'S_max': s_max,
        'R_min': r_min, 'R_max': r_max,
        'Q_min': q_min, 'Q_max': q_max
        }
    output_UW_VIKOR = {'Ranking': scores, 'Weights_min': w_min, 'Weights_max': w_max}

    return output_UW_VIKOR
