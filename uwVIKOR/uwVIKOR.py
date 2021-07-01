import pandas as pd
from numpy import *
import scipy.optimize as opt

def requirements(data, directions, L, U, v, w0, display):
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
        raise ValueError('[!] Te utility parameter "v" must belong to (0,1)')
    
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

def norm(x, w, p, n):
    # VIKOR data normalization
    z = array(w)*(array(p) - array(x))/(array(p) - array(n))
    return z

def get_ideals(data, directions):
    pos_max = [int(i == 'max') for i in directions]
    pos_min = [int(i == 'min') for i in directions]
    col_max = data.apply(lambda z: max(z))
    col_min = data.apply(lambda z: min(z))
    P = array(col_max)*array(pos_max) + array(col_min)*array(pos_min)
    N = array(col_max)*(1-array(pos_max)) + array(col_min)*(1-array(pos_min))
    return P, N

def initial_weights(w0, L, U, J):
    if len(w0) == 0:
        w = 1/(J-2)*(1-L-U)
    else:
        w = w0
    return w

def S(data, w, P, N):
    # S score according to the Manhattan metric
    s = data.sum(axis=1)
    return s

def R(data, w, P, N):
    # R score according to the Chebyshev metric
    r = data.max(axis=1)
    return r

def Q(data, w, P, N, v):
    # S score
    s = data.sum(axis=1)
    s_min = min(s)
    s_max = max(s)
    # R score
    r = data.max(axis=1)
    r_min = min(r)
    r_max = max(r)
    # Q score as the convex lineal combination of S and R
    q = v*(s-s_min)/(s_max-s_min)+(1-v)*(r-r_min)/(r_max-r_min)
    return q

def Q_i(w, data, P, N, v, i, mode):
    # Objective function
    norm_data = norm(data, w, P, N)
    q = Q(norm_data, w, P, N, v)
    if mode == 'min':
        q_i = q[i]
    else:
        q_i = -q[i]
    return q_i

def optimize_VIKOR(data, w, P, N, v, L, U, I, J, display, optimal_mode):
    
    bounds = [(l,u) for l, u in zip(L, U)]
    constraints = ({'type': 'ineq', 'fun': lambda w: 1-sum(w)},
                   {'type': 'ineq', 'fun': lambda w: sum(w)-1},)
    data_norm = norm(data, w, P, N)

    # Optimizing the Q-score according to the optimal_mode
    sol_opt = []
    for i in range(I):
        id_max = argmax(data_norm[i])
        id_min = argmin(data_norm[i])
        w0 = 1/(J-2)*(1-L-U)
        if optimal_mode == 'max':
            w0[id_max] = U[id_max]
            w0[id_min] = L[id_min]
        elif optimal_mode == 'min':
            w0[id_max] = L[id_max]
            w0[id_min] = U[id_min]
        opt_i = opt.minimize(fun = Q_i,
                            x0 = w0,
                            args = (data, P, N, v, i, optimal_mode),
                            method = 'SLSQP',
                            bounds = bounds,
                            constraints =  constraints,
                            tol = 10**(-9),
                            options = {'disp': display})
        if optimal_mode == 'max':
            opt_i.fun = -opt_i.fun
        sol_opt.append(opt_i)
    s = []
    r = []
    q = []
    w = []
    i = 0
    for sol in sol_opt:
        data_norm = norm(data, sol.x, P, N)
        s.append(S(data_norm, sol.x, P, N)[i])
        r.append(R(data_norm, sol.x, P, N)[i])
        q.append(sol.fun)
        w.append(sol.x)
        i += 1
    return s, r, q, w

def uwVIKOR(data, directions, L, U, v=0.5, w0=[], display = False):
    """
    uwVIKOR method
    Input:
        data: dataframe which contains the alternatives and the criteria.
        directions: array with the optimal direction of the criteria.
        L: array with the lower bounds of the weigths.
        U: array with the upper bounds of the weigths.
        v: value of the utility parameter. (By default v = 0.5)
        w0: array with the initial guess of the weights
        display: logical argument to indicate whether to show print convergence messages or not. (By default display = False)
    Output:
        List which contains three keys.
            Ranking: List with S, R and Q scores in regard of the optimal weights.
            Weights_min: List with the weights that minimizes the Q score.
            Weights_max: List with the weights that maximizes the Q score.
    """
    # Check whether the data input verifies the basic requirements
    requirements(data, directions, L, U, v, w0, display)

    # 1st step: Get PIS and NIS elements
    I = len(data.index)
    J = len(data.columns)
    P, N = get_ideals(data, directions)
    w0 = initial_weights(w0, L, U, J)

    # 2nd step: Optimize Q score and return their associate S and R scores with the weights
    s_min, r_min, q_min, w_min = optimize_VIKOR(data, w0, P, N, v, L, U, I, J, display, optimal_mode = 'min')
    s_max, r_max, q_max, w_max = optimize_VIKOR(data, w0, P, N, v, L, U, I, J, display, optimal_mode = 'max')
    
    # Output preparation
    scores = {'S_min': s_min, 'S_max': s_max,
              'R_min': r_min, 'R_max': r_max,
              'Q_min': q_min, 'Q_max': q_max}
    output_uwVIKOR = {'Ranking': scores, 'Weights_min': w_min, 'Weights_max': w_max}

    return output_uwVIKOR

