import numpy as np
from ortools.linear_solver import pywraplp

def geom_walks_approx(w0, r, K, s, step, gamma = 0.5):
    H = len(r)
    mu_effective = sum([r[h]/K[h] for h in range(H)]) / H
    sigma = np.sqrt((sum([(K[h+1] - K[h]) * 
                 ((r[h+1] - r[h])/(K[h + 1] - K[h]) - mu_effective)**2 for h in range(H-1)]) 
             + K[0] * (r[0]/K[0] - mu_effective)**2)/H)
    mu = mu_effective + 0.5 * sigma ** 2
    
    mu = mu * step
    sigma = sigma * np.sqrt(step)
    
    X = [-1.0, 1.0, w0, (mu + s)/(2 * gamma * sigma ** 2), (mu - s)/(2 * gamma * sigma ** 2)]
    X = [e for e in X if( e <= 1 and e >= -1.0)]
    Y = [mu * e - gamma * sigma ** 2 * e ** 2  - s * abs(e - w0) for e in X]
    I = np.argmax(Y)
    return X[I]

def min_profit_multi_scale(w0, r, sigma, s):
    H = len(r)
    solver = pywraplp.Solver.CreateSolver('GLOP')
    
    w = [solver.NumVar(0.0, 1.0, 'w' + str(h)) for h in range(1,H + 1)]
    
    z0 = solver.NumVar(-1.0,1.0, 'z' + str(0))
    solver.Add(z0 == w[0] - w0)
    
    z = [solver.NumVar(-1.0,1.0, 'z' + str(h)) for h in range(1,H)]
    for h in range(1,H - 1):
        solver.Add(z[h] == w[h] - w[h - 1])
        
    for h in range(H - 1):
        if (sigma[h] == np.inf):
            sigma[h] = 0
            solver.Add(z[h] == 0)
    
    if (sigma[H - 1] == np.inf):
        sigma[H - 1] = 0
        solver.Add(w[H - 1] == 0)
        
    u0 = solver.NumVar(0.0,1.0, 'u' + str(0))
    solver.Add(u0 >= z0)
    solver.Add(u0 >= -1 * z0)
    
    u = [solver.NumVar(0.0,1.0, 'u' + str(h)) for h in range(1,H)]
    for h in range(H-1):
        solver.Add(u[h] >= z[h])
        solver.Add(u[h] >= -1 * z[h])
    
    solver.Maximize( (r[H - 1] - sigma[H - 1]) * w[H - 1] 
                    - sum([r[h] * z[h] for h in range(H - 1)]) 
                    - sum([(sigma[h] + s) * u[h] for h in range(H - 1)]) 
                    - s * u0)
    
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        w1 = w[0].solution_value()
        res = solver.Objective().Value()
        return w1
    
    print('The problem does not have an optimal solution.')
    return w0

def percent_profit_multi_scale(w0, r, sigma, s, Gamma = 0):
    H = len(r)
    solver = pywraplp.Solver.CreateSolver('GLOP')
    
    w = [solver.NumVar(0.0, 1.0, 'w' + str(h)) for h in range(1,H + 1)]
    z0 = w[0] - w0
    z = [w[h] - w[h - 1] for h in range(1,H)]
    
    for h in range(H - 1):
        if (sigma[h] == np.inf):
            sigma[h] = 0
            solver.Add(z[h] == 0)
    
    if (sigma[H - 1] == np.inf):
        sigma[H - 1] = 0
        solver.Add(w[H - 1] == 0)
        
    u0 = solver.NumVar(0.0,1.0, 'u' + str(0))
    solver.Add(u0 >= z0)
    solver.Add(u0 >= -1 * z0)
    
    u = [solver.NumVar(0.0,1.0, 'u' + str(h)) for h in range(1,H)]
    for h in range(H-1):
        solver.Add(u[h] >= z[h])
        solver.Add(u[h] >= -1 * z[h])
    
    R_mean = (r[H - 1] - sigma[H - 1]) * w[H - 1] - sum([r[h] * z[h] for h in range(H - 1)])
    R = solver.NumVar(-solver.infinity(), solver.infinity(), 'R')
    
    p = [solver.NumVar(0.0, solver.infinity(), 'p_' + str(h)) for h in range(1, H + 1)]
    y = solver.NumVar(0.0, solver.infinity(), 'y')
    for h in range(H - 1):
        solver.Add(y + p[h] >= sigma[h] * u[h])
    solver.Add(y + p[H - 1] >= sigma[H - 1] * w[H - 1])
    solver.Add(R + sum(p) + (H + 1) * y <= R_mean)
    
    solver.Maximize(R - s * sum(u) - s * u0)
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        w1 = w[0].solution_value()
        res = solver.Objective().Value()
        return w1
    
    print('The problem does not have an optimal solution.')
    return w0

