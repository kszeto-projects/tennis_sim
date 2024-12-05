import casadi as ca
import numpy as np
from skimage.morphology import max_tree

from dynamics import D, C, g, forward_kinematics, g_casadi

def nlp(q, qdot, dt, T=1.0):
    m = 3
    n = 6

    N = int(T / dt)

    x = ca.MX.sym('x', n)  # state space variable
    u = ca.MX.sym('u', m)  # action space variable
    t = ca.MX.sym('t')

    # Create CasADi integrator
    ode = {'x': x, 'p': u, 'ode': f(x, u)}
    opts = {'tf': T / N}
    F = ca.integrator('F', 'rk', ode, opts)

    X = ca.MX.sym('X', N+1, n)
    U = ca.MX.sym('U', N, m)

    x0 = ca.vertcat(q, qdot).T
    cost = ca.MX(0)
    constraints = [X[0,:] - x0]
    for k in range(N):
        x_next = F(x0=X[k,:], p=U[k,:])['xf'].T
        cost += L(X[k,:], U[k,:], k * dt)
        constraints.append(X[k+1,:] - x_next)

    constraints = ca.vertcat(*constraints).reshape((-1,1))

    lbg = ca.MX.zeros(constraints.shape)
    ubg = ca.MX.zeros(constraints.shape)

    lbx = -10000 * ca.MX.ones((n * (N + 1) + m * N, 1))
    ubx = 10000 * ca.MX.ones((n * (N + 1) + m * N, 1))
    max_torque = .01
    ubx[-m * N:] = max_torque
    lbx[-m * N:] = -max_torque

    nlp_prob = {
        'x': ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))),
        'f': cost,
        'g': constraints
    }
    # opts = {
        # 'ipopt': {
        #     'print_level': 12,
        #     'max_iter': 1000,
        #     'tol': 1e-6,
        # }
    # }
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob)
    print('solver:', solver)
    x0_guess = np.zeros((n * (N + 1) + m * N, 1))
    x0_guess[0:n * (N + 1)] = np.tile(x0, N + 1).reshape(-1,1)
    x0_guess[n * (N + 1):] = np.tile(g(np.array(x0.T[:3])), N).flatten()[np.newaxis].T

    sol = solver(x0=x0_guess, lbx=-100000, ubx=100000, lbg=0, ubg=0)
    print(sol['x'])
    final_cost = sol['f']
    print(f'Final cost: {final_cost}')
    optimal_states = sol['x'][0:n * (N + 1)]
    optimal_controls = np.hstack([sol['x'][-3*N:-2*N], sol['x'][-2*N:-1*N], sol['x'][-1*N:]])
    print('optimal states:', optimal_states)
    print('optimal controls:', optimal_controls)
    return optimal_states, optimal_controls

def f(x, u):
    q = x[0:3]
    qdot = x[3:6]
    I = ca.MX.eye(3)
    O = ca.MX.zeros(3, 3)
    Dp = ca.vertcat(
        ca.horzcat(I, O),
        ca.horzcat(O, D(q))
    )

    C_x = ca.vertcat(-qdot, C(q, qdot))
    Gp = ca.vertcat(ca.MX.zeros((3, 1)), g_casadi(q))
    Up = ca.vertcat(ca.MX.zeros(3, 1), u)
    xdot = ca.pinv(Dp) @ (-C_x - Gp + Up)
    return xdot

def b(t):
    grav = np.array([0, 0, -9.81])
    ball_pos = np.array([0.8, 0, 0.2])
    ball_vel = np.array([-1, 0, 3])

    def vel(t):
        return ball_vel + grav * t

    def pos(t):
        return ball_pos + ball_vel * t + 0.5 * grav * t ** 2

    return pos(t), vel(t)

def L(x, u, t):
    ee_pos = forward_kinematics(x[0:3])
    control_weight = 0.2
    internal_cost = ca.norm_2(ee_pos - b(t)[0]) ** 2 + control_weight * ca.norm_2(u) ** 2
    return internal_cost

if __name__ == '__main__':
    q = np.array([0, 0, 0])
    qdot = np.array([0, 0, 0])
    dt = 10.0/240.0
    optimal_states, optimal_controls = nlp(q, qdot, dt)
    print(optimal_controls.shape)