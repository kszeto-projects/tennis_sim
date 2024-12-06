
import casadi as ca
import numpy as np

from dynamics import D, C, g, forward_kinematics, g_casadi
from config import grav, ball_pos, ball_vel, robot1_base, robot2_base

def nlp(q, qdot_initial, dt, T=1.0):
    m = 3
    n = 3

    N = int(T / dt)

    Xsym = ca.MX.sym('X', n*(N+1),1)
    Usym = ca.MX.sym('U', m*N,1)

    X = Xsym.reshape((n, N+1))
    U = Usym.reshape((m, N))
    x0 = ca.vertcat(q)
    cost = ca.MX(0)
    constraints = [X[:,0] - x0]
    for k in range(N):
        x_next = X[:,k] + U[:,k] * dt
        cost += L(X[:,k], U[:,k], k * dt)
        constraints.append(X[:,k+1] - x_next)
        
     #add acceleration_constraints
    constraints.append((U[:,0] - qdot_initial))
    for k in range(N-1):
        constraints.append((U[:,k+1] - U[:,k])/dt)
    constraints = ca.vertcat(*constraints).reshape((-1,1))

    lbg = np.zeros(constraints.shape)
    ubg = np.zeros(constraints.shape)
    max_joint_accel = 10000
    lbg[-m*N:] = -max_joint_accel
    ubg[-m*N:] = max_joint_accel

    lbx = -np.inf * np.ones((n * (N + 1) + m * N, 1))
    ubx = np.inf * np.ones((n * (N + 1) + m * N, 1))
    max_joint_vel = 100_000
    ubx[-m * N:] = max_joint_vel
    lbx[-m * N:] = -max_joint_vel

    #stack x and u into a single vector
    # xs, us = [], []
    # for i in range(N):
    #     xs.append(X[:,i])
    #     us.append(U[:,i])
    # xs.append(X[:,N])
    casadi_x = ca.vertcat(Xsym, Usym)
    nlp_prob = {
        'x': casadi_x,
        'f': cost,
        'g': constraints
    }
    # opts = {
    #     'ipopt': {
    #         'print_level': 12,
    #         'max_iter': 1000,
    #         'tol': 1e-6,
    #     }
    # }
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob)
    # print('solver:', solver)
    x0_guess = np.zeros((n * (N + 1) + m * N, 1))
    x0_guess[0:n * (N + 1)] = np.tile(q, N + 1).reshape(-1,1)
    x0_guess[n * (N + 1):] = np.tile(np.array([0.0, 0.01, 0.01]), N).flatten()[np.newaxis].T
    sol = solver(x0=x0_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    # print(sol['x'])
    final_cost = sol['f']
    print(f'Final cost: {final_cost}')
    optimal_states = np.array(sol['x'][0:n * (N + 1)].reshape((n, N + 1))).T
    optimal_controls = np.array(sol['x'][n * (N + 1):].reshape((m, N))).T
    # print('optimal states:', optimal_states)
    # print('optimal controls:', optimal_controls)
    return optimal_states, optimal_controls

def b(t):

    def vel(t):
        return ball_vel + grav * t

    def pos(t):
        return ball_pos + ball_vel * t + 0.5 * grav * t ** 2

    return pos(t), vel(t)

def L(x, u, t):
    #ee_pos = forward_kinematics(x[0:3]) for robot1
    # for robot2:
    ee_pos = forward_kinematics(x[0:3]) + ([robot2_base - robot1_base])
    control_weight = 0.0001
    internal_cost = ca.norm_2(ee_pos - b(t)[0]) ** 2 # + control_weight * ca.norm_2(u) ** 2
    return internal_cost

if __name__ == '__main__':
    q = np.array([0, 0.1, 0.1])
    dt = 10.0/240.0
    optimal_states, optimal_controls = nlp(q, dt)
    print(optimal_controls.shape)