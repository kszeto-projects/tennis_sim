
import casadi as ca
import numpy as np

from dynamics import D, C, g, forward_kinematics, g_casadi, jacobian
from config import * #GRAV, INIT_BALL_POS, INIT_BALL_VEL, ROBOT1_BASE, ROBOT2_BASE
throw_vel = None
throw_pos = None
t_catch = None
def nlp(robot, catch, q, qdot_initial, dt, T=1.0, throw_velocity=None, throw_position=None, catch_time=None):
    m = 3
    n = 3

    # global throw_pos,throw_vel, t_catch
    # throw_pos = throw_position
    # throw_vel = throw_velocity
    # t_catch = catch_time
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
        if catch:
            cost += L(X[:,k], U[:,k], k * dt, robot)
        else:
            cost += L_throw(X[:,k], U[:,k], k * dt, robot)
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
    opts = {
        'ipopt': {
            'print_level': 0,
            'max_iter': 500,
            # 'tol': 1e-6,
        }
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob,opts)
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
        return INIT_BALL_VEL + GRAV * t

    def pos(t):
        return INIT_BALL_POS + INIT_BALL_VEL * t + 0.5 * GRAV * t ** 2

    return pos(t), vel(t)

def throw(t): # calculate reverse trajectory starting at z = -0.2
    # z_vel = INIT_BALL_VEL[2]
    # z_pos_init = INIT_BALL_POS[2]
    # z_pos_end = -0.2
    # throw_t = (-z_vel - np.sqrt(z_vel ** 2 - (4 * 0.5 * GRAV * (z_pos_init - z_pos_end)))) / (GRAV)
    # init_throw_pos = b(throw_t)[0]
    # init_throw_vel = -b(throw_t)[1]

    def t_vel(t):
        return INIT_THROW_VEL + GRAV * t

    def t_pos(t):
        return INIT_THROW_POS + INIT_THROW_VEL * t + 0.5 * GRAV * t **2

    return t_pos(t), t_vel(t)

def L(x, u, t, robot):
    if robot: # robot1
        ee_pos = forward_kinematics(x)
    else:
        ee_pos = forward_kinematics(x) + np.reshape(np.array(ROBOT2_BASE) - np.array(ROBOT1_BASE),(3,1))
    #control_weight = 0.0001
    internal_cost = ca.norm_2(ee_pos - b(t)[0]) ** 2 # + control_weight * ca.norm_2(u) ** 2
    return internal_cost

def L_throw(x, u, t, robot):
    if robot:  # robot1
        ee_pos = forward_kinematics(x)
    else:
        ee_pos = forward_kinematics(x) + np.reshape(np.array(ROBOT2_BASE) - np.array(ROBOT1_BASE), (3, 1))
    # ee_vel = jacobian(x)@u
    #
    # pos_cost = ca.norm_2(ee_pos )**2
    # internal_cost =  ca.norm_2(ee_vel - throw_vel) **2
    internal_cost = ca.norm_2(ee_pos - throw(t)[0]) ** 2
    return internal_cost

if __name__ == '__main__':
    q = np.array([0, 0.1, 0.1])
    dt = 10.0/240.0
    optimal_states, optimal_controls = nlp(q, dt)
    print(optimal_controls.shape)