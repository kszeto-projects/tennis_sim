import casadi as ca
import numpy as np

from dynamics import D, C, g, forward_kinematics, g_casadi, jacobian
from config import * #GRAV, INIT_BALL_POS, INIT_BALL_VEL, ROBOT1_BASE, ROBOT2_BASE
throw_vel = None
throw_pos = None
t_catch = None

def nlp_combined(is_throw_robot1, q, qdot_initial, qdot_catch, q_catch, goal, dt, T = 1.0, T_final = 0.4):
    m = 3
    n = 3

    # global throw_pos,throw_vel, t_catch
    # throw_pos = throw_position
    # throw_vel = throw_velocity
    # t_catch = catch_time
    # T = ca.MX.sym('T') # can try to make T a part of the optimization variables
    # N = ca.floor(T / dt)
    N = int(T/dt)
    N_catch = N + int(T_final/dt)
    Xthrow_sym = ca.MX.sym('Xt', n * (N + 1), 1)
    Uthrow_sym = ca.MX.sym('Ut', m * N, 1)

    Xcatch_sym = ca.MX.sym('Xc', n * (N_catch + 1), 1)
    Ucatch_sym = ca.MX.sym('Uc', m * N_catch, 1)

    Xt = Xthrow_sym.reshape((n, N + 1))
    Ut = Uthrow_sym.reshape((m, N))
    Xc = Xcatch_sym.reshape((n, N + 1))
    Uc = Ucatch_sym.reshape((m, N))
    X_T = Xt[:,-1]
    U_T = Ut[:,-1]
    g = np.zeros((3,1))
    g[2] = -0.5*9.81
    ca_g = ca.DM(g)
    ca_end = ca.DM(goal)
    if is_throw_robot1:
        x_pos = forward_kinematics(X_T)
    else:
        x_pos = forward_kinematics(X_T) + np.reshape(np.array(ROBOT2_BASE) - np.array(ROBOT1_BASE), (3, 1))
    # offset by difference between robot bases

    v = x_pos + T_final * jacobian(X_T) @ U_T + (T_final**2) * ca_g - ca_end

    x0 = ca.vertcat(q)
    x0_catch = ca.vertcat(q_catch)
    cost = ca.MX(0)
    constraints = [Xt[:, 0] - x0]
    for k in range(N):
        x_next = Xt[:, k] + Ut[:, k] * dt
        constraints.append(Xt[:, k + 1] - x_next)
        cost += L_throw(Xt[:,k], Ut[:,k], k * dt, goal, is_throw_robot1)
        # cost += L_comb(Xc[:,k], Uc[:,k], k * dt, not is_throw_robot1)

    catch_constrains = [Xc[:, 0] - x0_catch]
    for k in range(N_catch):
        x_next_catch = Xc[:, k] + Uc[:, k] * dt
        catch_constrains.append(Xc[:, k + 1] - x_next_catch)

    cost += (ca.norm_2(v))**2

    if not is_throw_robot1:
        x_catch_pos = forward_kinematics(Xc[:,-1])
    else:
        x_catch_pos = forward_kinematics(Xc[:,-1]) + np.reshape(np.array(ROBOT2_BASE) - np.array(ROBOT1_BASE), (3, 1))
    cost += (ca.norm_2(x_catch_pos - goal))**2

    # add acceleration_constraints
    constraints.append((Ut[:, 0] - qdot_initial))
    catch_constrains.append((Uc[:, 0] - qdot_catch))
    for k in range(N - 1):
        constraints.append((Ut[:, k + 1] - Ut[:, k]) / dt)

    for k in range(N_catch-1):
        catch_constrains.append((Uc[:, k + 1] - Uc[:, k]) / dt)

    constraints = ca.vertcat(*constraints).reshape((-1, 1))
    catch_constrains = ca.vertcat(*catch_constrains).reshape((-1, 1))

    lbg = np.zeros(constraints.shape)
    ubg = np.zeros(constraints.shape)
    max_joint_accel = 500
    lbg[-m * N:] = -max_joint_accel
    ubg[-m * N:] = max_joint_accel
    lbg[-m * int(N*.15): ] = -5
    ubg[-m * int(N*.15): ] = 5


    ubx = np.inf * np.ones((n * (N + 1) + m * N, 1))
    lbx = -np.inf * np.ones((n * (N + 1) + m * N, 1))
    max_joint_vel = 500
    ubx[-m * N:] = max_joint_vel
    lbx[-m * N:] = -max_joint_vel

    lbg_catch = np.zeros(catch_constrains.shape)
    ubg_catch = np.zeros(catch_constrains.shape)
    lbg_catch[-m * N_catch:] = -max_joint_accel
    ubg_catch[-m * N_catch:] = max_joint_accel

    ubx_catch = np.inf * np.ones((n * (N_catch + 1) + m * N_catch, 1))
    lbx_catch = -np.inf * np.ones((n * (N_catch + 1) + m * N_catch, 1))
    ubx_catch[-m * N_catch:] = max_joint_vel
    lbx_catch[-m * N_catch:] = -max_joint_vel

    casadi_x = ca.vertcat(Xthrow_sym, Uthrow_sym, Xcatch_sym, Ucatch_sym)

    nlp_prob = {
        'x': casadi_x,
        'f': cost,
        'g': constraints
    }
    opts = {
        'ipopt': {
            # 'print_level': 0,
            'max_iter': 1000,
            'tol': 1e-6,
        }
    }


    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
    # print('solver:', solver)
    x0_guess = np.zeros((n * (N + 1) + m * N, 1))
    x0_guess[0:n * (N + 1)] = np.tile(q, N + 1).reshape(-1, 1)
    x0_guess[n * (N + 1):] = np.tile(np.array([0.0, 0.01, 0.01]), N).flatten()[np.newaxis].T

    x0_guess_catch = np.zeros((n * (N_catch + 1) + m * N_catch, 1))
    x0_guess_catch[0:n * (N_catch + 1)] = np.tile(q, N_catch + 1).reshape(-1, 1)
    x0_guess_catch[n * (N_catch + 1):] = np.tile(np.array([0.0, 0.01, 0.01]), N_catch).flatten()[np.newaxis].T

    sol = solver(x0=ca.vertcat(x0_guess, x0_guess_catch), lbx=ca.vertcat(lbx, lbx_catch), ubx=ca.vertcat(ubx, ubx_catch), lbg=ca.vertcat(lbg, lbg_catch), ubg=ca.vertcat(ubg, ubg_catch))
    final_cost = sol['f']
    print(f'Final cost: {final_cost}')
    optimal_states_throw = np.array(sol['x'][0:n * (N + 1)].reshape((n, N + 1))).T
    optimal_controls_throw = np.array(sol['x'][n * (N + 1): n * (N + 1) + m*N].reshape((m, N))).T

    optimal_states_catch = np.array(sol['x'][0:n * (N + 1)].reshape((n, N + 1))).T
    optimal_controls_catch = np.array(sol['x'][n * (N + 1): n * (N + 1) + m*N].reshape((m, N))).T

    return optimal_states, optimal_controls



def nlp(robot, q, qdot_initial, dt, T=1.0, init_ball_pos=INIT_BALL_POS, init_ball_vel=INIT_BALL_VEL):
    m = 3
    n = 3


    # Robot 1
    #
    #

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
        cost += L(X[:,k], U[:,k], k * dt, robot, init_ball_pos=init_ball_pos, init_ball_vel=init_ball_vel)
        # else:
        #     cost += L_throw(X[:,k], U[:,k], k * dt, robot)
        #     cost = Q(X[:, -1], U[:, -1])
        constraints.append(X[:,k+1] - x_next)

        
    #add acceleration_constraints
    constraints.append((U[:,0] - qdot_initial))
    for k in range(N-1):
        constraints.append((U[:,k+1] - U[:,k])/dt)
    constraints = ca.vertcat(*constraints).reshape((-1,1))

    lbg = np.zeros(constraints.shape)
    ubg = np.zeros(constraints.shape)
    max_joint_accel = 500
    lbg[-m*N:] = -max_joint_accel
    ubg[-m*N:] = max_joint_accel

    lbx = -np.inf * np.ones((n * (N + 1) + m * N, 1))
    ubx = np.inf * np.ones((n * (N + 1) + m * N, 1))
    max_joint_vel = 500
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
            # 'print_level': 0,
            'max_iter': 1000,
            'tol': 1e-6,
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

'''Throwing optimal control problem formulation: slightly different. trying to use planning horizon as optimization variable'''
def nlp_throw(is_robot1, q, qdot_initial, goal, dt, T = 1.0, T_final = 0.4):
    m = 3
    n = 3

    # global throw_pos,throw_vel, t_catch
    # throw_pos = throw_position
    # throw_vel = throw_velocity
    # t_catch = catch_time
    # T = ca.MX.sym('T') # can try to make T a part of the optimization variables
    # N = ca.floor(T / dt)
    N = int(T/dt)
    Xsym = ca.MX.sym('X', n * (N + 1), 1)
    Usym = ca.MX.sym('U', m * N, 1)

    X = Xsym.reshape((n, N + 1))
    U = Usym.reshape((m, N))
    X_T = X[:,-1]
    U_T = U[:,-1]
    g = np.zeros((3,1))
    g[2] = -0.5*9.81
    ca_g = ca.DM(g)
    ca_end = ca.DM(goal)
    if is_robot1:
        x_pos = forward_kinematics(X_T)
    else:
        x_pos = forward_kinematics(X_T) + np.reshape(np.array(ROBOT2_BASE) - np.array(ROBOT1_BASE), (3, 1))
    # offset by difference between robot bases

    v = x_pos + T_final * jacobian(X_T) @ U_T + (T_final**2) * ca_g - ca_end

    x0 = ca.vertcat(q)
    cost = ca.MX(0)
    constraints = [X[:, 0] - x0]
    for k in range(N):
        x_next = X[:, k] + U[:, k] * dt
        constraints.append(X[:, k + 1] - x_next)
        cost += L_throw(X[:,k], U[:,k], k * dt, goal, is_robot1)
    cost += (ca.norm_2(v))**2 #Q(X[:, -1], U[:, -1], goal)

    # add acceleration_constraints
    constraints.append((U[:, 0] - qdot_initial))
    for k in range(N - 1):
        constraints.append((U[:, k + 1] - U[:, k]) / dt)
    # constraints.append(ca.norm_2(forward_kinematics(X_T) + 0.8 * jacobian(X_T) @ U_T + 0.64 * ca_g - ca_end))
    constraints = ca.vertcat(*constraints).reshape((-1, 1))

    lbg = np.zeros(constraints.shape)
    ubg = np.zeros(constraints.shape)
    max_joint_accel = 500
    lbg[-m * N:] = -max_joint_accel
    ubg[-m * N:] = max_joint_accel
    # lbg[-m * N:-1] = -max_joint_accel
    # ubg[-m * N:-1] = max_joint_accel
    # lbg[-1] = -0.001
    # ubg[-1] = 0.001

    ubx = np.inf * np.ones((n * (N + 1) + m * N, 1))
    lbx = -np.inf * np.ones((n * (N + 1) + m * N, 1))
    max_joint_vel = 500
    ubx[-m * N:] = max_joint_vel
    lbx[-m * N:] = -max_joint_vel
    # ubx[-1] = 5
    # lbx[-1] = 0.5

    # stack x and u into a single vector
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
            # 'print_level': 0,
            'max_iter': 1000,
            'tol': 1e-6,
        }
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
    # print('solver:', solver)
    x0_guess = np.zeros((n * (N + 1) + m * N, 1))
    x0_guess[0:n * (N + 1)] = np.tile(q, N + 1).reshape(-1, 1)
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

def b(t, init_ball_pos=INIT_BALL_POS, init_ball_vel=INIT_BALL_VEL):

    def vel(t):
        return init_ball_vel + GRAV * t

    def pos(t):
        return init_ball_pos + init_ball_vel * t + 0.5 * GRAV * t ** 2

    return pos(t), vel(t)

def L(x, u, t, is_robot1, init_ball_pos=INIT_BALL_POS, init_ball_vel=INIT_BALL_VEL):
    if is_robot1: # robot1
        ee_pos = forward_kinematics(x)
    else:
        ee_pos = forward_kinematics(x) + np.reshape(np.array(ROBOT2_BASE) - np.array(ROBOT1_BASE),(3,1))
    #control_weight = 0.0001
    internal_cost = ca.norm_2(ee_pos - b(t, init_ball_pos, init_ball_vel)[0]) ** 2 # + control_weight * ca.norm_2(u) ** 2
    return internal_cost

def L_throw(x, u, t, goal, is_robot1):
    # if is_robot1:
    #     ee_pos = forward_kinematics(x)
    # else:
    #     ee_pos = forward_kinematics(x) + np.reshape(np.array(ROBOT2_BASE) - np.array(ROBOT1_BASE), (3, 1))
    # ee_vel = jacobian(x)@u
    # calculate trajectory
    # cost: want to reach state pos and vel that results in trajectory ending at desired x,y,z (robot2 ee)
    # path planning to reasonable position?
    # if initially far from reasonable trajectory, high cost
    # lower penalty in beginning and end of control sequence?
    # guide end-effector towards goal?
    # minimize end-effector travel distance?
    # ee_unit_vel = ee_vel[0:2]/ca.norm_2(ee_vel)
    # goal_unit_vel = (ee_pos[0:2] + goal[0:2]) / ca.norm_2(ee_pos[0:2] + goal[0:2])

    # internal_cost = 0.001 * ca.norm_2(x[2]) #(t**2*(1.0/24))*(ca.norm_2(ee_unit_vel - goal_unit_vel)) +
    # internal_cost = 0.01 * ca.norm_2(u)**2
    internal_cost = 0
    #0.05 * ca.norm_2(ee_vel) ** 2
    return internal_cost


if __name__ == '__main__':
    q = np.array([0, 0.1, 0.1])
    dt = 10.0/240.0
    optimal_states, optimal_controls = nlp(q, dt)
    print(optimal_controls.shape)