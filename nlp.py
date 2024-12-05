import casadi as ca
import numpy as np
from skimage.morphology import max_tree

from dynamics import D, C, g, forward_kinematics, g_casadi
from main import get_ball_trajectory
def nlp(q,qdot, dt):
    T = 1
    m = 3
    n = 6

    N = int(T/dt)
    
    x = ca.SX.sym('x',n) # state space variable
    u = ca.SX.sym('u',m) # action space variable
    t = ca.SX.sym('t')

    #create casadi integrator
    ode = {'x':x,'p':u,'ode':f(x,u)}
    opts = {'tf':T/N}
    F = ca.integrator('F','idas', ode ,opts)

    X = ca.SX.sym('X',n,N+1)
    U = ca.SX.sym('U',m,N)

    x0 = ca.vertcat(q,qdot)
    cost = 0
    #setup constraints for joint limits, velocity limits, and torque limits
    constraints = [X[:, 0] - x0]
    for k in range(N):
        x_next = F(x0 = X[:,k], p = U[:,k])['xf']
        cost += L(X[:,k], U[:,k], k*dt)
        constraints.append(X[:,k+1] - x_next)

    constraints = ca.vertcat(*constraints)

    lbg = ca.SX.zeros(constraints.shape)
    ubg = ca.MSXX.zeros(constraints.shape)

    #set torque limits
    lbx = -10000*ca.SX.ones((n*(N+1) + m*N, 1))
    ubx = 10000*ca.SX.ones((n*(N+1) + m*N, 1))
    max_torque = 10
    ubx[-m*N:] = max_torque
    lbx[-m*N:] = -max_torque
    print('X shape:', X.shape)
    print('U shape:', U.shape)

    # casadi nlp object
    # 'x' = [X,U] flattened -> n*(N+1) + m*N
    # 'f' = cost
    # 'g' = constraints ->


    nlp = {
        'x': ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))),
        'f': cost,
        'g': constraints
    }

    solver = ca.nlpsol('solver', 'ipopt', nlp)

    x0_guess = np.zeros((n*(N+1) + m*N,1))

    sol = solver(x0 = x0_guess, lbx = lbx, ubx = ubx, lbg = lbg, ubg = ubg)
    print(sol['x'])
    optimal_states = sol['x'][0:n*(N+1)].reshape((n,N+1))
    optimal_controls = sol['x'][n*(N+1):].reshape((m,N))
    print('optimal states:', optimal_states)
    print('optimal controls:', optimal_controls)
    return optimal_states, optimal_controls

    
def f(x,u):
    '''
    :param x: [q.T,qdot.T].T
    :param u: [tau.T].T
    :return: [qdot.T,qddot.T].T
    '''
    q = x[0:3]
    qdot = x[3:6]
    I = ca.SX.eye(3)
    O = ca.SX.zeros(3, 3)
    Dp = ca.vertcat(
        ca.horzcat(I, O),
        ca.horzcat(O, D(q))
    )

    C_x = ca.vertcat(-qdot, C(q, qdot))
    Gp = ca.vertcat(ca.SX.zeros((3, 1)), g_casadi(q))
    Up = ca.vertcat(ca.SX.zeros(3, 1), u)
    xdot = ca.pinv(Dp)@(-C_x - Gp + Up)
    return xdot

def b(t):
    grav = np.array([0, 0, -9.81])
    ball_pos = np.array([0.8, 0, 0.2])
    ball_vel = np.array([-1, 0, 3])

    def vel(t):
        return ball_vel + grav * t

    def pos(t):
        return ball_pos + ball_vel * t + 0.5 * grav * t ** 2

    # pt, vt = get_ball_trajectory()
    return pos(t), vel(t)

def L(x, u, t):
    #todo maybe include a cost for velocity also
    ee_pos = forward_kinematics(x[0:3])
    control_weight = 0.2
    cost = ca.norm_2(ee_pos - b(t)[0])**2 + control_weight * ca.norm_2(u)**2
    return cost

# if __name__ == '__main__':
#     # print(f(np.zeros(6), np.zeros(3)))
#     q = np.array([0, 0, 0])
#     qdot = np.array([0, 0, 0])
#     dt = 0.01
#     optimal_states, optimal_controls = nlp(q, qdot, dt)


if __name__ == '__main__':
    # test_dynamics()
    zeros = ca.SX.zeros(3, 1)
    ones = ca.SX.ones(3, 1)

    T = 1
    m = 3
    n = 6
    dt = .01
    N = int(T / dt)

    x = ca.SX.sym('x', n)  # state space variable
    u = ca.SX.sym('u', m)  # action space variable
    t = ca.SX.sym('t')

    # create casadi integrator
    print(f(x,u))
    ode = {'x': x, 'p': u, 'ode': f(x, u)}
    opts = {'tf': T / N}
    F = ca.integrator('F', 'idas', ode, opts)
    r = F(x0=ca.vertcat(zeros, zeros), p=ca.vertcat(zeros))['xf']

