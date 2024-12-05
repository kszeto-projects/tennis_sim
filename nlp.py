import casadi as ca
import numpy as np
from skimage.morphology import max_tree

from dynamics import D, C, g, forward_kinematics, g_casadi
def nlp(q,qdot, dt):
    T = .01
    m = 3
    n = 6

    N = int(T/dt)
    
    x = ca.MX.sym('x',n) # state space variable
    u = ca.MX.sym('u',m) # action space variable
    t = ca.MX.sym('t')

    #create casadi integrator
    ode = {'x':x,'p':u, 'ode':f(x,u)}
    opts = {'tf':T/N}
    F = ca.integrator('F','rk', ode, opts)

    X = ca.MX.sym('X',n,N+1)
    U = ca.MX.sym('U',m,N)

    x0 = ca.vertcat(q,qdot)
    cost = ca.MX(0)
    #setup constraints for joint limits, velocity limits, and torque limits
    constraints = [X[:, 0] - x0]
    for k in range(N):
        x_next = F(x0=X[:,k], p=U[:,k])['xf']
        cost += L(x_next, U[:,k], k*dt)
        constraints.append(X[:,k+1] - x_next)
    print(cost)
    constraints = ca.vertcat(*constraints)

    lbg = ca.MX.zeros(constraints.shape)
    ubg = ca.MX.zeros(constraints.shape)

    #set torque limits
    lbx = -10000*ca.MX.ones((n*(N+1) + m*N, 1))
    ubx = 10000*ca.MX.ones((n*(N+1) + m*N, 1))
    max_torque = .001
    ubx[-m*N:] = max_torque
    lbx[-m*N:] = -max_torque
    print('X shape:', X.shape)
    print('U shape:', U.shape)

    # casadi nlp object
    # 'x' = [X,U] flattened -> n*(N+1) + m*N
    # 'f' = cost
    # 'g' = constraints ->


    nlp_prob = {
        'x': ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))),
        'f': cost,
        'g': constraints
    }
    opts = {
        'ipopt': {
            'print_level': 12,  # Increase verbosity
            'max_iter': 1000,  # Increase the maximum number of iterations
            'tol': 1e-6,  # Set tolerance
            'acceptable_iter': 10,
        }
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    x0_guess = np.zeros((n*(N+1) + m*N,1))
    x0_guess[0:n*(N+1)] = np.tile(x0, N+1).flatten()[np.newaxis].T
    x0_guess[n*(N+1):] = np.tile(g(np.array(x0[:3])), N).flatten()[np.newaxis].T

    sol = solver(x0=x0_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    print(sol['x'])
    final_cost = sol['f']
    print(f'Final cost: {final_cost}')
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
    I = ca.MX.eye(3)
    O = ca.MX.zeros(3, 3)
    Dp = ca.vertcat(
        ca.horzcat(I, O),
        ca.horzcat(O, D(q))
    )

    C_x = ca.vertcat(-qdot, C(q, qdot))
    Gp = ca.vertcat(ca.MX.zeros((3, 1)), g_casadi(q))
    Up = ca.vertcat(ca.MX.zeros(3, 1), u)
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
    internal_cost = ca.norm_2(ee_pos - b(t)[0])**2 + control_weight * ca.norm_2(u)**2
    return internal_cost


def test_dynamics(q,qdot, u_app, dt):
    x0 = ca.vertcat(q, qdot)

    x = ca.MX.sym('x',6) # state space variable
    u = ca.MX.sym('u',3) # action space variable
    ode = {'x':x,'p':u,'ode': f(x,u)}
    F = ca.integrator('F','rk', ode , 0, dt)
    x_next = F(x0=x0, p=u_app)['xf']
    return x_next

if __name__ == '__main__':
    q = np.array([0, 0, 0])
    qdot = np.array([0, 0, 0])
    dt = .01
    optimal_states, optimal_controls = nlp(q, qdot, dt)
    # print(test_dynamics(q, qdot, np.zeros(3), dt))

