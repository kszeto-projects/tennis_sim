import casadi as ca

def nlp(q,qdot,ball_traj):
    N = 50
    T = 10
    m = 3
    n = 6
    
    x = ca.MX.sym('x',n) # state space variable
    u = ca.MX.sym('u',m) # action space variable
    t = ca.MX.sym('t')
    
def f(x,u):
    # TODO: implement motion model dynamics
    pass

def b(t):
    # TODO:
    pass

def L(x,u):