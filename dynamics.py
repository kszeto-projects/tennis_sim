import numpy as np
import casadi as ca
from casadi import sin, cos, pi

def D(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    ell1 = .2
    ell2 = .2
    ell3 = .2
    r1 = .02
    r2 = .02
    r3 = .02
    rho = 300

    D_arr = ca.vertcat(ca.horzcat(pi*rho*(0.5*ell1*r1**4 - 0.0833333333333333*ell2**3*r2**2*sin(q3)*sin(q2 + q3) + 0.0833333333333333*ell2**3*r2**2*cos(q2) + 0.25*ell2**3*r2**2*cos(q3)**2 + 1.0*ell2**2*ell3*r3**2*cos(q3)**2 - 0.166666666666667*ell2*ell3**2*r2**2*sin(q2)*sin(q3)*cos(q2 + q3) + 0.0833333333333333*ell2*ell3**2*r2**2*cos(q2)**2 + 0.0833333333333333*ell2*ell3**2*r2**2*cos(q3)**2 - 0.0833333333333333*ell2*ell3**2*r2**2 + 1.0*ell2*ell3**2*r3**2*cos(q3)*cos(q2 + q3) + 0.25*ell2*r2**4*sin(q3)*sin(q2 + q3) + 0.25*ell2*r2**4*cos(q2) + 0.5*ell2*r2**2*r3**2*sin(q2)*sin(q3)*cos(q2 + q3) - 0.25*ell2*r2**2*r3**2*cos(q2)**2 - 0.25*ell2*r2**2*r3**2*cos(q3)**2 + 0.75*ell2*r2**2*r3**2 - 0.5*ell3**3*r3**2*sin(q2)*sin(q3)*cos(q2 + q3) + 0.25*ell3**3*r3**2*cos(q2)**2 + 0.25*ell3**3*r3**2*cos(q3)**2 - 0.25*ell3**3*r3**2), 0.0, 0.0),
                   ca.horzcat(0.0, pi*rho*(0.333333333333333*ell2**3*r2**2 + 1.0*ell2**2*ell3*r3**2 + 0.0833333333333333*ell2*ell3**2*r2**2 + 1.0*ell2*ell3**2*r3**2*cos(q2) + 0.25*ell2*r2**4 + 0.25*ell2*r2**2*r3**2 + 0.25*ell3**3*r3**2), pi*rho*(0.0833333333333333*ell2*ell3**2*r2**2 + 0.5*ell2*ell3**2*r3**2*cos(q2) + 0.25*ell2*r2**2*r3**2 + 0.25*ell3**3*r3**2)),
                    ca.horzcat(0.0, pi*rho*(0.0833333333333333*ell2*ell3**2*r2**2 + 0.5*ell2*ell3**2*r3**2*cos(q2) + 0.25*ell2*r2**2*r3**2 + 0.25*ell3**3*r3**2), pi*rho*(0.0833333333333333*ell2*ell3**2*r2**2 + 0.25*ell2*r2**2*r3**2 + 0.25*ell3**3*r3**2)))

    return D_arr

def C(q, qdot):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q1dot = qdot[0]
    q2dot = qdot[1]
    q3dot = qdot[2]
    ell1 = .2
    ell2 = .2
    ell3 = .2
    r1 = .02
    r2 = .02
    r3 = .02
    rho = 300
    g = 9.81
    return ca.vertcat(ca.horzcat(q1dot*pi*rho*(-0.0416666666666667*q3dot*ell2**3*r2**2*sin(q2) - 0.0416666666666667*q3dot*ell2**3*r2**2*sin(q2 + 2*q3) - 0.0833333333333333*q3dot*ell2*ell3**2*r2**2*sin(2*q2 + 2*q3) - 0.5*q3dot*ell2*ell3**2*r3**2*sin(q2) - 0.5*q3dot*ell2*ell3**2*r3**2*sin(q2 + 2*q3) - 0.375*q3dot*ell2*r2**4*sin(q2) + 0.125*q3dot*ell2*r2**4*sin(q2 + 2*q3) + 0.25*q3dot*ell2*r2**2*r3**2*sin(2*q2 + 2*q3) - 0.25*q3dot*ell3**3*r3**2*sin(2*q2 + 2*q3) - 0.25*q2dot*ell2**3*r2**2*sin(2*q3) - 0.0833333333333333*q2dot*ell2**3*r2**2*sin(q2 + 2*q3) - 1.0*q2dot*ell2**2*ell3*r3**2*sin(2*q3) - 0.0833333333333333*q2dot*ell2*ell3**2*r2**2*sin(2*q2 + 2*q3) - 1.0*q2dot*ell2*ell3**2*r3**2*sin(q2 + 2*q3) + 0.25*q2dot*ell2*r2**4*sin(q2 + 2*q3) + 0.25*q2dot*ell2*r2**2*r3**2*sin(2*q2 + 2*q3) - 0.25*q2dot*ell3**3*r3**2*sin(2*q2 + 2*q3))),
                      ca.horzcat(pi*rho*(-0.5*q3dot**2*ell2*ell3**2*r3**2*sin(q2) - 1.0*q3dot*q2dot*ell2*ell3**2*r3**2*sin(q2) + 0.125*q1dot**2*ell2**3*r2**2*sin(2*q3) + 0.0416666666666667*q1dot**2*ell2**3*r2**2*sin(q2 + 2*q3) + 0.5*q1dot**2*ell2**2*ell3*r3**2*sin(2*q3) + 0.0416666666666667*q1dot**2*ell2*ell3**2*r2**2*sin(2*q2 + 2*q3) + 0.5*q1dot**2*ell2*ell3**2*r3**2*sin(q2 + 2*q3) - 0.125*q1dot**2*ell2*r2**4*sin(q2 + 2*q3) - 0.125*q1dot**2*ell2*r2**2*r3**2*sin(2*q2 + 2*q3) + 0.125*q1dot**2*ell3**3*r3**2*sin(2*q2 + 2*q3))),
                    ca.horzcat(pi*rho*(0.5*q2dot**2*ell2*ell3**2*r3**2*sin(q2) + 0.0208333333333333*q1dot**2*ell2**3*r2**2*sin(q2) + 0.0208333333333333*q1dot**2*ell2**3*r2**2*sin(q2 + 2*q3) + 0.0416666666666667*q1dot**2*ell2*ell3**2*r2**2*sin(2*q2 + 2*q3) + 0.25*q1dot**2*ell2*ell3**2*r3**2*sin(q2) + 0.25*q1dot**2*ell2*ell3**2*r3**2*sin(q2 + 2*q3) + 0.1875*q1dot**2*ell2*r2**4*sin(q2) - 0.0625*q1dot**2*ell2*r2**4*sin(q2 + 2*q3) - 0.125*q1dot**2*ell2*r2**2*r3**2*sin(2*q2 + 2*q3) + 0.125*q1dot**2*ell3**3*r3**2*sin(2*q2 + 2*q3))))


def g_casadi(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    ell1 = .2
    ell2 = .2
    ell3 = .2
    r1 = .02
    r2 = .02
    r3 = .02
    rho = 300.0
    g = 9.81

    G = ca.vertcat(0.0, (1.0 / 2.0) * pi * g * rho * (
        ell2 ** 2 * r2 ** 2 * cos(q2) + ell3 * r3 ** 2 * (2 * ell2 * cos(q2) + ell3 * cos(q2 + q3))),
        (1.0 / 2.0) * pi * ell3 ** 2 * g * r3 ** 2 * rho * cos(q2 + q3))
    return G
def g(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    ell1 = .2
    ell2 = .2
    ell3 = .2
    r1 = .02
    r2 = .02
    r3 = .02
    rho = 300.0
    g = 9.81
    pi = np.pi

    G = np.array([[0, (1.0 / 2.0) * pi * g * rho * (
        ell2 ** 2 * r2 ** 2 * cos(q2) + ell3 * r3 ** 2 * (2 * ell2 * cos(q2) + ell3 * cos(q2 + q3))),
        (1.0 / 2.0) * pi * ell3 ** 2 * g * r3 ** 2 * rho * cos(q2 + q3)]]).reshape((3, 1))

    return G

def forward_kinematics(q):
    q1, q2, q3 = q[0], q[1], q[2]
    ell1 = .2
    ell2 = .2
    ell3 = .2
    return ca.vertcat(
        (ell2 * cos(q2) + ell3 * cos(q2 + q3)) * cos(q1), (ell2 * cos(q2) + ell3 * cos(q2 + q3)) * sin(q1),
         ell1 + ell2 * sin(q2) + ell3 * sin(q2 + q3))
