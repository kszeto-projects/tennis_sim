import numpy as np

GRAV = np.array([0, 0, -9.81])
# initial ball pos and velocity (going to robot1)
INIT_BALL_POS = np.array([1.8, 0, 0.2])
INIT_BALL_VEL = np.array([-2.2, 0, 3.5])
# initial ball pos and velocity (going to robot2)
# ball_pos = np.array([0.2, 0.5, 0.2])
# ball_vel = np.array([2.2, -0.5, 3.5])
THROW_TIME = 0.8134
INIT_THROW_VEL = np.array([2.2, 0, 4.48534])
INIT_THROW_POS = np.array([0.0092, 0, -0.20103338])


# keep robot frames aligned for simplicity and to reuse forward kin
ROBOT1_BASE = (0,0,0)
ROBOT2_BASE = (2,0,0)

