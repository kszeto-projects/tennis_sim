import numpy as np

grav = np.array([0, 0, -9.81])
# initial ball pos and velocity (going to robot1)
ball_pos = np.array([1.8, 0, 0.2])
ball_vel = np.array([-2.2, 0, 3.5])
# initial ball pos and velocity (going to robot2)
# ball_pos = np.array([0.2, 0.5, 0.2])
# ball_vel = np.array([2.2, -0.5, 3.5])


# keep robot frames aligned for simplicity and to reuse forward kin
robot1_base = (0,0,0)
robot2_base = (2,0,0)

