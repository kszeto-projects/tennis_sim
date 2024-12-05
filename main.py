import time
import numpy as np
import pybullet as p
import pybullet_data
import dynamics
from numpy import sin, cos, pi
import os
import pdb
from kinematicNLP import nlp


movable_joints = None
ball_mass = 0.025
ball_rad = 0.025
do_ball_grav = True
catch_constraint = None
end_effector_link_idx = None
has_ball = False
ball = None
robot1 = None
robot2 = None

def grav_comp(q, robot):
    G = dynamics.g(q)
    if has_ball:
        g = np.array([[0], [0], [9.81]])
        torque = ball_mass * g.T @ jac(q, robot)
        G = G + torque.reshape((3,1))
    return G

def jac(q, robot):
    q1, q2, q3 = q
    zero_vec = [0, 0, 0]
    jac_t, jac_r = p.calculateJacobian(robot, end_effector_link_idx, zero_vec, [q1, q2, q3], zero_vec, zero_vec)
    return np.array(jac_t)

def generate_sphere(radius, mass=1.0):
    # create urdf text and write it to a file in tmp dir
    file_name = f"/tmp/sphere_{radius}_{mass}.urdf"
    #check if file already exists
    if os.path.exists(file_name):
        return file_name
    inertia = 2.0/5.0 * mass * radius**2
    urdf_text = f"""<?xml version="1.0" encoding="utf-8"?>
                    <robot name="sphere">
                      <link name="base_link">
                        <inertial>
                            <origin xyz="0 0 0" rpy="0 0 0" />
                            <mass value="{mass}"/>
                            <inertia ixx="{inertia}" iyy="{inertia}" izz="{inertia}" ixy="0" ixz="0" iyz="0" />
                        </inertial>
                        <visual>
                          <geometry>
                            <sphere radius="{radius}"/>
                          </geometry>
                          <material name="blue"/>
                        </visual>
                        <collision>
                          <geometry>
                            <sphere radius="{radius}"/>
                          </geometry>
                        </collision>
                      </link>
                    </robot> """

    # write to file
    with open(file_name, "w") as f:
        f.write(urdf_text)
    return file_name


def get_joint_angles(robot):
    angles = []
    for i in range(len(movable_joints)):
        angles.append(p.getJointState(robot, movable_joints[i])[0])
    return np.array(angles)


def get_joint_velocities(robot):
    velocities = []
    for i in range(len(movable_joints)):
        velocities.append(p.getJointState(robot, movable_joints[i])[1])
    return np.array(velocities)


def get_end_effector_pos(robot, end_effector_link_idx):
    ee_pose = p.getLinkState(
        robot, end_effector_link_idx, computeLinkVelocity=1)
    return np.array(ee_pose[0])

def apply_torques(robot, torques):
    for i, joint_index in enumerate(movable_joints):
        p.setJointMotorControl2(bodyIndex=robot, jointIndex=joint_index,
                                controlMode=p.TORQUE_CONTROL,
                                force=torques[i])

def set_joint_angles(robot, joint_angles):
    for i, joint_index in enumerate(movable_joints):
        p.resetJointState(robot, joint_index, joint_angles[i], 0)

def set_joint_vels(robot, joint_velocities):
    #set the joint velocity using resetJointState
    for i, joint_index in enumerate(movable_joints):
        p.resetJointState(robot, joint_index, get_joint_angles(robot)[i], joint_velocities[i])


def apply_joint_vels(robot, joint_velocities):
    for i, joint_index in enumerate(movable_joints):
        p.setJointMotorControl2(bodyIndex=robot, jointIndex=joint_index,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=joint_velocities[i])
def set_robot_angles(robot, q):
    for i, joint_index in enumerate(movable_joints):
        p.resetJointState(robot, joint_index, q[i], 0)

def set_ball_pos(pos):
    p.resetBasePositionAndOrientation(ball, pos, [0, 0, 0, 1])

def set_ball_velocity(vel):
    p.resetBaseVelocity(ball, vel, [0, 0, 0])

def get_ball_state():
    #return pos, vel
    return np.array(p.getBasePositionAndOrientation(ball)[0]), np.array(p.getBaseVelocity(ball)[0])
def get_end_effector_vel(robot, end_effector_link_idx):
    ee_vel = p.getLinkState(
        robot, end_effector_link_idx, computeLinkVelocity=1)[6]
    return np.array(ee_vel)

def get_ball_trajectory():
    #generate a trajectory for the ball to follow
    #get end effector position + velocity
    ball_pos, ball_vel = get_ball_state()
    # we can describe the velocity of the ball as a linear function of time
    grav = np.array([0, 0, -9.81])
    def vel(t):
        return ball_vel + grav*t

    def pos(t):
        return ball_pos + ball_vel*t + 0.5*grav*t**2

    vt = vel
    pt = pos

    # we can describe the position of the ball as a quadratic function of time
    # x = x0 + v0*t + 0.5*a*t^2
    return pt, vt

def plot_ball_trajectory(pt, num_points = 50):
    #plot the trajectory of the ball
    ts = np.linspace(0, 1, num_points).reshape(-1,1)
    points = pt(ts)
    p.addUserDebugPoints(points, 255*np.ones_like(points), pointSize=5)

def attempt_catch(robot, ball):
    global catch_constraint, has_ball
    if has_ball:
        return
    #get robot end effector position + velocity
    ee_pos = get_end_effector_pos(robot, end_effector_link_idx)
    ee_vel = get_end_effector_vel(robot, end_effector_link_idx)
    #get ball position + velocity
    ball_pos, ball_vel = get_ball_state()

    #if ball is close enough to end effector, and moving at a similar speed, apply a constraint to "catch" the ball
    if np.linalg.norm(ee_pos - ball_pos) < 0.025: # and np.linalg.norm(ee_vel - ball_vel) < 0.1:
        catch_constraint = p.createConstraint(robot, end_effector_link_idx, ball, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
        has_ball = True
        print(f"Ball caught! at pos: {ball_pos}, ee_pos : {ee_pos}")


def release_ball():
    global catch_constraint, has_ball

    if catch_constraint is not None:
        p.removeConstraint(catch_constraint)
        has_ball = False
        catch_constraint = None
        print("Ball released!")

def toggle_ball_grav():
    global do_ball_grav
    do_ball_grav = not do_ball_grav


def quaternion_to_rotation_matrix(quaternion): 
    x, y, z, w = quaternion
    
    # Calculate the elements of the rotation matrix
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    
    return R

## Main code
if __name__ == '__main__': 
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0., 0., -9.81)
    #plane = p.loadURDF('plane.urdf')
    robot1 = p.loadURDF('three_link.urdf', useFixedBase=True)
    robot2 = p.loadURDF('three_link.urdf', basePosition=(2, 0, 0), useFixedBase=True)
    ball = p.loadURDF(generate_sphere(ball_rad, mass=ball_mass))
    set_ball_pos([0.8, 0, 0.2])
    set_ball_velocity([-1, 0, 3])

    # get three movable joints and the end-effector link's index
    num_joints = p.getNumJoints(robot1)
    movable_joints = []
    end_effector_link_idx = None
    for idx in range(num_joints):
        info = p.getJointInfo(robot1, idx)
        print('Joint {}: {}'.format(idx, info))

        joint_type = info[2]
        if joint_type != p.JOINT_FIXED:
            movable_joints.append(idx)

        link_name = info[12].decode('utf-8')
        if link_name == 'end_effector':
            end_effector_link_idx = idx
            
    for joint_index in movable_joints:
        p.changeDynamics(robot1, joint_index, linearDamping=0, angularDamping=0)

    # Set joint control mode to make the joints free to move (no motor control)
    for joint_index in movable_joints:
        p.setJointMotorControl2(bodyIndex=robot1,
                                jointIndex=joint_index,
                                controlMode=p.VELOCITY_CONTROL,
                                force=0)  # Ensure no motor is controlling the joint
        

    # Set the initial joint angles
    set_joint_angles(robot1, [0, 0, 0])
    set_joint_angles(robot2, [np.pi, 0, 0])

    # TODO: Your code here
    
    locations = np.array([[0.2,0.2,0.2],[0.1,0.2,0.2],[0.1,-0.2,0.2],[0.2,-0.2,0.2],[0.2,0.2,0.2]])
    
    # for location in locations:
    #     p.stepSimulation()

    # hold Ctrl and use the mouse to rotate, pan, or zoom
    last_q = np.zeros(3)
    last_qdot = np.zeros(3)
    dt = 1. / 240.
    optimal_states, optimal_controls = nlp([0, 0, 0],  dt, T=1.0)
    # print(p.getPhysicsEngineParameters()['fixedTimeStep'])
    for step in range(10000):
        q1 = get_joint_angles(robot1)
        q2 = get_joint_angles(robot2)
        # ctrl_idx = _ // int(p.getPhysicsEngineParameters()['fixedTimeStep'] / dt)
        # print(f"ctrl_idx: {ctrl_idx}")
        if step < len(optimal_controls):
            # set_robot_angles(robot1, optimal_states[step])
            # set_joint_vels(robot1, optimal_controls[step])
            apply_joint_vels(robot1, optimal_controls[step])



        if do_ball_grav:
            pt, vt = get_ball_trajectory()
            plot_ball_trajectory(pt)
            toggle_ball_grav()

        # if not do_ball_grav:
        #     p.applyExternalForce(ball, -1, [0, 0, ball_mass * 9.81], [0, 0, 0], p.LINK_FRAME)
        # print("robot1_ee_pos: " + str(get_end_effector_pos(robot1, end_effector_link_idx)))
        # print("robot2_ee_pos: " + str(get_end_effector_pos(robot2, end_effector_link_idx)))
        # print("q1, q1dot = " + str(q1), str(get_joint_velocities(robot1)))
        # res = test_dynamics(q1, get_joint_velocities(robot1), grav_comp(q1, robot1) + .0001, dt)
        # print("diff: " + str(q1 - last_q) + " " + str(get_joint_velocities(robot1) - last_qdot))
        # last_q = res[:3]
        # last_qdot = res[3:]

        attempt_catch(robot1, ball)
        p.stepSimulation()
        time.sleep(10./240.)

    p.disconnect()
