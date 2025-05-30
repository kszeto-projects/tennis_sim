import time
import numpy as np
import pybullet as p
import pybullet_data
import dynamics
from numpy import sin, cos, pi
import os
import pdb
from kinematicNLP import nlp, nlp_throw
from config import GRAV, INIT_BALL_POS, INIT_BALL_VEL, ROBOT1_BASE, ROBOT2_BASE
import matplotlib.pyplot as plt

plot_id = None
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
did_calc_throw = False
did_calc_catch = False
throw_controls = None
catch_controls = None
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

def generate_sphere(radius, mass=1.0, color_rgba=[1, 0, 0, 0.7]):
    # create urdf text and write it to a file in tmp dir
    color = " ".join(map(str, color_rgba))
    print(color)
    color_str= "-".join(map(str, color_rgba))
    file_name = f"/tmp/sphere_{radius}_{mass}_{color_str}.urdf"
    #check if file already exists
    if os.path.exists(file_name):
        return file_name
    inertia = 2.0/5.0 * mass * radius**2
    urdf_text = f"""<?xml version="1.0" encoding="utf-8"?>
                    <robot name="sphere">
                        <material name="color">
                            <color rgba="{color}"/>
                        </material>
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
                              <material name="color"/>
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
    for i in range(len(movable_joints[robot])):
        angles.append(p.getJointState(robot, movable_joints[robot][i])[0])
    return np.array(angles)


def get_joint_velocities(robot):
    velocities = []
    for i in range(len(movable_joints[robot])):
        velocities.append(p.getJointState(robot, movable_joints[robot][i])[1])
    return np.array(velocities)


def get_end_effector_pos(robot, end_effector_link_idx):
    ee_pose = p.getLinkState(
        robot, end_effector_link_idx, computeLinkVelocity=1)
    return np.array(ee_pose[0])

def apply_torques(robot, torques):
    for i, joint_index in enumerate(movable_joints[robot]):
        p.setJointMotorControl2(bodyIndex=robot, jointIndex=joint_index,
                                controlMode=p.TORQUE_CONTROL,
                                force=torques[i])

def set_joint_angles(robot, joint_angles):
    for i, joint_index in enumerate(movable_joints[robot]):
        p.resetJointState(robot, joint_index, joint_angles[i], 0)

def set_joint_vels(robot, joint_velocities):
    #set the joint velocity using resetJointState
    for i, joint_index in enumerate(movable_joints[robot]):
        p.resetJointState(robot, joint_index, get_joint_angles(robot)[i], joint_velocities[i])


def apply_joint_vels(robot, joint_velocities):
    for i, joint_index in enumerate(movable_joints[robot]):
        p.setJointMotorControl2(bodyIndex=robot, jointIndex=joint_index,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=joint_velocities[i])
def set_robot_angles(robot, q):
    for i, joint_index in enumerate(movable_joints[robot]):
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
    #grav = np.array([0, 0, -9.81])
    def vel(t):
        return ball_vel + GRAV*t

    def pos(t):
        return ball_pos + ball_vel*t + 0.5*GRAV*t**2

    vt = vel
    pt = pos

    # we can describe the position of the ball as a quadratic function of time
    # x = x0 + v0*t + 0.5*a*t^2
    return pt, vt

def plot_ball_trajectory(pt, num_points = 50):
    global plot_id
    #plot the trajectory of the ball
    ts = np.linspace(0, 1, num_points).reshape(-1,1)
    points = pt(ts)
    if plot_id is not None:
        plot_id = p.addUserDebugPoints(points, 255*np.ones_like(points), pointSize=3, replaceItemUniqueId=plot_id)
    else:
        plot_id = p.addUserDebugPoints(points, 255*np.ones_like(points), pointSize=3)
    return plot_id

def attempt_catch(robot, ball):
    global catch_constraint, has_ball
    if has_ball:
        return
    #get robot end effector position + velocity
    ee_pos = get_end_effector_pos(robot, end_effector_link_idx)
    ee_vel = get_end_effector_vel(robot, end_effector_link_idx)
    #get ball position + velocity
    cur_ball_pos, cur_ball_vel = get_ball_state()

    #if ball is close enough to end effector, and moving at a similar speed, apply a constraint to "catch" the ball
    if np.linalg.norm(ee_pos - cur_ball_pos) < 0.075: # and np.linalg.norm(ee_vel - cur_ball_vel) < 0.1:
        catch_constraint = p.createConstraint(robot, end_effector_link_idx, ball, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
        # p.stepSimulation()
        has_ball = True
        print(f"Ball caught! at pos: {cur_ball_pos}, ee_pos : {ee_pos}")
        return (cur_ball_pos, cur_ball_vel)
    
    return None, None


def release_ball():
    global catch_constraint, has_ball

    if catch_constraint is not None:
        p.removeConstraint(catch_constraint)
        has_ball = False
        catch_constraint = None
        print("Ball released!")


def quaternion_to_rotation_matrix(quaternion): 
    x, y, z, w = quaternion
    
    # Calculate the elements of the rotation matrix
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    
    return R

def catch_phase(robot, step, enable_mpc=False, mpc_step_interval = 50):
    global did_calc_catch, catch_controls, last_computation_step
    if not did_calc_catch:
        last_computation_step = step
        print("Calculating catch for robot ", robot)
        q_robot = get_joint_angles(robot)
        qdot_robot = get_joint_velocities(robot)
        ball_pos, ball_vel = get_ball_state()
        catch_states, catch_controls = nlp(robot==robot1, q_robot, qdot_robot, dt, T=.85, init_ball_pos=ball_pos, init_ball_vel=ball_vel)
        did_calc_catch = True
    if step-last_computation_step > mpc_step_interval and enable_mpc:
            did_calc_catch = False
            plot_ball_trajectory(get_ball_trajectory()[0])
    if step < len(catch_controls):
        apply_joint_vels(robot, catch_controls[step-last_computation_step])
        apply_joint_vels(robot2 if robot == robot1 else robot1, np.zeros(3))
    if not has_ball:
        attempt_catch(robot, ball)

    if has_ball:
        did_calc_catch = False
        apply_joint_vels(robot, np.zeros(3))

        return True

    return False

def throw_phase(robot, step):
    global did_calc_throw
    global throw_controls

    if not did_calc_throw:
        throwing_robot = robot
        catching_robot = robot2 if robot == robot1 else robot1
        throwing_q = get_joint_angles(throwing_robot)
        catching_q = get_joint_angles(catching_robot)
        throwing_qdot = get_joint_velocities(throwing_robot)
        catching_qdot = get_joint_velocities(catching_robot)
        goal_pt = get_end_effector_pos(catching_robot, end_effector_link_idx-2) #+ np.random.normal(np.array([0, -0.1, 0]), np.array([0, 0.1, 0]), 3)
        throw_states, throw_controls = nlp_throw(throwing_robot == robot1, throwing_q, throwing_qdot, goal_pt, dt, T=.5, T_final=0.65)
        print("goal:", goal_pt)
        did_calc_throw = True
    if step < len(throw_controls):
        apply_joint_vels(robot, throw_controls[step])
        apply_joint_vels(robot2 if robot == robot1 else robot1, np.zeros(3))
    if has_ball and step == len(throw_controls):
        release_ball()
        plot_ball_trajectory(get_ball_trajectory()[0])
        apply_joint_vels(robot, np.zeros(3))
        did_calc_throw = False
        return True
    return False

def combined_phase(robot, step):
    from throw_catch_nlp import nlp_combined
    global did_calc_throw, throw_controls, catch_controls, did_calc_catch
    throwing_robot = robot
    catching_robot = robot2 if robot == robot1 else robot1
    if not did_calc_throw and not did_calc_catch:
        print("Calculating combined for robot ", robot)
        throwing_q = get_joint_angles(throwing_robot)
        catching_q = get_joint_angles(catching_robot)
        throwing_qdot = get_joint_velocities(throwing_robot)
        catching_qdot = get_joint_velocities(catching_robot)
        goal_pt = get_end_effector_pos(catching_robot, end_effector_link_idx - 1)
        # is_throw_robot1, q, qdot_initial, qdot_catch, q_catch, goal, dt, T = 1.0, T_final = 0.4)
        print("goal:", goal_pt)
        throw_states, throw_controls, catch_states, catch_controls = nlp_combined(throwing_robot == robot1, throwing_q, throwing_qdot, catching_qdot, catching_q, goal_pt, dt, T=.3, T_final=0.2)
        did_calc_throw = True
        did_calc_catch = True

    if step < len(throw_controls):
        apply_joint_vels(throwing_robot, throw_controls[step])
    else:
        apply_joint_vels(throwing_robot, np.zeros(3))

    if has_ball and step == len(throw_controls):
        release_ball()
        plot_ball_trajectory(get_ball_trajectory()[0])
        apply_joint_vels(throwing_robot, np.zeros(3))

    if step < len(catch_controls):
        apply_joint_vels(catching_robot, catch_controls[step])
    else:
        apply_joint_vels(catching_robot, np.zeros(3))

    if not has_ball:
        attempt_catch(catching_robot, ball)

    if has_ball and step > len(throw_controls):
        did_calc_throw = False
        did_calc_catch = False
        return True

    return False

#select_method = 0 is baseline, 1 is combined solver, 2 is mpc for noise
def run_game_of_catch(select_method, num_steps = 10000, do_sleep=True, max_num_catches = 10):
    #setup
    set_ball_pos(INIT_BALL_POS)
    set_ball_velocity(INIT_BALL_VEL)
    # Set joint control mode to make the joints free to move (no motor control)
    for robot_num, robot in enumerate([robot1, robot2]):
        for joint_index in movable_joints[robot_num]:
            p.setJointMotorControl2(bodyIndex=robot,
                                    jointIndex=joint_index,
                                    controlMode=p.VELOCITY_CONTROL,
                                    force=0)  # Ensure no motor is controlling the joint

    set_joint_angles(robot1, [0, 0, 0])
    set_joint_angles(robot2, [np.pi, 0, 0])
    set_joint_vels(robot1, [0, 0, 0])
    set_joint_vels(robot2, [0, 0, 0])
    catching_robot = robot1
    throwing_robot = robot1
    state = "C"
    catch_step = 0
    throw_step = 0
    plot_ball_trajectory(get_ball_trajectory()[0])
    throw_catch_count = 0
    last_cycle_time = 0
    last_cycle_step = 0
    cycle_times = []

    #use damping if doing mpc (similar to air resistance)
    if(select_method==2):
            p.changeDynamics(ball, -1, linearDamping=0.04, angularDamping=0.04)
    else:
        p.changeDynamics(ball, -1, linearDamping=0, angularDamping=0)

    for step in range(num_steps):

        #BASELINE
        if(select_method == 0):
            if state == "C" and catch_phase(catching_robot, step - catch_step):
                throw_step = step
                state = "T"
                # if(catching_robot == robot1): #measure time between two consecutive bot1 caches
                #     print("cycle steps:", step-catch_step)
                if(throwing_robot == robot1): #cycle begins each time robot 2 catches
                    print("cycle steps:", step-last_cycle_step)
                    cycle_times.append(step-last_cycle_step)
                    last_cycle_step = step 
                throwing_robot = catching_robot
            
            if state == "T" and throw_phase(throwing_robot, step-throw_step):
                state = "C"
                catching_robot = robot2 if throwing_robot == robot1 else robot1
                catch_step = step
                throw_catch_count += 1

        #AUGMENTATION: Combined Solver
        if(select_method == 1):
            if state == "C" and catch_phase(catching_robot, step - catch_step):
                throw_step = step
                state = "comb"
                throwing_robot = catching_robot

            if state == "comb" and combined_phase(throwing_robot, step-throw_step):
                state = "comb"
                if(throwing_robot == robot1): #cycle begins every time robot 2 catches
                    print("cycle steps:", step-last_cycle_step)
                    cycle_times.append(step-last_cycle_step)        
                    last_cycle_step = step
                throwing_robot = robot2 if throwing_robot == robot1 else robot1
                catch_step = step
                throw_step = step
                throw_catch_count += 1

        #AUGMENTATION: MPC & Noise
        if(select_method==2):
            if state == "C":
                #apply wind to ball(TODO - change this to be a random, changing wind over duration of throw)
                if(throw_catch_count == 0): #lower force for initial throw, because farther flight
                    p.applyExternalForce(ball, -1, [0,0.05,0.01], [0, 0, 0], p.LINK_FRAME)      
                else: 
                    #to prevent wind blowing ball out of range of arm, blow in direction from goal towards robot center 
                    y_dir = 1 if (goal_pt[1] - get_end_effector_pos(catching_robot, 1)[1]) < 0 else -1
                    z_dir = 1 if (goal_pt[2] - get_end_effector_pos(catching_robot, 1)[2]) < 0 else -1
                    p.applyExternalForce(ball, -1, [0,y_dir*0.03,z_dir*0.05], [0, 0, 0], p.WORLD_FRAME)

                if catch_phase(catching_robot, step - catch_step, enable_mpc = True):
                    throw_step = step
                    state = "T"
                    throwing_robot = catching_robot
                    
        
            if state == "T" and throw_phase(throwing_robot, step-throw_step):
                state = "C"
                catching_robot = robot2 if throwing_robot == robot1 else robot1
                catch_step = step
                throw_catch_count += 1
                goal_pt = get_end_effector_pos(catching_robot, end_effector_link_idx-1)
        if throw_catch_count > max_num_catches:
            break
        p.stepSimulation()
        if(do_sleep):
            time.sleep(1./3000.)

    #clean up sim
    release_ball()
    return cycle_times, throw_catch_count
 


## Main code
if __name__ == '__main__': 

    ### Sim Setup
    physics_client = p.connect(p.GUI, options=f"--mp4=throw_catch.mp4")
    # physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0., 0., -9.81)
    #plane = p.loadURDF('plane.urdf')
    robot1 = p.loadURDF('three_link.urdf', basePosition=ROBOT1_BASE, useFixedBase=True)
    robot2 = p.loadURDF('three_link.urdf', basePosition=ROBOT2_BASE, useFixedBase=True)
    ball = p.loadURDF(generate_sphere(ball_rad, mass=ball_mass))
    dt = 1. / 240. #set timestep

    # get three movable joints and the end-effector link's index
    movable_joints = []
    end_effector_link_idx = None
    for robot_num, robot in enumerate([robot1, robot2]):
        movable_joints.append([])
        num_joints = p.getNumJoints(robot)
        for idx in range(num_joints):
            info = p.getJointInfo(robot, idx)
            print('Joint {}: {}'.format(idx, info))

            joint_type = info[2]
            if joint_type != p.JOINT_FIXED:
                movable_joints[robot_num].append(idx)

            link_name = info[12].decode('utf-8')
            if link_name == 'end_effector':
                end_effector_link_idx = idx

        for joint_index in movable_joints[robot_num]:
            p.changeDynamics(robot, joint_index, linearDamping=0, angularDamping=0)
        

    ### Run Throwing/Catching

    # Baseline (runs first): 
    run_game_of_catch(select_method=0, do_sleep = True, num_steps=5000, max_num_catches=5)
    
    
    #reset states after last game (unclear why doesn't take effect inside function)
    did_calc_catch = False
    did_calc_throw = False
    
    # Combined Solver Augmentation: 
    run_game_of_catch(select_method=1, do_sleep = True, num_steps=5000,max_num_catches=5)

    #reset states after last game 
    did_calc_catch = False
    did_calc_throw = False

    # MPC Augmentation:
    run_game_of_catch(select_method=2, do_sleep = True, num_steps=5000, max_num_catches=5)

    p.disconnect()

