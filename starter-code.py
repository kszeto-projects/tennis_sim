import time
import numpy as np
import pybullet as p
import pybullet_data
import pdb
from scipy.spatial.transform import Rotation as R

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
    robot = p.loadURDF('Files/three_link.urdf', useFixedBase = True)

    # get three movable joints and the end-effector link's index
    num_joints = p.getNumJoints(robot)
    movable_joints = []
    end_effector_link_idx = None
    for idx in range(num_joints):
        info = p.getJointInfo(robot, idx)
        print('Joint {}: {}'.format(idx, info))

        joint_type = info[2]
        if joint_type != p.JOINT_FIXED:
            movable_joints.append(idx)

        link_name = info[12].decode('utf-8')
        if link_name == 'end_effector':
            end_effector_link_idx = idx
            
    for joint_index in movable_joints:
        p.changeDynamics(robot, joint_index, linearDamping=0, angularDamping=0)

    # Set joint control mode to make the joints free to move (no motor control)
    for joint_index in movable_joints:
        p.setJointMotorControl2(bodyIndex=robot,
                                jointIndex=joint_index,
                                controlMode=p.VELOCITY_CONTROL,
                                force=0)  # Ensure no motor is controlling the joint
        
    # Set the initial joint states (angles and velocities)
    p.resetJointState(robot, movable_joints[1], -1.57, 0)
    p.resetJointState(robot, movable_joints[2], -0.75, 0)

    # TODO: Your code here
    
    # Problem 5: stiffness controller for cartesian setpoint 
    # Location  |   x   |   y   |   z
    #     1     |  0.2  |  0.2  |  0.2
    #     2     |  0.1  |  0.2  |  0.2
    #     3     |  0.1  | -0.2  |  0.2
    #     4     |  0.2  | -0.2  |  0.2
    
    locations = np.array([[0.2,0.2,0.2],[0.1,0.2,0.2],[0.1,-0.2,0.2],[0.2,-0.2,0.2],[0.2,0.2,0.2]])
    
    for location in locations:
        p.stepSimulation()

    # hold Ctrl and use the mouse to rotate, pan, or zoom
    for _ in range(2400): 
        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()
