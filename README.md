# ECE 276C Fall 2024 Final Project
**TEAM: Aram Chemishkian, Jason Stanley, Kevin Szeto**
## Project: Throw and Catch
***
## How to run project
In order to run the project, all you need to do is download the repo, install the necessary packages, and run the main script. This script will run each of three methods for throw and catch (baseline, combined solver, and mpc for noise) in consecutive order.

***
## Project Proposal
**What are you trying to do?** 

Control two independent robot manipulators to play a game of catch. The robots should be able to catch and throw the ball between themselves. Assuming a known (deterministic) trajectory of the ball (to not introduce complexity with camera models), the robot should catch the ball as quickly as possible. This entails moving to the nearest feasible future position of the ball.

**What is your baseline and how is it lacking?**

We start with the optimal time controller to intercept the ball and grasp it. This will apply a harsh declaration to the ball. We can try and minimize the deceleration by matching the ball velocity, and by adding motion planning to the transition between a throw and a catch. 

**What is your alternative/improvement and why do you think it will work?**

To improve time, we can also take into account the pose of the other robot’s end-effector and throw the ball to that location. This would improve the cycle time for a toss, catch, toss, catch cycle, compared to simply minimizing the catch and release time for a single robot individually. 

**How will you simulate the problem (what simulator, what robot):**

We will use pybullet simulator, starting with the 2D case (reusing HW1 robot), and once we have a proof of concept working, plan to scale up to 3D. In 3D, we will first use a generic HW2-style robot arm with 3DOF, and simply “catch” the ball by matching the ball’s position and velocity. Depending on how fast we are able to implement the baseline and augmented versions for the HW-2 style robot arms, we may investigate trying to implement them with the HW3 panda arm and actually require gripping the ball during the trajectory.

