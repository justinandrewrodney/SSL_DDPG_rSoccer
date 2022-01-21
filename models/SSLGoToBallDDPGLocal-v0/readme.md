In this folder you will find pt files that were saved using torchscript, meaning that they can be used in C++. You can build models in python/rSoccer and use them in C++ but you will need to make sure that the observations and actions are standardized.

Our time step is set to 30 ms(.03sec) because that is roughly
the performance of our software in release mode. Debug mode
is roughly double but can greatly vary depending on what behaviors
are doing.(just now debug mode is actually running faster...its totally unpredictable)

For this Environment/Skill, you will need the following
State: **These are in reference to the robots orientation. Global coordinates are not the reference point**
Input = [Sin(theta), Cos(theta), Dist_to_Ball, robot.x, robot.y, robot.v]
Theta: Angle from robot to ball in radians [-3.14 - 3.14]
Dist_to_ball: Distance in meters.
x and y: Translational velocity in meters per second.
robot v: angular velocity in radians per second(?)


You will get the following from the model
Actions= [Vx, Vy, Vtheta] [-1 - 1] (you will need to convert to your max speeds.)