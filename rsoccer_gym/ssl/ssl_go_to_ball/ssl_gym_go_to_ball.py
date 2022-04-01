import math
import random
from typing import Dict

import gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree

e

"""
Uses similar state-action-rewards as this paper
https://zhuyifengzju.github.io/files/2018Robocup.pdf
"""
class SSLGoToBallEnv(SSLBaseEnv):
    """The SSL robot needs to reach the ball 


        Description:
            One blue robot and a ball are randomly placed on a div B field,
            the episode ends when the robots is closer than 0.2m from the ball
        Observation:
            Type: Box(3 + 3*robots_blue )
            All cordinates are using robot as reference frame.
            Normalized Bounds to [-1.2, 1.2]
            Num      Observation normalized  
            0->3     Ball [Sin(theta), , Cos(theta), Dist_to_ball]
            4->6    id 0 Blue [ v_x, v_y, v_theta]
        Actions:
            Type: Box(3, )
            Num     Action **all actions are given in range [-1 - 1] must be scaled here.
            0       id 0 Blue Local X Direction Speed  (%) 
            1       id 0 Blue Local Y Direction Speed  (%)
            2       id 0 Blue Angular Speed  (%)
        Reward:
            100 if ball is reached
            **needs more details***
        Starting State:
            Randomized Robots and Ball initial Position
        Episode Termination:
            Ball is reached or (330 steps*.03 seconds)
    """

    def __init__(self, field_type=1, n_robots_yellow=0):
        super().__init__(field_type=field_type, n_robots_blue=1, 
                         n_robots_yellow=n_robots_yellow, time_step=0.030) #time_step=0.025

        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(3, ), dtype=np.float32)
        
        n_obs = 6 

        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_obs, ),
                                                dtype=np.float32)
        
        # Limit robot speeds
        self.max_v = 1.5
        self.max_w = 1.395

        print('Environment initialized')


    def get_sin_angle_dist(self, rob, rob_ang, point2):
        angle_between = math.atan2(point2.y - rob.y, point2.x - rob.x);
        angle_diff = math.atan2(math.sin(angle_between-rob_ang), math.cos(angle_between-rob_ang))
        angle_s = math.sin(angle_diff);
        angle_c = math.cos(angle_diff);
        dist_between = np.linalg.norm(np.array([point2.x, point2.y]) - np.array([rob.x, rob.y]))
        return angle_s, angle_c, dist_between


    def _frame_to_observations(self):
        observation = []

        the_robot = self.frame.robots_blue[0]
        rob_ang = np.deg2rad(the_robot.theta)

        angle_2ball_s, angle_2ball_c, dist_robot_ball = self.get_sin_angle_dist(the_robot, rob_ang, self.frame.ball)


        observation.append(angle_2ball_s)
        observation.append(angle_2ball_c)
        observation.append(dist_robot_ball)
        
        v_x, v_y, v_theta= self.frame.robots_blue[0].v_x, self.frame.robots_blue[0].v_y, np.deg2rad(self.frame.robots_blue[0].v_theta)
        v_x, v_y = v_x*np.cos(rob_ang) + v_y*np.sin(rob_ang),\
            -v_x*np.sin(rob_ang) + v_y*np.cos(rob_ang)

        observation.append(v_x)
        observation.append(v_y)
        observation.append(v_theta)

        
        return np.array(observation, dtype=np.float32)


    def _get_commands(self, actions):
        commands = []

        angle = self.frame.robots_blue[0].theta
        v_x, v_y, v_theta = self.convert_actions(actions, np.deg2rad(angle))
        
        cmd = Robot(yellow=False, id=0, v_x=v_x, v_y=v_y, v_theta=v_theta)
        commands.append(cmd)

        return commands

    def convert_actions(self, action, angle):
        """Denormalize, clip to absolute max"""  #and convert to local"""
        # Denormalize
        v_x = action[0] * self.max_v
        v_y = action[1] * self.max_v
        v_theta = action[2] * self.max_w

        # clip by max absolute
        v_norm = np.linalg.norm([v_x,v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x*c, v_y*c
        
        return v_x, v_y, v_theta


    def _calculate_reward_and_done(self):
        """   float reward = 0;    //rtotal = rcontact + rdistance + rorientation

   if(robot->hasBall()) //contact...perhaps change to robot has ball on dribbler like in paper?
       reward+= 100.f;
   //distance
    //   reward += ( (5.f/sqrt(2.f*M_PI))*exp(-(d*d)/2.f) )- 2.f;  
   //orientation
   float theta_r_b  = fabs(Measurements::angleDiff(Measurements::angleBetween(*robot, *ball), robot->getOrientation()));
   reward += static_cast<float>( (1.f/sqrt(2.f*M_PI))*exp(-2.f*(theta_r_b/(M_PI*M_PI))) );          //qInfo()<<"Reward angle: "<<reward;
        """
        reward = 0
        done = 0

        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        robot_ang = np.deg2rad(robot.theta)
        
        #Distance Reward
        dist_robot_ball = np.linalg.norm(np.array([ball.x, ball.y]) - np.array([robot.x, robot.y]))
        reward+= ((5.0/math.sqrt(2.0*math.pi))*math.exp(-(dist_robot_ball*dist_robot_ball)/2.0) )- 2.0; 

        #Faceball reward
        angle_between = math.atan2(ball.y - robot.y, ball.x - robot.x);
        angle_diff = math.atan2(math.sin(angle_between-robot_ang), math.cos(angle_between-robot_ang))


        reward += (1.0/math.sqrt(2.0*math.pi))*math.exp(-2.0*(abs(angle_diff)/(math.pi*math.pi)));

        # Check if robot is less than 0.2m from ball
        #if dist_robot_ball < 0.2 and abs(angle_diff) < math.pi/6:
        if(self.frame.robots_blue[0].infrared):
            reward += 100.0
            done = 1

        return reward, done
    
    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x(): return random.uniform(-field_half_length + 0.1,
                                       field_half_length - 0.1)

        def y(): return random.uniform(-field_half_width + 0.1,
                                       field_half_width - 0.1)

        def theta(): return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=x(), y=y())

        min_dist = 0.2

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))
        
        for i in range(self.n_robots_blue):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        for i in range(self.n_robots_yellow):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        return pos_frame
