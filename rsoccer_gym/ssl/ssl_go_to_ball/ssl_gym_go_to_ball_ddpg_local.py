import math
import random
from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from typing import Dict

import gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree


class SSLGoToBallDDPGLocalEnv(SSLBaseEnv):
    """The SSL robot needs to reach the ball 


        Description:
            One blue robot and a ball are randomly placed on a div B field,
            the episode ends when the robots is closer than 0.2m from the ball
        Observation:
            Type: Box(4 + 7*n_robots_blue + 5*n_robots_yellow)
            Normalized Bounds to [-1.2, 1.2]
            Num      Observation normalized  
            0->3     Ball [X, Y, V_x, V_y]
            4->10    id 0 Blue [X, Y, sin(theta), cos(theta), v_x, v_y, v_theta]
            +5*i     id i Yellow Robot [X, Y, v_x, v_y, v_theta]
        Actions:
            Type: Box(3, )
            Num     Action
            0       id 0 Blue Global X Direction Speed  (%)
            1       id 0 Blue Global Y Direction Speed  (%)
            2       id 0 Blue Angular Speed  (%)
        Reward:
            1 if ball is reached
        Starting State:
            Randomized Robots and Ball initial Position
        Episode Termination:
            Ball is reached or 30 seconds (1200 steps)
    """

    def __init__(self, field_type=1, n_robots_yellow=0):
        super().__init__(field_type=field_type, n_robots_blue=1, 
                         n_robots_yellow=n_robots_yellow, time_step=0.030) #time_step=0.025

        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(3, ), dtype=np.float32)
        
        #n_obs = 4 + 7*self.n_robots_blue + 2*self.n_robots_yellow
        #sin_ball, cos_ball, dist_ball, robot_vx(local), robot_vy(local), vtheta
        n_obs = 6 

        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_obs, ),
                                                dtype=np.float32)
        
        # Limit robot speeds
        self.max_v = 2.5
        self.max_w = 10

        print('Environment initialized')


    def get_sin_angle_dist(self, rob, rob_ang,point2):
        angle_between = math.atan2(point2.y - rob.y, point2.x - rob.x);
        angle_diff = math.atan2(math.sin(angle_between-rob_ang), math.cos(angle_between-rob_ang))
        #print("ahhh"+str(angle_diff))
        angle_s = math.sin(angle_diff);
        angle_c = math.cos(angle_diff);
        dist_between = np.linalg.norm(np.array([point2.x, point2.y]) - np.array([rob.x, rob.y]))
        return angle_s, angle_c, dist_between


    def _frame_to_observations(self):
        observation = []

        #ball_x_local = self.frame.ball.x - self.frame.robot.x
        #ball_y_local = self.frame.ball.y - self.frame.robot.y
        #ball_x_local, ball_y_local = self.rotate_local([ball_x_local, ball_y_local], )
        #angle_to_ball = math.atan2(self.frame.ball.y - self.frame.robot.y, self.frame.ball.x - self.frame.robot.x);
        #angle_2ball_s = math.sin(angle_to_ball);
        #angle_2ball_c = math.cos(angle_to_ball);
        #dist_robot_ball = np.linalg.norm(np.array([self.frame.ball.x, self.frame.ball.y]) - np.array([self.frame.robot.x, self.frame.robot.y]))
        the_robot = self.frame.robots_blue[0]
        rob_ang = np.deg2rad(the_robot.theta)

        angle_2ball_s, angle_2ball_c, dist_robot_ball = self.get_sin_angle_dist(the_robot, rob_ang, self.frame.ball)

        """observation.append(self.norm_pos(self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))"""
        observation.append(angle_2ball_s)
        observation.append(angle_2ball_c)
        observation.append(dist_robot_ball)
        
        v_x, v_y, v_theta= self.frame.robots_blue[0].v_x, self.frame.robots_blue[0].v_y, np.deg2rad(self.frame.robots_blue[0].v_theta)
        v_x, v_y = v_x*np.cos(rob_ang) + v_y*np.sin(rob_ang),\
            -v_x*np.sin(rob_ang) + v_y*np.cos(rob_ang)

        observation.append(v_x)
        observation.append(v_y)
        observation.append(v_theta)

        """for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(
                np.sin(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(
                np.cos(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(self.norm_v(self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(self.frame.robots_blue[i].v_theta))

        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))"""

        return np.array(observation, dtype=np.float32)

        """observation = []

        observation.append(self.norm_pos(self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(
                np.sin(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(
                np.cos(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(self.norm_v(self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(self.frame.robots_blue[i].v_theta))

        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))

        return np.array(observation, dtype=np.float32)"""

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
        #print("actions: " +str(action))
        v_x = action[0] * self.max_v
        v_y = action[1] * self.max_v
        v_theta = action[2] * self.max_w
        # Convert to local
        #v_x, v_y = v_x*np.cos(angle) + v_y*np.sin(angle),\
        #    -v_x*np.sin(angle) + v_y*np.cos(angle)

        # clip by max absolute
        v_norm = np.linalg.norm([v_x,v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x*c, v_y*c
        
        return v_x, v_y, v_theta

    def rotate_local(self, point, angle):
        """convert to local"""
        x = point[0] 
        y = point[1]
        # Convert to local
        #x, y = x*np.cos(angle) + y*np.sin(angle),\
        #    -x*np.sin(angle) + y*np.cos(angle)
        #np.rotate
        return x, y

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
        
        dist_robot_ball = np.linalg.norm(np.array([ball.x, ball.y]) - np.array([robot.x, robot.y]))
        reward+= ((5.0/math.sqrt(2.0*math.pi))*math.exp(-(dist_robot_ball*dist_robot_ball)/2.0) )- 2.0; 

        
        angle_between = math.atan2(ball.y - robot.y, ball.x - robot.x);
        angle_diff = math.atan2(math.sin(angle_between-robot_ang), math.cos(angle_between-robot_ang))

        #angle_diff = angle_between - robot_ang
        #print("robot pos: "+str(robot.x) +", "+str(robot.y))
        #print("ball pos: "+str(ball.x) +", "+str(ball.y))

        #print("y_diff: " +str(ball.y - robot.y))
        #print("x_diff: " +str(ball.x - robot.x))

        
        #print("dist_robot_ball"+str(dist_robot_ball))
        #print("angle_between: " +str(angle_between))
        #print("angle_diff"+str(angle_diff))

        reward += (1.0/math.sqrt(2.0*math.pi))*math.exp(-2.0*(abs(angle_diff)/(math.pi*math.pi)));

        # Check if robot is less than 0.2m from ball
        if dist_robot_ball < 0.2 and abs(angle_diff) < math.pi/6:
            reward += 100.0
            done = 1

        #print(reward)
        #done = reward

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
