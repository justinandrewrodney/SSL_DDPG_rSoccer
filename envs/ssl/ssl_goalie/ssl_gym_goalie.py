import math
import random
from typing import Dict

import gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
class SSLGoalieEnv(SSLBaseEnv):
    """The SSL robot needs to make a goal


        Description:
            One blue robot and a ball are placed on fixed position on a half 
            div B field, the robot is rewarded if it makes a goal
        Observation:
            Type: Box(10)
            (PBx, P By, V Bx, V By, ωR, dr−g,sin(θl), cos(θl),sin(θr), cos(θr)) **in relation to robot**
            Num      Observation normalized  
            0->1     Point of ball
            2->3     Velocity of ball
            4        Angular velocity of robot
            5        Distance of robot to goal(middle point)
            6->7     Sin and Cos of angle from robot to Top goalpost 
            8->9    Sin and Cos of angle from robot to Bottom goalpost
        Actions:
            Type: Box(2, )
            Num     Action
            0       V_theta
            1       Kick
            
        Reward:
            
    """

    def __init__(self, field_type=1, random_init=False, enter_goal_area=False, burn_in=False):
        super().__init__(field_type=field_type, n_robots_blue=1, 
                         n_robots_yellow=1, time_step=0.03)
        self.random_init = random_init
        self.enter_goal_area= enter_goal_area
        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(3, ), dtype=np.float32)
        
        n_obs = 11#10
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_obs, ),
                                                dtype=np.float32)

        # Limit robot speeds
        self.max_v = 2.5
        self.max_w = 10

        self.goal_post_top = Ball(x=self.field.length/2, y=self.field.goal_width / 2)
        self.goal_post_bot = Ball(x=self.field.length/2, y=-self.field.goal_width / 2)
        self.goal_post_mid = Ball(x=self.field.length/2, y=0)
        

        self.cur_episode = 0
        self.burning_in = False#burn_in
        self.burn_in_time = 400
        
        print(self.goal_post_bot)
        
        print('Environment initialized: SSL-ShootShortNew')

    def reset(self):
        self.reward_shaping_total = None



        self.cur_episode +=1
        if(self.burning_in and self.cur_episode >= self.burn_in_time):
            self.burning_in = False
            print("**********Done burning in")
        return super().reset()

    def step(self, action):
        observation, reward, done, _ = super().step(action)
        return observation, reward, done, self.reward_shaping_total

    def get_sin_angle_dist(self, rob, rob_ang,point2):
        angle_between = math.atan2(point2.y - rob.y, point2.x - rob.x);
        angle_diff = math.atan2(math.sin(angle_between-rob_ang), math.cos(angle_between-rob_ang))
        angle_s = math.sin(angle_diff);
        angle_c = math.cos(angle_diff);
        dist_between = np.linalg.norm(np.array([point2.x, point2.y]) - np.array([rob.x, rob.y]))
        return angle_s, angle_c, dist_between

    def _frame_to_observations(self):
        #= (PBx, P By, V Bx, V By, ωR, dr−g,sin(θl), cos(θl),sin(θr), cos(θr))
        observation = []

        the_robot = self.frame.robots_blue[0]
        rob_ang = np.deg2rad(the_robot.theta)
        ball = self.frame.ball


        # angle_2ball_s, angle_2ball_c, dist_ball = self.get_sin_angle_dist(the_robot, rob_ang, ball)
        # observation.append(angle_2ball_s)
        # observation.append(angle_2ball_c)
        # observation.append(dist_ball)

        ball_x, ball_y = ball.x - the_robot.x, ball.y- the_robot.y
        ball_x, ball_y = ball_x*np.cos(rob_ang) + ball_y*np.sin(rob_ang),\
            -ball_x*np.sin(rob_ang) + ball_y*np.cos(rob_ang)
        #PBx, P By,
        observation.append(ball_x)
        observation.append(ball_y)

        ball_v_x, ball_v_y =   self.frame.ball.v_x, self.frame.ball.v_y
        ball_v_x, ball_v_y = ball_v_x*np.cos(rob_ang) + ball_v_y*np.sin(rob_ang),\
            -ball_v_x*np.sin(rob_ang) + ball_v_y*np.cos(rob_ang)
        #V Bx, V By,
        observation.append(ball_v_x)
        observation.append(ball_v_y)
        
        
        """
        
        """
        #dist_to_ball = np.linalg.norm(np.array([ball.x, ball.y]) - np.array([the_robot.x, the_robot.y]))
        #observation.append(dist_to_ball)

        # observation.append(
        #         np.sin(np.deg2rad(the_robot.theta))
        # )
        
        # observation.append(
        #     np.cos(np.deg2rad(the_robot.theta))
        # )

        
        v_x, v_y, v_theta= self.frame.robots_blue[0].v_x, self.frame.robots_blue[0].v_y, np.deg2rad(self.frame.robots_blue[0].v_theta)
        v_x, v_y = v_x*np.cos(rob_ang) + v_y*np.sin(rob_ang),\
            -v_x*np.sin(rob_ang) + v_y*np.cos(rob_ang)

        observation.append(v_x)
        observation.append(v_y)
        observation.append(v_theta)

        # dr−g
        # dist_to_goal = np.linalg.norm(np.array([self.goal_post_mid.x, self.goal_post_mid.y]) - np.array([the_robot.x, the_robot.y]))
        # observation.append(dist_to_goal)

        #rbt.x > half_len - pen_len and abs(rbt.y) < half_pen_wid
        half_len = self.field.length / 2
        pen_len = self.field.penalty_length


        dist_top = (self.field.penalty_width / 2) - the_robot.y
        dist_bot = the_robot.y - (- self.field.penalty_width / 2)
        dist_left = the_robot.x - (half_len - pen_len)
        dist_right = half_len - the_robot.x

        observation.append(dist_top)
        observation.append(dist_bot)
        observation.append(dist_left)
        observation.append(dist_right)


        
        # """Delete"""
        # print("ball_x",ball_x)
        # print("ball_y",ball_y)
        # print("ball_v_x",ball_v_x)
        # print("ball_v_y",ball_v_y)
        # print("robot_v_theta",robot_v_theta)
        # print("dist_to_goal",dist_to_goal)

        # print("angle_2top_s",angle_2top_s)
        # print("angle_2top_c",angle_2top_c)
        # print("angle_2bottom_s",angle_2bottom_s)
        # print("angle_2bottom_c",angle_2bottom_c)
        # ###############
        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []

        angle = self.frame.robots_blue[0].theta
        v_x, v_y, v_theta = self.convert_actions(actions, np.deg2rad(angle))
        
        cmd = Robot(yellow=False, id=0, v_x=v_x, v_y=v_y, v_theta=v_theta)
        commands.append(cmd)

        #Have attacker kick
        if(self.steps>24):
            cmd = Robot(yellow=True, id=0, v_x=0, v_y=0, v_theta=0, kick_v_x=5.0,dribbler=True)
            commands.append(cmd)
        return commands

    def convert_actions(self, action, angle):
        """Denormalize, clip to absolute max and convert to local"""
        # Denormalize
        v_x = action[0] * self.max_v
        v_y = action[1] * self.max_v
        v_theta = action[2] * self.max_w
        # Convert to local
        # v_x, v_y = v_x*np.cos(angle) + v_y*np.sin(angle),\
        #     -v_x*np.sin(angle) + v_y*np.cos(angle)

        # clip by max absolute
        v_norm = np.linalg.norm([v_x,v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x*c, v_y*c
        
        return v_x, v_y, v_theta

    def _calculate_reward_and_done(self):
        if self.reward_shaping_total is None:
            self.reward_shaping_total = {
                'goal': 0,
                'rbt_in_gk_area': 0,
                'done_ball_out': 0,
                'done_ball_out_right': 0,
                'done_rbt_out': 0,
                'ball_dist': 0,
                'ball_grad': 0,
                'energy': 0,
                'angle_goal': 0
            }
        reward = 0
        done = False
        
        # Field parameters
        half_len = self.field.length / 2
        half_wid = self.field.width / 2
        pen_len = self.field.penalty_length
        half_pen_wid = self.field.penalty_width / 2
        half_goal_wid = self.field.goal_width / 2
        
        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        
        def robot_in_gk_area(rbt):
            return rbt.x > half_len - pen_len and abs(rbt.y) < half_pen_wid
        
        # Check if goalie not in goalkeeper area
        if(not robot_in_gk_area(robot)):
            done=True
            reward = -20
        elif ball.x > half_len:
            done = True
            if abs(ball.y) < half_goal_wid:
                reward = -20
                print("Goal")
                self.reward_shaping_total['goal'] += 1
            else:
                self.reward_shaping_total['done_ball_out_right'] += 1
        elif self.last_frame is not None:
            reward = self.face_ball_rw() + self.intersect_ball_path_rw()#+ face_goal_rw
            if(robot.infrared):
                #caught ball
                reward += 100
                done = True
 
                           

        done = done

        return reward, done
    
    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        if self.random_init:
            half_len = self.field.length / 2
            half_wid = self.field.width / 2
            penalty_len = self.field.penalty_length
            def x(): return random.uniform(0.3, half_len - penalty_len - 0.3)
            def y(): return random.uniform(-half_wid + 0.1, half_wid - 0.1)
            def theta(): return random.uniform(0, 360)
        else:
            def x(): return self.field.length / 4
            def y(): return self.field.width / 8
            def theta(): return 0
        def x_goalie(): return random.uniform(half_len - penalty_len +.2 , half_len - .2)
        def y_goalie(): return random.uniform(-self.field.penalty_width/2 +.2, self.field.penalty_width/2 -.2)

        pos_frame: Frame = Frame()

        #spawn ball and robot together
        rob_x, rob_y, rob_theta = x(), y(), theta()
        
        
        rob_theta = math.atan2((self.goal_post_mid.y + random.uniform(-self.field.goal_width / 2, self.field.goal_width / 2)) - rob_y, \
                                 self.goal_post_mid.x - rob_x) 
        #if (self.burning_in and self.cur_episode < self.burn_in_time):
        #    rob_theta = math.atan2(self.goal_post_mid.y - rob_y, self.goal_post_mid.x - rob_x);
            
        pos_frame.robots_yellow[0] = Robot(x=rob_x, y=rob_y, theta=np.rad2deg(rob_theta)) 
        d_ball_rbt = (self.field.ball_radius + self.field.rbt_radius)*1.001
        rob_ang = np.deg2rad(pos_frame.robots_yellow[0].theta)

        pos_frame.ball = Ball(x=pos_frame.robots_yellow[0].x +  math.cos(rob_ang)*d_ball_rbt,\
                               y= pos_frame.robots_yellow[0].y + math.sin(rob_ang)*d_ball_rbt  )


        pos_frame.robots_blue[0] = Robot(x=x_goalie(), y=y_goalie(), theta=np.rad2deg(theta())) 

        return pos_frame


    def intersect_ball_path_rw(self):
        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        robot_ang = np.deg2rad(robot.theta)
        #Faceball reward
        ball_vel = np.linalg.norm(np.array([ball.v_x, ball.v_y])) 
        if(ball_vel < .1):
            angle_of_velocity_ball = math.atan2(self.goal_post_mid.y- ball.y,  self.goal_post_mid.x - ball.x);
        else:
            angle_of_velocity_ball = math.atan2(ball.v_y, ball.v_x)
        angle_between_ball_robot = math.atan2(robot.y - ball.y,  robot.x - ball.x);

        angle_diff = math.atan2(math.sin(angle_between_ball_robot - angle_of_velocity_ball), math.cos(angle_between_ball_robot - angle_of_velocity_ball))
        
        return (1.0/math.sqrt(2.0*math.pi))*math.exp(-2.0*(abs(angle_diff)/(math.pi*math.pi)));





    def face_ball_rw(self):
        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        robot_ang = np.deg2rad(robot.theta)
        #Faceball reward
        angle_between = math.atan2(ball.y - robot.y, ball.x - robot.x);
        angle_diff = math.atan2(math.sin(angle_between-robot_ang), math.cos(angle_between-robot_ang))

        return (1.0/math.sqrt(2.0*math.pi))*math.exp(-2.0*(abs(angle_diff)/(math.pi*math.pi)));

        # reward += (1.0/math.sqrt(2.0*math.pi))*math.exp(-2.0*(abs(angle_diff)/(math.pi*math.pi)));

        # # Check if robot is less than 0.2m from ball
        # #if dist_robot_ball < 0.2 and abs(angle_diff) < math.pi/6:
        # if(self.frame.robots_blue[0].infrared):
        #     reward += 100.0
        #     done = 1
    


    def __face_goal_rw(self):
        assert(self.last_frame is not None)

        if (self.kicked):
            return 0

        robot = self.frame.robots_blue[0]
        robot_ang = np.deg2rad(robot.theta)
        
        # angle_between_top = math.atan2(self.goal_post_top.y - robot.y, self.goal_post_top.x - robot.x);
        # angle_between_bottom = math.atan2(self.goal_post_bot.y - robot.y, self.goal_post_bot.x - robot.x);
        
        # if(angle_between_bottom <= robot_ang and robot_ang<= angle_between_top):
        #     return 0

        #angle from robot to goal top
        angle_between_mid = math.atan2(self.goal_post_mid.y - robot.y, self.goal_post_mid.x - robot.x);
        #Difference in radians between where robot is facing and top of goal area
        angle_diff_mid = math.atan2(math.sin(angle_between_mid-robot_ang), math.cos(angle_between_mid-robot_ang))

        last_robot = self.last_frame.robots_blue[0]
        last_robot_ang = np.deg2rad(last_robot.theta)

        #angle from robot to goal top
        last_angle_between_mid = math.atan2(self.goal_post_mid.y - last_robot.y, self.goal_post_mid.x - last_robot.x);
        #Difference in radians between where robot is facing and top of goal area
        last_angle_diff_mid = math.atan2(math.sin(last_angle_between_mid-last_robot_ang), math.cos(last_angle_between_mid-last_robot_ang))

        return (abs(last_angle_diff_mid)- abs(angle_diff_mid)) #/(self.max_v*self.time_step)
 