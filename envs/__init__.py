from gym.envs.registration import register


register(id='SSLGoToBall-v0',
         entry_point='envs.ssl_go_to_ball:SSLGoToBallEnv',
         kwargs={'field_type': 2, 'n_robots_yellow': 0},
         max_episode_steps=1200
         )

register(id='SSLShoot-v0',
         entry_point='envs.ssl_shoot:SSLShootEnv',
         kwargs={'field_type': 1, 'random_init': True,
                 'enter_goal_area': False},
         max_episode_steps=100
         )

register(id='SSLShootCMU-v0',
         entry_point='envs.ssl_shoot:SSLShootCMUEnv',
         kwargs={'field_type': 1, 'random_init': True,
                 'enter_goal_area': False},
         max_episode_steps=100
         )


register(id='SSLGoalie-v0',
         entry_point='envs.ssl_goalie:SSLGoalieEnv',
         kwargs={'field_type': 1, 'random_init': True,
                 'enter_goal_area': False},
         max_episode_steps=100
         )
