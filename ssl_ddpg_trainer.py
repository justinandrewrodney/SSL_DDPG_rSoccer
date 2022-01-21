#from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import rsoccer_gym
import time
import spinup.algos.pytorch.ddpg.core as core
from spinup.utils.logx import EpochLogger
from actorcritic import Actor, Critic
from actorcriticmod import ActorMod, CriticMod

from spinupactorcritic import TScriptMLPActor, TScriptMLPCritic
from replaybuffer import ReplayBuffer


def test(env_fn, actor_model_path, max_ep_len=1000, num_test_episodes = 10, logger_kwargs=dict()):
    #import io
    test_env = env_fn()
    act_limit = test_env.action_space.high[0]


    print("Loading")
    actor = torch.jit.load("models/actors/"+actor_model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(actor.code)
    print("Loaded!")

    logger = EpochLogger(**logger_kwargs)

    def get_action(o):#added action limit here
        with torch.no_grad():
            a = actor.forward(torch.as_tensor(o, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))).cpu().numpy()
        return np.clip(a, -1, 1)

    for j in range(num_test_episodes):
        o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
        while not(d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time (noise_scale=0)
            o, r, d, _ = test_env.step(act_limit*get_action(o))
            test_env.render()
            ep_ret += r
            ep_len += 1
        logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

def write_to_csv(file_name, str_to_write):
        f = open(file_name, "a")
        f.write(file_name)
        f.close()

def train(env_fn, env_name, actor_init=Actor, critic_init=Critic,ac_kwargs=dict(), seed=0, 
         steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99, 
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
         update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10, 
         max_ep_len=1000, logger_kwargs=dict(), save_freq=1, render_test_freq=5):

    def create_directories(pytorch_save_path):
        import os
        try_path = pytorch_save_path
        okay = True
        iterations = 5
        for i in range(iterations):
            try:
                os.makedirs(try_path , exist_ok=False)
                okay = True
                pytorch_save_path = try_path
                break
            except:
                print(try_path + " already exists.")
                if(i<iterations-1):
                    try_path = pytorch_save_path+str(i+1)
                    okay = False
                    print("Trying " + try_path +"\n")
        if(not okay):
            print("Those folders already exist please try a new name")
            raise RuntimeError
        os.makedirs(pytorch_save_path+"/actors" , exist_ok=False)
        os.makedirs(pytorch_save_path+"/critics" , exist_ok=False)
        os.makedirs(pytorch_save_path+"/targets" , exist_ok=False)
        return  pytorch_save_path
    """
    Deep Deterministic Policy Gradient (DDPG)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor: constructor for actor model

        critic: constructor for critic model

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    pytorch_save_path = create_directories("models/"+env_name+"/"+logger_kwargs['exp_name'])
    print("The pytorch models will be saved in: "+pytorch_save_path+"\n(**the actor will also be saved in the top directory**)")
    
    logger = EpochLogger(**logger_kwargs)
    print(logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # keep? output of actor is [-1 - 1], scaling to action should be done on env side? act_limit is 1.0 in these envs anyways...
    act_limit = env.action_space.high[0]
    #print(act_limit)
    environment_info = "env_name,"+env_name+""+str(env.time_step)",obs_dim,"+str(obs_dim[0])+",act_dim,"+str(act_dim)+\
                                                ",act_limit,"+str(act_limit)+",act_limit,"+str(act_limit)+",\n" 
    data_info = "EpochNum,EpisodeNum,Total_rewards,Total_steps,Done,\n"
    write_to_csv(pytorch_save_path+"/testperformance.csv", environment_info+data_info)
    write_to_csv(pytorch_save_path+"/trainperformance.csv", environment_info+data_info)

    # Create actor-critic module and target networks
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actor = torch.jit.script(actor_init(obs_dim[0], act_dim)).to(dev)
    critic = torch.jit.script(critic_init(obs_dim[0], act_dim)).to(dev)
    #print(actor.code)
    #print(critic.code)
    for p in actor.parameters():
        print(p.device)
    for p in critic.parameters():
        print(p.device)


    actor_targ = torch.jit.script(actor_init(obs_dim[0], act_dim)).to(dev)
    critic_targ = torch.jit.script(critic_init(obs_dim[0], act_dim)).to(dev)
   
    #Super lazy way to copy parameters without copying a reference to them. Could be implemented better? Deepcopy?
    with torch.no_grad():
        for p, p_targ in zip(actor.parameters(), actor_targ.parameters()):
            p_targ.data.mul_(0)
            p_targ.data.add_(p.data)
    with torch.no_grad():
        for p, p_targ in zip(critic.parameters(), critic_targ.parameters()):
            p_targ.data.mul_(0)
            p_targ.data.add_(p.data)


    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in actor_targ.parameters():
        p.requires_grad = False
    for p in critic_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [actor, critic])
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        #print("o:" +str(o.device))
        #print("a:" +str(a.device))

        q = critic(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = critic_targ(o2, actor_targ(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().cpu().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q_pi = critic(o, actor(o))
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(actor.parameters(), lr=pi_lr)
    q_optimizer = Adam(critic.parameters(), lr=q_lr)

    # Set up model saving
    #logger.setup_pytorch_saver(ac)
    logger.setup_pytorch_saver(actor)
    logger.setup_pytorch_saver(critic)

    def update(data):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()
        #print(loss_q.device)

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in critic.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in critic.parameters():
            p.requires_grad = True

        # Record things
        logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(actor.parameters(), actor_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
            for p, p_targ in zip(critic.parameters(), critic_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        with torch.no_grad():
            a = actor.forward(torch.as_tensor(o, dtype=torch.float32, device=dev)).cpu().numpy()
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -1, 1)
    
    def test_agent():
        should_render = (epoch%render_test_freq == 0) 
        if(should_render):
            print("Rendering test episodes")
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(act_limit*get_action(o, 0))
                if(should_render):
                    test_env.render()
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            write_to_csv(pytorch_save_path+"/testperformance.csv",str(epoch)+','+str(j)+','+str(ep_ret)+','+str(ep_len)+','+str(d)+','+'\n')
            test_ep_number+=1

    
    def savePtFiles():
        actor.save(pytorch_save_path+"/"+logger_kwargs['exp_name']+ str(epoch)+'.pt')
        actor.save(pytorch_save_path+"/actors/"+logger_kwargs['exp_name']+"_actor_local_"+ str(epoch)+'.pt')
        critic.save(pytorch_save_path+"/critics/"+logger_kwargs['exp_name'] +"_critic_local_"+ str(epoch)+'.pt')
        actor_targ.save(pytorch_save_path+"/targets/"+logger_kwargs['exp_name']+"_actor_targ_"+ str(epoch)+'.pt')
        critic_targ.save(pytorch_save_path+"/targets/"+ logger_kwargs['exp_name'] +"_critic_targ_"+ str(epoch)+'.pt')
        print("Pytorch models (epoch:" +str(epoch) +") saved in: " +str(pytorch_save_path) +
                        "\nexample(actor): "+pytorch_save_path+"/"+logger_kwargs['exp_name']+ str(epoch)+'.pt')
    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    ep_number, epoch = 0, 0
    

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()
            #a[0] =-1.0
            #a[1] =-1.0
            #a[2] =0.0
            

        # Step the env
        o2, r, d, _ = env.step(act_limit*a)
        #env.render()

        #print("********\nstep: "+str(t))
        #print("Reward: "+str(r))
        #print("\nobs2: "+str(o2))

        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        """if(d):
            print("Time: " +str(t) +" eplen="+str(ep_len)+" max"+str(max_ep_len))
        """
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            write_to_csv(pytorch_save_path+"/trainperformance.csv", str(epoch)+','+str(ep_number)+','+str(ep_ret)+','+str(ep_len)+','+str(d)+','+'\n')
            o, ep_ret, ep_len = env.reset(), 0, 0
            ep_number+=1

        # Update handling
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                #logger.save_state({'env': env}, epoch)
                savePtFiles()

            # Test the performance of the deterministic version of the agent.
            print("Testing")
            test_agent()
            print("Done\ntotal steps: " +str(t))

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='SSLGoToBallDDPG-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    parser.add_argument('--test_file_name', type=str, default='')
    parser.add_argument('--r_test', type=int, default='5')    
    parser.add_argument('--spinup', type=bool, default=False)
    parser.add_argument('--mod', type=bool, default=False)

    args = parser.parse_args()

    spec = gym.envs.registration.spec(args.env)
    max_ep_st = spec.max_episode_steps
        
    should_load = True if(args.test_file_name != '') else False
    
    
    if(args.spinup==True):
        print("Using (modified) spinup actor-critic(warning, saying --spinup false still evaluates to true, run with no flag to not use spinup")
        actor_init=TScriptMLPActor; critic_init=TScriptMLPCritic;
    elif(args.mod==True):
        print("Using actor-critic-mod")
        actor_init=ActorMod; critic_init=CriticMod;

    else: 
        print("Using new actor-critic")
        actor_init=Actor; critic_init=Critic;
        
        
    
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    
    if(should_load):
        test(lambda : gym.make(args.env), actor_model_path=args.test_file_name, logger_kwargs=logger_kwargs,num_test_episodes=10)
    else:
        train(lambda : gym.make(args.env), env_name =args.env,
                    actor_init=actor_init, critic_init=critic_init,
                        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
                            gamma=args.gamma, seed=args.seed, max_ep_len = max_ep_st,
                            epochs=args.epochs, logger_kwargs=logger_kwargs, render_test_freq=args.r_test)