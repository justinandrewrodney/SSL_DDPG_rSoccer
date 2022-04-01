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
"""This is program is heavily influenced by the spinning up example(https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/ddpg/ddpg.html)
    Modifications have been made to work with rSoccer environments as well as utilize any Actor and Critic
    modules(not the wrapper module used in the example)  and allow models to be exported as TorchScript 
    to allow deployment of the models into C++ systems
"""
def test(env_fn,exp_name,env_name, actor_model_path, max_ep_len, num_test_episodes = 10, logger_kwargs=dict()):
    #import io
    test_env = env_fn()
    act_limit = test_env.action_space.high[0]


    print("Loading")
    actor = torch.jit.load("models/"+env_name+"/"+exp_name+"/"+actor_model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
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
        f.write(str_to_write+'\n')
        f.close()
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
        print("The pytorch models will be saved in: "+pytorch_save_path+"\n(**the actor will also be saved in the top directory**)")
        return  pytorch_save_path

def train(env_fn, env_name, max_ep_len, actor_init=Actor, critic_init=Critic,  seed=0, 
         steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99, 
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
         update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10, 
          logger_kwargs=dict(), save_freq=1, render_test_freq=5):
    """
    Deep Deterministic Policy Gradient (DDPG)
    This is modified from: https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/ddpg/ddpg.html
    to work with rSoccer environments.

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor: constructor for actor model. Takes in an observation, returns an actions ranged [-1,1]

        critic: constructor for critic model. Takes in an observation and actions.

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
    #*                                                                  *
    #*          Setting up actor-critic local and target networks       *
    #*          As well as logger and csv to store results              *
    #*                                                                  *
    pytorch_save_path = create_directories("models/"+env_name+"/"+logger_kwargs['exp_name'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger = EpochLogger(**logger_kwargs)#logger from spinup. Not sure fully supported with this implementation

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]# keep? output of actor is [-1 - 1], scaling to action should be done on env side? act_limit is 1.0 in these envs anyways...

    meta_d = "env_name,"+env_name+",time_step,"+str(env.time_step)+",obs_dim,"+str(obs_dim[0])+\
                ",act_dim,"+str(act_dim)+",act_limit,"+str(act_limit)+",act_limit,"+str(act_limit)+",\n"+\
                                                        "EpochNum,EpisodeNum,Total_rewards,Total_steps,Done,\n" 
    write_to_csv(pytorch_save_path+"/testperformance.csv", meta_d)
    write_to_csv(pytorch_save_path+"/trainperformance.csv", meta_d)

    # Create actor-critic module and target networks
    actor = torch.jit.script(actor_init(obs_dim[0], act_dim)).to(dev)
    critic = torch.jit.script(critic_init(obs_dim[0], act_dim)).to(dev)
    
    [print("Checking actor device: " +str(ap.device)) for ap in actor.parameters()]
    [print("Checking critic device: " +str(cp.device)) for cp in critic.parameters()]

    actor_targ = torch.jit.script(actor_init(obs_dim[0], act_dim)).to(dev)
    critic_targ = torch.jit.script(critic_init(obs_dim[0], act_dim)).to(dev)
   
    #Super lazy way to copy parameters without copying a reference to them. Could be implemented better? Deepcopy?
    with torch.no_grad():
        for ap, ap_targ in zip(actor.parameters(), actor_targ.parameters()):
            ap_targ.data.mul_(0)
            ap_targ.data.add_(ap.data)
        for cp, cp_targ in zip(critic.parameters(), critic_targ.parameters()):
            cp_targ.data.mul_(0)
            cp_targ.data.add_(cp.data)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for ap in actor_targ.parameters():
        ap.requires_grad = False
    for cp in critic_targ.parameters():
        cp.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [actor, critic])
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)

    # Set up optimizers for policy and q-function
    actor_optimizer = Adam(actor.parameters(), lr=pi_lr)
    critic_optimizer = Adam(critic.parameters(), lr=q_lr)

    # Set up model saving
    #logger.setup_pytorch_saver(ac)
    logger.setup_pytorch_saver(actor)
    logger.setup_pytorch_saver(critic)
    #*                      End setup                                   *

    #*         **START DEFINING FUNCTIONS FOR TRAINING                  *
    def savePtFiles():
        actor.save(pytorch_save_path+"/"+logger_kwargs['exp_name']+ str(epoch+1)+'.pt')
        actor.save(pytorch_save_path+"/actors/"+logger_kwargs['exp_name']+"_actor_local_"+ str(epoch+1)+'.pt')
        critic.save(pytorch_save_path+"/critics/"+logger_kwargs['exp_name'] +"_critic_local_"+ str(epoch+1)+'.pt')
        actor_targ.save(pytorch_save_path+"/targets/"+logger_kwargs['exp_name']+"_actor_targ_"+ str(epoch+1)+'.pt')
        critic_targ.save(pytorch_save_path+"/targets/"+ logger_kwargs['exp_name'] +"_critic_targ_"+ str(epoch+1)+'.pt')
        print("Pytorch models (epoch:" +str(epoch+1) +") saved in: " +str(pytorch_save_path) +
                        "\nexample(actor): "+pytorch_save_path+"/"+logger_kwargs['exp_name']+ str(epoch+1)+'.pt')

    def log_end_of_epoch():
        # Log info about epoch
        logger.log_tabular('Epoch', epoch+1)
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
    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

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

    def update(data):
        # gradient descent for critic/Q.
        critic_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        critic_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in critic.parameters():
            p.requires_grad = False

        # Gradient ascent from actor
        actor_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        actor_optimizer.step()

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
        should_render = ( (epoch+1) %render_test_freq == 0) 
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
            write_to_csv(pytorch_save_path+"/testperformance.csv",str(epoch+1)+','+str(j)+','+str(ep_ret)+','+str(ep_len)+','+str(d)+','+'\n')    
    #*                   End train functions                            *

    #*                 **START TRAINING LOOP**                          *
    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    ep_number = 0
    t = 0   #total steps taken
    
    # Training Loop: Run each epoch x number steps per epoch
    for epoch in range(epochs):
        for cur_step in range(steps_per_epoch):
            
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy (with some noise, via act_noise).
            #Total steps already taken = t(steps taken this epoch) + epoch*steps_per epoch
            if (t) > start_steps:
                a = get_action(o, act_noise)
            else:
                a = env.action_space.sample()#random exploration for start steps
                #a = np.random.randn(act_dim)(this instead?)
                
            # Step the env
            o2, r, d, _ = env.step(act_limit*a)
            #env.render()
            #time.sleep(.5)
            #print("********\nstep: "+str(t)+"Reward: "+str(r)+"\nobs2: "+str(o2))

            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==max_ep_len else d

            # Store experience to replay buffer
            replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2
            if t >= update_after and t % update_every == 0:
                for _ in range(update_every):
                    batch = replay_buffer.sample_batch(batch_size)
                    update(data=batch)
            
            t+=1
            
            #**          End of episode                 **
            if(d or ep_len==max_ep_len):
                #Log end of episode
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                write_to_csv(pytorch_save_path+"/trainperformance.csv", str(epoch)+','+str(ep_number)+','+str(ep_ret)+','+str(ep_len)+','+str(d)+',')
                
                #reset environment
                o, ep_ret, ep_len = env.reset(), 0, 0
                ep_number+=1            

        
        #**          End of epoch handling              **
        # Save model
        if ( (epoch+1) % save_freq == 0) or ( (epoch+1) == epochs):
            #logger.save_state({'env': env}, epoch) <this doesnt work with the rSoccer env
            savePtFiles()

        # Test the performance of the deterministic version of the agent.
        print("Testing")
        test_agent()
        print("Done\ntotal steps: " +str(t))
        log_end_of_epoch()
    #**          End of training              **
    print("Done training. Should have taken " +str(steps_per_epoch * epochs) + " steps. taken "+str(t))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='SSLGoToBallDDPG-v0')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    parser.add_argument('--test_file_name', type=str, default='') #Specify which actor to load for test. Turns off training
    parser.add_argument('--r_test', type=int, default='5')  #How many epochs between rendering test episodes duringtraining
    parser.add_argument('--spinup', type=bool, default=False) #Use modified spinningup actor-critic modules
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
        test(lambda : gym.make(args.env),exp_name=args.exp_name,env_name =args.env, max_ep_len = max_ep_st, actor_model_path=args.test_file_name, 
                                                        logger_kwargs=logger_kwargs,num_test_episodes=10)
    else:
        train(lambda : gym.make(args.env), env_name =args.env, 
                    max_ep_len = max_ep_st, actor_init=actor_init, critic_init=critic_init,
                         gamma=args.gamma, seed=args.seed,  epochs=args.epochs, 
                            logger_kwargs=logger_kwargs, render_test_freq=args.r_test)