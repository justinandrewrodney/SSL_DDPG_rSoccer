# Setup
You can use the environments and training script with rSoccer

# SSL_DDPG_rSoccer
# Requirements
Spinning up: https://spinningup.openai.com/en/latest/user/installation.html

rSoccer: https://github.com/robocin/rSoccer

You will also need any of their dependencies.

# System
My system is running

**Ubuntu 20.04 LTS** (Through WSL2)

**Python 3.7.11** (I believe you need python 3.6=< version <3.8 to get spinning up to work.

**Pytorch version 1.3.1**

The version requirements for some of the packages don't play nicely if not correct, so make sure you closely follow dependencies.
# Description
This project provides a python program for training robot skills for **Small Sized League(SSL)** in the **rSoccer** environment utilizing the **Deep Deterministic Policy Gradient** algorithm, building upon and modifying the **spinningup** DDPG example. 
The trained models are saved as **Torch Scripts** so that they may be utilized in different systems written in different languages. 
This also provides a modified SSLGoToBall environment that is modified for actions and states that are using the robots orientation as the starting frame, as well as different reward functions.

If the training script in the top folder does not work, the ones in old_src will most definitely. The one in the top of the directory has less changes and is more similar to the spinningup script, but I have not had time to completely verify it.