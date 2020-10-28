#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017

"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle

num_total_states = 99
last_action = 1
last_state = 1
episode = []

def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    #initialize the policy array in a smart way
    global pi
    global Q
    global R
    pi = []
    Q = [[0.0 for x in range(num_total_states/2+1)]for y in range(num_total_states)]
    R = dict()
    for s in range(1,num_total_states+1):
        pi.append(np.random.randint(1,min(s+1,num_total_states-s+2)))
        #Q.append(np.zeros(min(s+1,num_total_states-s+2)))
        for i in range(min(s+1,num_total_states-s+2)):
            R[(s,i)] = [0,0]

def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts
    action = np.random.randint(1,min(state+1,num_total_states-state+2))
    last_action = action
    last_state = state
    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    episode.append((last_state,last_action,reward))
    action = pi[state[0]-1]
    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    episode.append((last_state,last_action,reward))
    end = reward
    for e in range(len(episode)):
        (state,action,reward) = episode[e]
        R[(state,action)] = [int(R[(state,action)][0])+end,int(R[(state,action)][1])+1]
        Q[state][action-1] = sum(R[(state,action)])/len(R[(state,action)])
    for i in range(len(pi)):
        pi[i] = Q[i].index(max(Q[i])) + 1
    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global Q
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
            return pickle.dumps(np.max(Q, axis=1), protocol=0)
    else:
        return "I don't know what to return!!"
# agent_init()
# episode = [(1,1,0),(2,2,0),(4,4,0),(8,8,0),(16,8,0),(8,8,0),(16,16,0),(32,32,0)]
# last_state = 64
# last_action = 36
# agent_end(1.0)
