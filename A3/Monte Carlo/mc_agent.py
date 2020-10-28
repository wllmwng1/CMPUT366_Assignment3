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
last_action = None
last_state = None

def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    #initialize the policy array in a smart way
    global R
    global Q
    global Episode
    global last_action
    global last_state
    last_action = 1
    last_state = 1
    Q = [[0 for x in range(num_total_states/2+1)] for y in range(num_total_states)]
    R = dict()
    Episode = list()
    for x in range(1,num_total_states+1):
        for y in range(min(x+1,num_total_states-x+2)):
            R[(x,y)] = [0,0]

def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts
    global last_state
    global last_action
    global Episode
    Episode = []
    action = np.random.randint(1,min(state[0]+1,num_total_states-state[0]+2))
    last_state = state[0]
    last_action = action
    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    global last_state
    global last_action
    global Episode
    Episode.append((last_state,last_action,reward))
    action = Q[state[0]-1].index(max(Q[state[0]-1])) + 1
    last_state = state[0]
    last_action = action
    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    global Episode
    # do learning and update pi
    Episode.append((last_state,last_action,reward))
    for e in Episode:
        (s,a,r) = e
        R[(s,a)] = [R[(s,a)][0]+reward,R[(s,a)][1]+1]
        Q[s-1][a-1] = R[(s,a)][0]/R[(s,a)][1]
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
