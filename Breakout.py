#!/usr/bin/env python
# Agent coded by Samiran Roy
# Will Play Breakout (Hopefully)
# 
# http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html


import sys
sys.path.append('/home/sushant/miniconda2/lib/python2.7/site-packages')

from random import randrange,random,sample
from ale_python_interface import ALEInterface
import numpy as np
import cv2
import tensorflow as tf
from collections import deque
from stack import stack
import numpy.core.numeric as _nx
from Nature_Network import *
import pickle




ale = ALEInterface()


# Get & Set the desired settings
ale.setInt('random_seed', 123)


# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = True
if USE_SDL:
  if sys.platform == 'darwin':
    import pygame
    pygame.init()
    ale.setBool('sound', False) # Sound doesn't work on OSX
  elif sys.platform.startswith('linux'):
    ale.setBool('sound', True)
  ale.setBool('display_screen', True)

# Load the ROM file
ale.loadROM("Breakout.bin")
print(ale.getMinimalActionSet())


  
session = tf.InteractiveSession(config=tf.ConfigProto(
    allow_soft_placement=True))

#session = tf.InteractiveSession()

s, deep_Q, h_fc1 = createNetwork()


# define the cost function


a = tf.placeholder("float", [None, 2])

y = tf.placeholder("float", [None])
deep_Q_action = tf.reduce_sum(tf.mul(deep_Q, a), reduction_indices = 1)
    
cost = tf.reduce_mean(tf.square(y - deep_Q_action))

train_step = tf.train.AdamOptimizer(1e-5).minimize(cost)



#Initializing Empty Replay Memory
Replay_Memory = deque()

terminal = ale.game_over()
screen = ale.getScreenGrayscale()
screen=cv2.resize(screen, (80, 80)) 


reward = ale.act(1);
ret, x_t = cv2.threshold(screen,1,255,cv2.THRESH_BINARY)
current_state = stack((x_t, x_t, x_t, x_t), axis = 2)




# Storing Results of each episode

Score={}


# saving and loading networks
saver = tf.train.Saver()
session.run(tf.initialize_all_variables())
checkpoint = tf.train.get_checkpoint_state("History")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(session, checkpoint.model_checkpoint_path)
    print("Loading:", checkpoint.model_checkpoint_path)




epsilon=1
t = 0
episode=0


# Time steps after which we start training
annealing_threshold=100000

lives=ale.lives()
total_reward=0

while 1:
    # choose an action epsilon greedily

    deep_Q_t = deep_Q.eval(feed_dict = {s : [current_state]})[0]

    a_t = np.zeros([2])
    action_index = 0
    if random() <= epsilon or t <= annealing_threshold:
        action_index = randrange(2)
        a_t[action_index] = 1
    else:
        action_index = np.argmax(deep_Q_t)

        a_t[action_index] = 1

    
    
    if t>annealing_threshold and t<(2*annealing_threshold):
        epsilon -= 0.9 / annealing_threshold
    

        # run the selected action and observe next state and reward
    for i in range(0, 4):   
 
      if i==1:
        ale.act(1)
      r_t=ale.act(action_index+3);

      total_reward+=r_t
      terminal = ale.game_over()
      screen = ale.getScreenGrayscale()
      screen=cv2.resize(screen, (80, 80)) 

      ret, x_t1 = cv2.threshold(screen,1,255,cv2.THRESH_BINARY)
      x_t1 = np.reshape(x_t1, (80, 80, 1))
      next_state = np.append(x_t1, current_state[:,:,1:], axis = 2)

      # Store phi(st),at,rt,phi(st+1) in Replay Memory where phi() preprocesses the state image
      Replay_Memory.append((current_state, a_t, r_t, next_state, terminal))
      # Bounding Replay Memory by 1 million frames
      if len(Replay_Memory) > annealing_threshold:
          Replay_Memory.popleft()

        
    if terminal==True:
        
        print("Episode ",episode," ended with reward ",total_reward," at timestep ",t, " and Q_max: " , np.max(deep_Q_t))
        
        Score[episode]=total_reward
        total_reward=0
        episode+=1
        ale.reset_game()

    # Populate the Replay Memory for 1 million frames, then start training
    if t==annealing_threshold:
        print("Stopped annealing, started training")
    if t > annealing_threshold:
        # Size of minibatch=32
        minibatch = sample(Replay_Memory, 32)


        # get the batch variables
        initial_state_sample = [d[0] for d in minibatch]
        a_batch = [d[1] for d in minibatch]
        r_batch = [d[2] for d in minibatch]
        next_state__sample = [d[3] for d in minibatch]

        y_updated = []  
        deep_Q_j1_batch = deep_Q.eval(feed_dict = {s : next_state__sample})
        for i in range(0, len(minibatch)):
            if minibatch[i][4]:
                y_updated.append(r_batch[i])
            else:
                y_updated.append(r_batch[i] + 0.95 * np.max(deep_Q_j1_batch[i]))

        # Update weights using gradient descent
        train_step.run(feed_dict = {
            y : y_updated,
            a : a_batch,
            s : initial_state_sample})



    current_state = next_state
    t += 1

    # Save weights and results every n iterations
    if t % 10000 == 0:
        saver.save(session, 'History/' + "Breakout" , global_step = t)
        pickle.dump( Score, open( "History/scores.p", "wb" ) )
        print("Saved Progess: Trained for "+str(t)+" iterations")   


    #print  t, "-   action: ", action_index, " reward: ", r_t," epsilon: ",epsilon, 


