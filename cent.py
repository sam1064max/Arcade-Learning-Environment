import sys
from random import randrange,random,sample
from ale_python_interface import ALEInterface
import numpy as np
import time

def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

ale = ALEInterface()

# Get & Set the desired settings
ale.setInt('random_seed', 123)

ballx = 99
bally = 101


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
  
ale.loadROM("space_invaders.bin")
print ale.getMinimalActionSet()

ram_size = ale.getRAMSize()
ram=np.zeros((ram_size),dtype=np.uint8)
ale.getRAM(ram)
print ram[54:108]



(screen_width,screen_height) =ale.getScreenDims()
screen_data=np.zeros(screen_width*screen_height,dtype=np.uint32)

legal_actions = ale.getLegalActionSet()

# Play 10 episodes
for episode in xrange(10):
  total_reward = 0
  a=4
  while not ale.game_over():
    time.sleep(0.1)
    a = legal_actions[randrange(len(legal_actions))]
    reward = ale.act(a);
    total_reward += reward
    temp = [i for i in ram]
    ale.getRAM(ram)

      #print ram
    # print [temp[i]-ram[i] for i in range(len(ram))] 
    ale.getScreen(screen_data)
    print screen_data
  print 'Episode', episode, 'ended with score:', total_reward
  ale.reset_game()
