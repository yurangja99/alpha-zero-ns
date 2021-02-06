import numpy as np
from collections import deque

class Memory:
  '''
  Memory class stores history of matches in short-term and long-term. 
  Memory size is usually fixed to config.MEMORY_SIZE. 
  Short-term memory stores history of one episode. 
  Long-term memory stores history of multiple episodes. 
  (I commit short-term memories to long-term memory.)
  '''
  def __init__(self, memory_size):
    '''
    create new Memory instance. 

    Parameters
      memory_size: int. size of maximum memory
    
    Fields
      self.memory_size: maximum memory size
      self.short_term_memory: deque. short term memory of one episode
      self.long_term_memory: deque. long term memory of multiple episodes
    '''
    self.memory_size = memory_size
    self.short_term_memory = deque(maxlen=self.memory_size)
    self.long_term_memory = deque(maxlen=self.memory_size)
  
  def commit_short_term_memory(self, items):
    '''
    Add new items to short-term memory. 

    Items are stored in tuples. 
    - state: GameState instance
    - key: string
    - action_probs: np.array float
    - turn: 0 or 1
    - value: reward of the episode (not set immediately, set after episode ends)

    Parameters
      items: list of tuple (state, action_probs)
        state: GameState instance. 
        action_probs: np array of action values. 
    '''
    for state, action_probs in items:
      self.short_term_memory.append({
        'state': state, 
        'action_probs': action_probs, 
        'turn': state.turn,
        'value': None
      })

  def commit_long_term_memory(self):
    '''
    Add all items in short-term memory to long-term memory. 
    After that, clear short-term memory. 
    '''
    for item in self.short_term_memory:
      self.long_term_memory.append(item)
    self.clear_short_term_memory()

  def clear_short_term_memory(self):
    '''
    Clear short-term memory
    '''
    self.short_term_memory = deque(maxlen=self.memory_size)
