import numpy as np
import matplotlib.pyplot as plt

class GameState:
  '''
  Custom state for custom game.
  '''
  def __init__(self, state, turn):
    '''
    create a new state instance.

    Parameters
      state: np array of 6x7x2
      turn: int of 0 or 1

    Fields
      self.state: np.array. state
      self.turn: int. turn
      self.key: string key that expresses this state. (used in MCTS class)
      self.model_input: np array. turn-considered state. (see __to_model_input() method.)
      self.black_win: whether black wins
      self.white_win: whether white wins
      self.draw: whether draw
    '''
    self.state = state
    self.turn = turn
    self.key = self.__generate_key(state, turn)
    self.model_input = self.__to_model_input()
    self.black_win = self.__black_win()
    self.white_win = self.__white_win()
    self.draw = self.__draw()
  
  def __to_model_input(self):
    '''
    Convert current state to fit model input. 

    Return
      return: converted np array for model input.
    '''
    pass

  def __generate_key(self, state, turn):
    '''
    generate unique key for the Node (state)

    Parameters
      state: np array
      turn: 0 or 1

    Return
      return: string value
    '''
    pass

  def render(self, mode='rgb_array', logger=None):
    '''
    Returns different results according to mode: rgb_array, log, terminal, and human. 
    '''
    if mode == 'rgb_array':  
      pass
    elif mode == 'log':
      pass
    elif mode == 'terminal':
      pass
    elif mode == 'human':
      pass

  def __black_win(self):
    '''
    return whether black wins or not. 
    '''
    pass
  
  def __white_win(self):
    '''
    return whether white wins or not. 
    '''
    pass
  
  def __draw(self):
    '''
    return whether draw situation occurred or not. 
    '''
    pass

  def step(self, action):
    '''
    given user's action, manage interaction between agent and the environment.

    Parameters
      action: integer value
    
    Return
      return: time_step (state, reward, done, additional info)

    reward for winning: -1 for lose and 1 for win (to next turn)
    '''
    pass

class GameEnv:
  '''
  Custom environment for custom game. 
  '''
  def __init__(self):
    '''
    create environment for custom game and init some variables
    
    Fields
      self.name: name of the game. 
      self._state: GameState instance.
      self._turn: int of 0 or 1
      self._episode_ended: whether the episode is ended, or not
      self._observation_spec: dictionary. info about state
      self._action_spec: dictionary. info about action
    '''
    self.name = 'custom'
    self._turn = 0
    self._state = GameState(np.zeros((6, 7, 2), dtype=np.int32), self._turn)
    self._episode_ended = False
    self._observation_spec = { 'shape': self._state.state.shape, 'dtype': np.int32, 'min': 0, 'max': 1 }
    self._action_spec = { 'shape': (1,), 'dtype': np.int32, 'min': 0, 'max': 6 }

  def action_spec(self):
    '''
    return the environment's action spec
    '''
    return self._action_spec
  
  def observation_spec(self):
    '''
    return the environment's observation spec
    '''
    return self._observation_spec
  
  def state(self):
    '''
    return state of GameState
    '''
    return self._state

  def turn(self):
    '''
    return turn of int
    '''
    return self._turn

  def reset(self):
    '''
    reset the environment. init state, turn, and episode_ended. 
    finally, return start time_step (state, reward, done, additional info)
    '''
    self._turn = 0
    self._state = GameState(np.zeros((6, 7, 2), dtype=np.int32), self._turn)
    self._episode_ended = False
    return (self._state, 0.0, self._episode_ended, None)

  def identities(self, state, action_values):
    '''
    Returns identities of state - action value pair. 

    Parameters
      state: GameState instance.
      action: np array of action values. 
    
    Return
      return: list of tuple (state, action_values)
    '''
    return [(state, action_values)]

  def step(self, action):
    '''
    given agent's action, manage interaction between agent and the environment.

    Parameters
      action: integer value
    
    Return
      return: time_step (state, reward, done, additional info)
    '''
    # if episode is already ended, reset
    if self._episode_ended:
      return self.reset()
    
    # apply action to the state, and get results
    next_state, value, done, info = self._state.step(action)
    self._state = next_state
    self._turn = 1 - self._turn
    self._episode_ended = done
    return (self._state, value, done, info)