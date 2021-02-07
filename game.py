import numpy as np
import matplotlib.pyplot as plt

class GameState:
  '''
  Custom state for connected four game.

  1. State: 6x7x2 np array
  2. Observation: this is board game, so observation ~= state and have slight difference.

  Observation: [
    [state[0, 0], state[0, 1], state[0, 2], state[0, 3], state[0, 4], state[0, 5], state[0, 6]],
    [state[1, 0], state[1, 1], state[1, 2], state[1, 3], state[1, 4], state[1, 5], state[1, 6]],
    [state[2, 0], state[2, 1], state[2, 2], state[2, 3], state[2, 4], state[2, 5], state[2, 6]],
    [state[3, 0], state[3, 1], state[3, 2], state[3, 3], state[3, 4], state[3, 5], state[3, 6]],
    [state[4, 0], state[4, 1], state[4, 2], state[4, 3], state[4, 4], state[4, 5], state[4, 6]],
    [state[5, 0], state[5, 1], state[5, 2], state[5, 3], state[5, 4], state[5, 5], state[5, 6]]
  ]
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
    shape of (6, 7, 2), with return[i, j, 0] for current turn, 
    and return[i, j, 1] for opponent turn.

    Return
      return: converted np array for model input.
    '''
    if self.turn == 0:
      return np.copy(self.state)
    else:
      return np.flip(self.state, axis=-1)

  def __generate_key(self, state, turn):
    '''
    generate unique key for the Node (state)

    Parameters
      state: np array
      turn: 0 or 1

    Return
      return: string value
    '''
    def generate_state(state):
      shape = state.shape
      if len(shape) == 1:
        return ''.join(map(str, state))
      elif len(shape) > 1:
        return ''.join([generate_state(state[i]) for i in range(shape[0])])
      else:
        raise Exception('Bad State Shape: {}'.format(shape))
    return generate_state(state) + str(turn)

  def render(self, mode='rgb_array', logger=None):
    '''
    Returns different results according to mode
      if mode == rgb_array, returns its state as RGB scale (R: opponent, G: agent, B: nothing)
      else if mode == 'log', logs (into logger) its state as list with 'O', 'X', and '-'.
      else if mode == 'terminal', (at terminal) logs its state as list with 'O', 'X', and '-'. 
      otherwise, plot it.
    '''
    pixels = np.zeros((6, 7, 3), dtype=np.int32)
    pixels[:, :, 1] = self.state[:, :, 0] * 255
    pixels[:, :, 0] = self.state[:, :, 1] * 255
    if mode == 'rgb_array':  
      # returns raw rgb state
      return pixels
    elif mode == 'log':
      # use logger to log state
      piece = lambda x: 'O' if x[1] > 0 else ('X' if x[0] > 0 else '-')
      for i in range(6):
        logger.info([piece(pixels[i, j]) for j in range(7)])
      logger.info('--------------------------------------------------------')
    elif mode == 'terminal':
      # print terminal
      piece = lambda x: 'O' if x[1] > 0 else ('X' if x[0] > 0 else '-')
      for i in range(6):
        print([piece(pixels[i, j]) for j in range(7)])
      print('--------------------------------------------------------')
    elif mode == 'human':
      # plot state
      plt.imshow(pixels)
      plt.show(block=False)
      plt.pause(0.1)

  def __max_blank_row(self, column):
    '''
    for given column, return maximum blank index. 
    (rock falls to the lowest grid of the column)

    Parameters
      column: 1 dimension vector

    Return
      return: integer (row number)
    '''
    if np.sum(column[0]) > 0:
      return -1
    else:
      return np.max(np.where(np.sum(column, axis=-1) == 0))

  def __black_win(self):
    '''
    return whether black completed 4 consecutive rocks 
    in a row, column, or diagonal
    '''
    for r in range(6):
      for c in range(7):
        if (c < 7 - 3 and np.sum(self.state[r, c:c+4, 0]) == 4) or \
           (r < 6 - 3 and np.sum(self.state[r:r+4, c, 0]) == 4) or \
           (r < 6 - 3 and c < 7 - 3 and np.trace(self.state[r:r+4, c:c+4, 0]) == 4) or \
           (r < 6 - 3 and c < 7 - 3 and np.trace(np.fliplr(self.state[r:r+4, c:c+4, 0])) == 4):
          return True
    return False
  
  def __white_win(self):
    '''
    return whether white completed 4 consecutive rocks 
    in a row, column, or diagonal
    '''
    for r in range(6):
      for c in range(7):
        if (c < 7 - 3 and np.sum(self.state[r, c:c+4, 1]) == 4) or \
           (r < 6 - 3 and np.sum(self.state[r:r+4, c, 1]) == 4) or \
           (r < 6 - 3 and c < 7 - 3 and np.trace(self.state[r:r+4, c:c+4, 1]) == 4) or \
           (r < 6 - 3 and c < 7 - 3 and np.trace(np.fliplr(self.state[r:r+4, c:c+4, 1])) == 4):
          return True
    return False
  
  def __draw(self):
    '''
    return whether draw situation occurred. 
    In this game, draw occurred when all columns are full. 
    '''
    for a in range(7):
      if self.__max_blank_row(self.state[:, a]) >= 0:
        return False
    return True

  def fliplr(self):
    '''
    flip left-right and return the state

    Return
      return: GameState instance with flipped board. 
    '''
    flipped_state = np.fliplr(np.copy(self.state))
    return GameState(flipped_state, self.turn)

  def step(self, action):
    '''
    given user's action, manage interaction between agent and the environment.

    Parameters
      action: integer value
    
    Return
      return: time_step (state, reward, done, additional info)

    0. if selected column is full, (user with the turn) loses
    2. otherwise, put the rock to the column
    3. check whether the (user with the turn) win or not
    4. if (user with the turn) didn't win, switch turn and return transition

    reward for winning: -1 for lose and 1 for win
    '''
    # if selected column is full, immediately end the episode with reward = 1 - turn*2
    row = self.__max_blank_row(self.state[:, action])
    if row < 0:
      return (self, -1, True, None)

    # otherwise, change state according to action
    copy_state = np.copy(self.state)
    copy_state[row, action, self.turn] = 1
    next_state = GameState(copy_state, 1 - self.turn)

    # check whether the game is over
    # reward is -1 because the opponent loses. 
    # (user with the turn win)
    if next_state.black_win or next_state.white_win:
      return (next_state, -1, True, None)
    # case draw.
    if next_state.draw:
      return (next_state, 0, True, None)

    # return new state with reward 0.0
    return (next_state, 0, False, None)

class GameEnv:
  '''
  Custom environment for connected four game. 

  Custom rules from original connected four:
  1. the turn starts with 0
  2. agent can do one of seven actions: 0 ~ 6th column
  3. target is making four connected rocks in a row, column, or diagonal

  About Environment:
  - action: 0, 1, 2, 3, 4, 5, 6
  - observation: GameState instance. 
    It contains np array of 6x7x2 (6 rows, 7 columns, and 2 kinds of rocks)
  - reward: +1 for win, -1 for lose, and -1 for selecting full column
  '''
  def __init__(self):
    '''
    create environment for connected four and init some variables
    
    Fields
      self.name: name of the game. 
      self._state: GameState instance.
      self._turn: int of 0 or 1
      self._episode_ended: whether the episode is ended, or not
      self._observation_spec: dictionary. info about state
      self._action_spec: dictionary. info about action
    '''
    self.name = 'connected_four'
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
    For example, in this game, left-right flipped state and flipped action value 
    works same to each other. 

    Parameters
      state: GameState instance.
      action: np array of action values. 
    
    Return
      return: list of tuple (state, action_values)
    '''
    return [(state, action_values), (state.fliplr(), np.flip(action_values))]

  def step(self, action):
    '''
    given agent's action, manage interaction between agent and the environment.

    Parameters
      action: integer value
    
    Return
      return: time_step (state, reward, done, additional info)

    0. if episode is already ended, return initialized state
    1. apply action to the state, and get result
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