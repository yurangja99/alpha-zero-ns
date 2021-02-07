import numpy as np
import matplotlib.pyplot as plt

class GameState:
  '''
  Custom state for Twelve Shogi game. 

  State: (4, 3, 18) np.array
    5 layers for black's king, jang, sang, ja, and hu. (minimum 0.0 to maximum 1.0) 
    5 layers for white's king, jang, sang, ja, and hu. (minimum 0.0 to maximum 1.0) 
    4 layers for black's slave king, jang, sang, ja. (minimum 0.0 to maximum 2.0) 
    4 layers for white's slave king, jang, sang, ja. (minimum 0.0 to maximum 2.0) 

  Observation: (4, 3, 18) np.array
    By default (turn == 0), observation = state. 
    If turn == 1, flip black and white values. 

  Action: (144, ) np.array
    Moving a piece on the table
      Choosing piece to move: 12 (4 x 3) 
      Choosing where to move: 8 (0 for front, and clockwise.) 
    Choose captured piece and drop it on the table
      Choosing piece to drop: 4 (king, jang, sang, ja) 
      Choosing where to drop: 12 (4 x 3)
  '''
  def __init__(self, state, turn):
    '''
    create a new state instance.

    Parameters
      state: np.array of (4, 3, 18)
      turn: int of 0 or 1

    Fields
      self.state: np.array. state
      self.turn: int. turn
      self.key: string key that expresses this state. (used in MCTS class)
      self.model_input: np array. turn-considered state. (see __to_model_input() method.)
      self.black_win: whether black wins
      self.white_win: whether white wins
      self.allowed_moves: list of lists of tuples. provide allowed actions for each pieces. 
    '''
    self.state = state
    self.turn = turn
    self.key = self.__generate_key(state, turn)
    self.model_input = self.__to_model_input()
    self.black_win = self.__black_win()
    self.white_win = self.__white_win()

    self.__allowed_moves = [
      [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)], # king (black)
      [(-1, 0), (0, 1), (1, 0), (0, -1)], # jang (black)
      [(-1, 1), (1, 1), (1, -1), (-1, -1)], # sang (black)
      [(-1, 0)], # ja (black)
      [(-1, 0), (-1, 1), (0, 1), (1, 0), (0, -1), (-1, -1)], # hu (black)
      [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)], # king (white)
      [(-1, 0), (0, 1), (1, 0), (0, -1)], # jang (white)
      [(-1, 1), (1, 1), (1, -1), (-1, -1)], # sang (white)
      [(1, 0)], # ja (white)
      [(-1, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]  # hu (white)
    ]
  
  def __to_model_input(self):
    '''
    Convert current state to fit model input. (shape of (4, 3, 18)) 
    result[i, j, k] with k % 2 == 0 for turn == 0, 
    result[i, j, k] with k % 2 == 1 for turn == 1. 

    Return
      return: converted np array for model input.
    '''
    if self.turn == 0:
      return np.copy(self.state)
    else:
      return np.flip(np.copy(self.state)[:, :, [5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 14, 15, 16, 17, 10, 11, 12, 13]], axis=[0, 1])

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
      else if mode == 'log', returns its state as list with 'K', 'J', 'S', 'j', 'H', and '-'.
      else if mode == 'terminal', prints its state as list with 'K', 'J', 'S', 'j', 'H', and '-'. 
      otherwise, plot it.
    '''
    if mode == 'log' or mode == 'terminal':
      # use logger to log state
      def piece(x):
        '''
        convert vector x to str

        Parameters
          x: np.array (10, )
        '''
        padding = ' '
        if np.any(x[[0, 1, 2, 3, 4]] > 0):
          # black piece
          padding = ' '
        elif np.any(x[[5, 6, 7, 8, 9]] > 0):
          # white piece
          padding = '_'
        if np.any(x[[0, 5]] > 0):
          # king piece
          return padding + 'K' + padding
        elif np.any(x[[1, 6]] > 0):
          # jang piece
          return padding + 'J' + padding
        elif np.any(x[[2, 7]] > 0):
          # sang piece
          return padding + 'S' + padding
        elif np.any(x[[3, 8]] > 0):
          # ja piece (black, and white)
          return padding + 'j' + padding
        elif np.any(x[[4, 9]] > 0):
          # hu piece (black, and white)
          return padding + 'H' + padding
        return padding + '-' + padding
      def captured_pieces(x):
        '''
        convert captured pieces as str

        Parameters
          x: np.array of (4,)
        '''
        return 'K ' * x[0] + 'J ' * x[1] + 'S ' * x[2] + 'j ' * x[3]
      if mode == 'log':
        # first, render pieces on the table. 
        for i in range(4):
          logger.info([piece(pos) for pos in self.state[i, :, :10]])
        # second, render captured pieces. 
        logger.info('Player1(below)\'s Captured: ' + captured_pieces(self.state[0, 0, 10:14]))
        logger.info('Player2(upper)\'s Captured: ' + captured_pieces(self.state[0, 0, 14:18]))
        logger.info('--------------------------------------------------------')
      else:
        # first, render pieces on the table. 
        for i in range(4):
          print([piece(pos) for pos in self.state[i, :, :10]])
        # second, render captured pieces. 
        print('Player1(below)\'s Captured: ' + captured_pieces(self.state[0, 0, 10:14]))
        print('Player2(upper)\'s Captured: ' + captured_pieces(self.state[0, 0, 14:18]))
        print('--------------------------------------------------------')
    else:
      def piece(x):
        '''
        convert vector x to (5, 5, 3) np array

        Parameters
          x: np.array of (10, )
        '''
        piece_pixels = np.zeros((5, 5, 3), dtype=np.int32)
        if np.any(x[[0, 1, 2, 3, 4]] > 0):
          # black piece
          piece_pixels[:, :, 1] = np.max(x[[0, 1, 2, 3, 4]]) * 127
        elif np.any(x[[5, 6, 7, 8, 9]] > 0):
          # white piece
          piece_pixels[:, :, 0] = np.max(x[[5, 6, 7, 8, 9]]) * 127
        if np.any(x[[0, 5]] > 0):
          # king piece
          piece_pixels[[0, 0, 0, 2, 2, 4, 4, 4], [0, 2, 4, 0, 4, 0, 2, 4], :] = 200
        elif np.any(x[[1, 6]] > 0):
          # jang piece
          piece_pixels[[0, 2, 2, 4], [2, 0, 4, 2], :] = 200
        elif np.any(x[[2, 7]] > 0):
          # sang piece
          piece_pixels[[0, 0, 4, 4], [0, 4, 0, 4], :] = 200
        elif x[3] > 0:
          # ja piece (black)
          piece_pixels[0, 2, :] = 200
        elif x[8] > 0:
          # ja piece (white)
          piece_pixels[4, 2, :] = 200
        elif x[4] > 0:
          # hu piece (black)
          piece_pixels[[0, 0, 0, 2, 2, 4], [0, 2, 4, 0, 4, 2], :] = 200
        elif x[9] > 0:
          # hu piece (white)
          piece_pixels[[0, 2, 2, 4, 4, 4], [2, 0, 4, 0, 2, 4], :] = 200
        return piece_pixels
      def captured(x, turn, cap_type):
        '''
        convert captured piece x to (5, 5, 3) np array

        Parameters
          x: np.int32 (0.0 to 2.0)
          turn: 0 or 1
          cap_type: int. one of 0 for 'king', 1 for 'jang', 2 for 'sang', and 3 for 'ja'. 
        '''
        piece_pixels = np.zeros((5, 5, 3), dtype=np.int32)
        if x > 0:
          # black or white
          piece_pixels[:, :, 1 - turn] = x * 127
          if cap_type == 0:
            # captured king
            piece_pixels[[0, 0, 0, 2, 2, 4, 4, 4], [0, 2, 4, 0, 4, 0, 2, 4], :] = 200
          elif cap_type == 1:
            # captured jang
            piece_pixels[[0, 2, 2, 4], [2, 0, 4, 2], :] = 200
          elif cap_type == 2:
            # captured sang
            piece_pixels[[0, 0, 4, 4], [0, 4, 0, 4], :] = 200
          elif cap_type == 3 and turn == 0:
            # captured ja (black side)
            piece_pixels[0, 2, :] = 200
          elif cap_type == 3 and turn == 1:
            # captured ja (white side)
            piece_pixels[4, 2, :] = 200
        return piece_pixels
      # initialize result pixels
      pixels = np.zeros((4 * 5 + 3, 7 * 5 + 6, 3), dtype=np.int32)
      # first, render pieces on the table. 
      for i in range(4):
        pixels[i * 6 : (i + 1) * 6 - 1, 0:5, :] = captured(self.state[0, 0, 14 + i], 1, i)
        pixels[i * 6 : (i + 1) * 6 - 1, 36:41, :] = captured(self.state[0, 0, 10 + i], 0, i)
        for j in range(3):
          pixels[i * 6 : (i + 1) * 6 - 1, (j + 2) * 6 : (j + 3) * 6 - 1, :] = piece(self.state[i, j, :10])
      # second, render captured pieces. 
      if mode == 'rgb_array':
        # returns raw rgb state
        return pixels
      else:
        # plot state
        plt.imshow(pixels)
        plt.show(block=False)
        plt.pause(0.1)

  def __black_win(self):
    '''
    return whether black captured white's king or black's king survived one turn in white side, or not. 
    '''
    return self.state[0, 0, 10] > 0 or (np.any(self.state[0, :, 0] > 0) and self.turn == 0)
  
  def __white_win(self):
    '''
    return whether white captured black's king or white's king survived one turn in black side, or not. 
    '''
    return self.state[0, 0, 14] > 0 or (np.any(self.state[3, :, 5] > 0) and self.turn == 1)

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
      action: integer value (12 x 8 + 4 x 12 = 144 actions)
    
    Return
      return: time_step (state, reward, done, additional info)

    0. find out whether the action is move action, or drop action. 
    1. try to do the action, and get success or not (True or False)
    2. switch turn. 
    3. if False, return -1 (forbidden move or drop!)
    4. if True, check win conditions (king captured or king survived one turn at opposite side)
    5. if no one wins, switch turn and return transition

    reward for winning: -1 for lose and 1 for win
    '''
    # decide move or drop. 
    if action < 96:
      # set move action. if turn == 1, flip
      if self.turn == 0:
        base_r = (action // 8) // 3
        base_c = (action // 8) % 3
        direction = action % 8
      else:
        base_r = 3 - (action // 8) // 3
        base_c = 2 - (action // 8) % 3
        direction = (action % 8 + 4) % 8
      #print('\nfrom ({}, {}), direction {}'.format(base_r, base_c, direction))
      # check whether the piece exists and belongs to self.turn
      my_pieces_on_base = np.where(self.state[base_r, base_c, self.turn * 5 : self.turn * 5 + 5] > 0)[0]
      #print('my pieces on base: {}'.format(my_pieces_on_base))
      if len(my_pieces_on_base) != 1:
        # lose. no pieces belong to the player or invalid situation. 
        return self, -1, True, None

      # check whether the piece can move as given action
      # (consider the piece's type, table boundary, and target position's piece)
      dr = 1 if direction in [3, 4, 5] else (-1 if direction in [7, 0, 1] else 0)
      dc = 1 if direction in [1, 2, 3] else (-1 if direction in [5, 6, 7] else 0)
      if (dr, dc) not in self.__allowed_moves[self.turn * 5 + my_pieces_on_base[0]]:
        # lose. the type of piece can't move to (dr, dc)
        return self, -1, True, None
      target_r = base_r + dr
      target_c = base_c + dc
      #print('target ({}, {})'.format(target_r, target_c))
      if target_r < 0 or target_r > 3 or target_c < 0 or target_c > 2:
        # lose. move outside the boundary
        return self, -1, True, None
      my_pieces_on_target = np.where(self.state[target_r, target_c, self.turn * 5 : self.turn * 5 + 5] > 0)[0]
      op_pieces_on_target = np.where(self.state[target_r, target_c, (1 - self.turn) * 5 : (1 - self.turn) * 5 + 5] > 0)[0]
      #print('my pieces on target: {}'.format(my_pieces_on_target))
      #print('op pieces on target: {}'.format(op_pieces_on_target))
      if len(my_pieces_on_target) > 0:
        # lose. can't move to the place that my piece placed on. 
        return self, -1, True, None

      # move and change state 
      # (clear past position, set new position, and capture if there were enemy piece)
      next_board = np.copy(self.state)
      if len(op_pieces_on_target) > 0:
        # capture pieces (should be one)
        for op_piece in op_pieces_on_target:
          next_board[target_r, target_c, (1 - self.turn) * 5 + op_piece] = 0
          next_board[:, :, 10 + self.turn * 4 + min(op_piece, 3)] += 1 # min(op_piece, 3) changes hu to ja.
      next_board[base_r, base_c, self.turn * 5 + my_pieces_on_base[0]] = 0
      next_board[target_r, target_c, self.turn * 5 + my_pieces_on_base[0]] = 1
      
    else:
      # set drop action. if turn == 1, flip
      if self.turn == 0:
        base_layer = (action - 96) // 12
        target_r = ((action - 96) % 12) // 3
        target_c = ((action - 96) % 12) % 3
      else:
        base_layer = (action - 96) // 12
        target_r = 3 - ((action - 96) % 12) // 3
        target_c = 2 - ((action - 96) % 12) % 3
      
      # check whether the base layer's value is over zero. 
      if self.state[0, 0, 10 + self.turn * 4 + base_layer] == 0:
        # lose. no captured to drop. 
        return self, -1, True, None
      
      # check whether the new position is empty and not enemy's side
      if np.any(self.state[target_r, target_c, :10] > 0):
        # lose. can't drop at non-empty side. 
        return self, -1, True, None
      if target_r == self.turn * 3:
        # lose. can't drop at opponent's side. 
        return self, -1, True, None

      # drop and change state (set new position)
      next_board = np.copy(self.state)
      next_board[:, :, 10 + self.turn * 4 + base_layer] -= 1
      next_board[target_r, target_c, self.turn * 5 + base_layer] = 1

    # check whether the game is over
    next_state = GameState(next_board, 1 - self.turn)
    if next_state.black_win:
      return next_state, 1 - 2 * next_state.turn, True, None
    elif next_state.white_win:
      return next_state, -1 + 2 * next_state.turn, True, None

    # return new state with reward 0.0
    return next_state, 0, False, None

class GameEnv:
  '''
  Custom environment for Twelve Shogi game. 

  Custom rules from original Twelve Shogi:
  1. the turn starts with 0
  2. agent can do one of 192 actions: take one of 16 candidates, and move to 12 targets. 
  3. target is capturing opponent's king, or move own king to opponent's side and survive for one turn. 

  State: (4, 3, 18) np.array
    5 layers for black's king, jang, sang, ja, and hu. (minimum 0.0 to maximum 1.0) 
    5 layers for white's king, jang, sang, ja, and hu. (minimum 0.0 to maximum 1.0) 
    4 layers for black's slave king, jang, sang, ja. (minimum 0.0 to maximum 2.0) 
    4 layers for white's slave king, jang, sang, ja. (minimum 0.0 to maximum 2.0) 

  Observation: (4, 3, 18) np.array
    By default (turn == 0), observation = state. 
    If turn == 1, flip black and white values. 

  Action: (144, ) np.array
    Moving a piece on the table
      Choosing piece to move: 12 (4 x 3) 
      Choosing where to move: 8 (0 for front, and clockwise.) 
    Choose captured piece and drop it on the table
      Choosing piece to drop: 4 (king, jang, sang, ja) 
      Choosing where to drop: 12 (4 x 3)
  
  Reward: +1 for win, -1 for lose, and -1 for invalid move. 
  '''
  def __init__(self):
    '''
    create environment for Twelve Shogi and init some variables
    
    Fields
      self.name: name of the game. 
      self._state: GameState instance.
      self._turn: int of 0 or 1
      self._episode_ended: whether the episode is ended, or not
      self._observation_spec: dictionary. info about state
      self._action_spec: dictionary. info about action
    '''
    self.name = 'twelve_shogi'
    self._turn = 0
    self._state = GameState(self.__init_state(), self._turn)
    self._episode_ended = False
    self._observation_spec = { 'shape': self._state.state.shape, 'dtype': np.int32, 'min': 0, 'max': 2 }
    self._action_spec = { 'shape': (1,), 'dtype': np.int32, 'min': 0, 'max': 143 }

  def __init_state(self):
    '''
    return the initial np.array state
    '''
    init_state = np.zeros((4, 3, 18), dtype=np.int32)
    init_state[[0, 0, 0, 1, 2, 3, 3, 3], [0, 1, 2, 1, 1, 0, 1, 2], [6, 5, 7, 8, 3, 2, 0, 1]] = 1
    return init_state
    
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
    self._state = GameState(self.__init_state(), self._turn)
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
    move_action_values = np.copy(action_values[:96]).reshape((4, 3, 8))
    drop_action_values = np.copy(action_values[96:]).reshape((4, 4, 3))

    flipped_move_action_values = move_action_values[:, :, [0, 7, 6, 5, 4, 3, 2, 1]]
    flipped_move_action_values = np.flip(flipped_move_action_values, axis=1)
    flipped_drop_action_values = np.flip(drop_action_values, axis=-1)
    
    flipped_action_values = np.concatenate((
      flipped_move_action_values.reshape((96, )), 
      flipped_drop_action_values.reshape((48, ))), 
      axis=0)
    return [(state, action_values), (state.fliplr(), flipped_action_values)]

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