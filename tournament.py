import tqdm
import matplotlib.pyplot as plt

class Tournament:
  '''
  in Tournament class, two agents can play against each other. 
  Tournament includes multiple episodes, and returns the statistics of the results. 
  '''

  def __init__(self, env, agent1, agent2, episodes, logger, turns_until_tau0, memory=None, render_option=None):
    '''
    create new Tournament instance. 

    Parameters
      env: GameEnv instance. 
      agent1: User or Agent instance. The first agent. 
      agent2: User or Agent instance. Second agent. 
      episodes: int. number of episodes
      logger: non-nullable logger in logger.py. which logger to write. 
      turns_until_tau0: int. maximum turn to take greedy action. 
      memory: nullable Memory instance. memory to save the moves.
      render: str. render option. 
    
    Fields
      env: GameEnv instance to play
      agent1, agent2: agents
      episodes: episodes
      logger: logger
      turns_until_tau0: max turn to take greedy action. 
      memory: memory instance
      agent_scores: dict of {'agent1': 0, 'draw': 1, 'agent2': 3}
      stone_scores: dict of {'0': 0, 'draw': 1, '1': 3}
      render_option: render option. 
    '''
    self.env = env
    self.agent1 = agent1
    self.agent2 = agent2
    self.episodes = episodes
    self.logger = logger
    self.turns_until_tau0 = turns_until_tau0
    self.memory = memory # nullable
    self.render_option = render_option
    
    self.agent_scores = { self.agent1.name: 0, 'draw': 0, self.agent2.name: 0 }
    self.stone_scores = { '0': 0, 'draw': 0, '1': 0 }

  def reset(self, agent1=None, agent2=None):
    '''
    Reset this Tournament instance's results. 

    Parameters
      agent1: Agent/User instance. optional.  
      agent2: Agent/User instance. optional. 
    '''
    if agent1 != None:
      self.agent1 = agent1
    if agent2 != None:
      self.agent2 = agent2
    self.agent_scores = { self.agent1.name: 0, 'draw': 0, self.agent2.name: 0 }
    self.stone_scores = { '0': 0, 'draw': 0, '1': 0 }

  def run_tournament(self):
    '''
    Run tournament with some options given as parameters of __init__, 
    and return the statistics. 

    Return
      return: tuple of dicts. (agent_scores, stone_scores)
    '''
    t_range = tqdm.trange(self.episodes)
    for episode_cnt in t_range:
      self.logger.info('******************')
      self.logger.info('TOURNAMENT {} / {}'.format(episode_cnt, self.episodes))
      self.logger.info('******************')
      
      # initialize an episode
      state, value, done, _ = self.env.reset()
      time_cnt = 0

      # initialize two agents' MCTS
      self.agent1.initialize_mcts()
      self.agent2.initialize_mcts()

      # start an episode
      tau = 1
      while not done:
        # update tau if turns_until_tau0 expired. 
        if time_cnt >= self.turns_until_tau0:
          tau = 0
        
        # render if self.render_option is enabled. 
        if self.render_option != None:
          state.render(self.render_option)

        # in even episodes, agent1 is first and agent2 is second.
        # in odd episodes, agent1 is second and agent2 is first.
        # (let two agents play both first and second player!)
        if state.turn == (episode_cnt % 2):
          # agent1 acts
          action, pi, MCTS_value, NN_value = self.agent1.act(state, tau)
        else:
          # agent2 acts
          action, pi, MCTS_value, NN_value = self.agent2.act(state, tau)
        
        # write memory
        if self.memory != None:
          self.memory.commit_short_term_memory(self.env.identities(state, pi))
        
        self.logger.info('AT {}TH TURN ({})'.format(time_cnt, self.agent1.name if state.turn == (episode_cnt % 2) else self.agent2.name))
        self.logger.info('ACTION: {}'.format(action))
        self.logger.info('PI: {}'.format(pi))
        self.logger.info('MCTS_VALUE: {}'.format(MCTS_value))
        self.logger.info('NN_VALUE: {}'.format(NN_value))
        self.logger.info('********************************************************')

        # do the action
        state, value, done, _ = self.env.step(action)
        state.render('log', self.logger)

        # increase time_cnt and continue
        time_cnt += 1
      
      # render if self.render_option is enabled. 
      if self.render_option != None:
        state.render(self.render_option)
        if self.render_option == 'human':
          plt.close('all')

      # after the episode ends, fill the memory. 
      if self.memory != None:
        for move in self.memory.short_term_memory:
          move['value'] = value * (1 if move['turn'] == state.turn else -1)
        self.memory.commit_long_term_memory()
      
      # log result and update scores. 
      if value > 0:
        winner = self.agent1 if state.turn == (episode_cnt % 2) else self.agent2
        self.logger.info('VALUE {}, SO {} WINS!'.format(value, winner.name))
        self.agent_scores[winner.name] += 1
        self.stone_scores[str(state.turn)] += 1
      elif value < 0:
        winner = self.agent2 if state.turn == (episode_cnt % 2) else self.agent1
        self.logger.info('VALUE {}, SO OPPONENT {} WINS!'.format(value, winner.name))
        self.agent_scores[winner.name] += 1
        self.stone_scores[str(1 - state.turn)] += 1
      else:
        self.logger.info('VALUE {}, SO DRAW...'.format(value))
        self.agent_scores['draw'] += 1
        self.stone_scores['draw'] += 1
      t_range.set_description('Episode {}'.format(episode_cnt))
      t_range.set_postfix(status=str(self.agent_scores))
    # return results
    return (self.agent_scores, self.stone_scores)

