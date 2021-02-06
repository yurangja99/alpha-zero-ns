import numpy as np
import random
import matplotlib.pyplot as plt

import MCTS
import logger
import setting

class User:
  '''
  An agent that uses user's input as its action. 
  This Agent didn't have any neural network stuffs. 
  '''
  def __init__(self, name, state_size, action_size):
    '''
    create new User instance. 

    Parameters
      name: str. name of the agent
      state_size: int. state size
      action_size: int. action size

    Fields
      self.name: name
      self.state_size: state size
      self.action_size: action size
    '''
    self.name = name
    self.state_size = state_size
    self.action_size = action_size
  
  def act(self, state, tau):
    '''
    Given state, the agent choose action. 
    In this instance, user input is needed.

    Parameters
      state: Game instance. (not used)
      tau: 0 or 1. temperature variable (not used)
    
    Return
      return: tuple (action, pi, value, NN_value)
        action: int. selected action
        pi: np array. action probabilities
        value: value of selected action by MCTS (not used)
        NN_value: value of selected action by model (not used)
    '''
    action = int(input('>> {}: Enter your chosen action: '.format(self.name)))
    pi = np.zeros(self.action_size)
    pi[action] = 1
    return (action, pi, None, None)

  def initialize_mcts(self):
    '''
    (Not Used) Initialize MCTS data. 
    '''
    pass

class Agent:
  '''
  An agent with MCTS and Neural Networks. 
  '''
  def __init__(self, name, state_size, action_size, mcts_simulations, cpuct, model):
    '''
    Create new Agent agent. 

    Parameters
      name: str. name of the agent
      state_size: int. state size
      action_size: int. action size
      mcts_simulations: int. number of simulation episodes when using MCTS
      cpuct: float. affects U's ratio to Q.
      model: ModelManager. Neural network model
    
    Fields
      self.name: name
      self.state_size: state size
      self.action_size: action size
      self.mcts_simulations: number of simulation episodes
      self.cpuct: cpuct * U + Q
      self.model: ModelManager
      self.mcts: MCTS instance of the agent
    '''
    self.name = name
    self.state_size = state_size
    self.action_size = action_size
    self.mcts_simulations = mcts_simulations
    self.cpuct = cpuct
    self.model = model

    self.mcts = None

    self.train_overall_loss = []
    self.train_value_loss = []
    self.train_policy_loss = []

  def initialize_mcts(self):
    '''
    Initialize MCTS data. 
    '''
    self.mcts = None

  def act(self, state, tau):
    '''
    Given state, the agent choose action. 
    In this instance, MCTS and Neural network used. 

    Parameters
      state: Game instance. 
      tau: 0 or 1. temperature variable. 0 for greedy, 1 for stochastic. 
    
    Return
      return: tuple (action, pi, value, NN_value)
        action: int. selected action
        pi: np array. action probabilities
        value: value of selected action by MCTS 
        NN_value: value of selected action by model 
    '''
    # set self.mcts first
    self.__set_MCTS(state)

    # run some simulations
    for i in range(self.mcts_simulations):
      logger.mcts_logger.info('***************************')
      logger.mcts_logger.info('****** SIMULATION {} ******'.format(i))
      logger.mcts_logger.info('***************************')
      self.__simulate()

    # after MCTS adjusting, get action values from the root state
    pi, values = self.__get_action_values()

    # select one action and return it.
    action, value = self.__choose_action(pi, values, tau)
    next_state, _, _, _ = state.step(action)
    value_head, _ = self.get_preds(next_state)

    logger.mcts_logger.info('ACTION VALUES...{}'.format(pi))
    logger.mcts_logger.info('CHOSEN ACTION...{}'.format(action))
    logger.mcts_logger.info('MCTS PERCEIVED VALUE...{}'.format(value))
    logger.mcts_logger.info('NN PERCEIVED VALUE...{}'.format(-value_head[0]))

    return (action, pi, value, -value_head[0])

  def __set_MCTS(self, state):
    '''
    Create a new MCTS instance for this agent, or change its root to given state. 

    Parameters
      state: GameState instance. this will be the root of the MCTS instance. 
    '''
    if self.mcts == None or state.key not in self.mcts.tree.keys():
      # if self.mcts is not set yet or given state is not in self.mcts, 
      # create new MCTS and set the state as a root node. 
      logger.mcts_logger.info('****** BUILDING NEW MCTS TREE FOR AGENT {} ******'
                              .format(self.name))
      self.mcts = MCTS.MCTS(MCTS.Node(state), self.cpuct)
    else:
      # if self.mcts exists and given state is in self.mcts, 
      # move the root of the MCTS to the given state. 
      logger.mcts_logger.info('****** CHANGING ROOT OF MCTS TREE TO {} FOR AGENT {} ******'
                              .format(state.key, self.name))
      self.mcts.root = self.mcts.tree[state.key]

  def __simulate(self):
    '''
    simulate one episode by Monte-Carlo method.
    0. it moves to the leaf of MCTS instance. 
    1. it uses neural network to evaluate the leaf node. 
    2. it backfills the evaluated value to MCTS breadcrumbs (path)

    Return
      return: (leaf, value, breadcrumbs)
        leaf: MSTC.Node. selected leaf node instance 
        value: float. reward of the transition to the leaf. 
        breadcrumbs: list of MSTC.Edge. path from root to the leaf. 
    '''
    # log simulation state
    logger.mcts_logger.info('ROOT NODE...{}'.format(self.mcts.root.key))
    self.mcts.root.state.render('log', logger.mcts_logger)
    logger.mcts_logger.info('CURRENT TURN...{}'.format(self.mcts.root.turn))

    # move to the leaf node
    leaf_node, value, done, breadcrumbs = self.mcts.move_to_leaf()

    # evaluate the leaf node
    value, breadcrumbs = self.__evaluate_leaf(leaf_node, value, done, breadcrumbs)

    # backfill the value through the path
    self.mcts.back_fill(leaf_node, value, breadcrumbs)

  def __evaluate_leaf(self, leaf_node, value, done, breadcrumbs):
    '''
    Evaluate the leaf node using neural network. 
    
    Using the results from it, this function does two things. 
    1. get policy probabilities, then add new Nodes and Edges below the leaf node. 
    2. get action value, and return it for backfill.
    
    Parameters
      leaf_node: MSTC.Node. leaf node to evaluate
      value: float. reward of the transition to the leaf. 
      done: boolean. whether the leaf node is terminal or not. 
      breadcrumbs: list of MSTC.Edge. path from root to the leaf node. 
    
    Return
      return: tuple of (float, list of MSTC.Edge). (value, breadcrumbs)
        value: if leaf_node is terminal, return it without any changes. otherwise, return the final value. 
        breadcrumbs: return the input breadcrumbs without any changes. 
    '''
    logger.mcts_logger.info('***** EVALUATING LEAF ******')
    if not done:
      # get evaluation from neural network.
      value, action_probs = self.get_preds(leaf_node.state)
      logger.mcts_logger.info('PREDICTED VALUE BY NN FOR TURN {}: {}'.format(leaf_node.state.turn, value))
      # add bew Nodes and Edges to the leaf node
      for action in range(len(action_probs)):
        new_state, _, _, _ = leaf_node.state.step(action)
        if new_state.key not in self.mcts.tree:
          node = MCTS.Node(new_state)
          self.mcts.add_node(node)
          logger.mcts_logger.info('ADDED NODE...{}...p = {}'.format(node.key, action_probs[action]))
        else:
          node = self.mcts.tree[new_state.key]
          logger.mcts_logger.info('EXISTING NODE...{}...'.format(node.key))
        new_edge = MCTS.Edge(leaf_node, node, action_probs[action], action)
        leaf_node.add_edge(action, new_edge)
    else:
      logger.mcts_logger.info('GAME VALUE FOR {}: {}'.format(leaf_node.turn, value))
    return (value, breadcrumbs)

  def get_preds(self, state):
    '''
    Put the state to the model (neural network) and return the result. 

    Parameters
      state: GameState instance. state to get predictions
    
    Return
      return: tuple of (float, np.array). (value, action probabilities)
    '''
    value_head, policy_head = self.model.predict(np.array([state.model_input]))
    exp_probs = np.exp(policy_head[0])
    return (value_head[0], exp_probs / np.sum(exp_probs))

  def __get_action_values(self):
    '''
    Get action values using MCTS instance. 
    Through some simulations, MCTS backfill was processed. 
    So, MCTS will better perform. 

    Return
      return: tuple of (np.array, np.array). (pi, values)
        pi: according to edge.stats['N'], select best Edge. 
        values: according to edge.stats['Q'], get Q values
    '''
    pi = np.zeros(self.action_size, dtype=np.float32)
    values = np.zeros(self.action_size, dtype=np.float32)

    for action, edge in self.mcts.root.edges:
      pi[action] = float(edge.stats['N'])
      values[action] = edge.stats['Q']
    
    return (pi / np.sum(pi), values)

  def __choose_action(self, pi, values, tau):
    '''
    Choose action according to pi and Q values from self.__get_action_values. 

    Parameters
      pi: np.array. action probabilities 
      values: np.array. Q values for each action 
      tau: 0 or 1. temperature parameter
    
    Return
      return: tuple of (int, float). (selected action, value for the action)
    '''
    if tau == 0:
      actions = np.argwhere(pi == np.max(pi))
      action = random.choice(actions)[0]
    else:
      action = np.random.choice(self.action_size, p=pi)
    value = values[action]
    return (action, value)

  def replay(self, long_term_memory, training_loops, epochs, batch_size):
    '''
    Using the given long-term memory, retrain the agent's model. (Neural Network)

    Parameters
      long_term_memory: deque. Memory instance's long-term memory
      training_loops: int. training loops. (number of samplings)
      epochs: int. batch size
      batch_size: int. batch size
    '''
    logger.mcts_logger.info('***** RETRAINING MODEL *****')
    for _ in range(training_loops):
      # get random mini batch
      minibatch = random.sample(long_term_memory, min(batch_size, len(long_term_memory)))
      # from mini batch, get training datasets
      training_states = np.array([item['state'].model_input for item in minibatch])
      training_targets = {
        'value_head': np.array([item['value'] for item in minibatch]), 
        'policy_head': np.array([item['action_probs'] for item in minibatch])
      }
      # fit model
      fit = self.model.fit(training_states, training_targets, 
                           epochs=epochs, verbose=1, callbacks=[], 
                           validation_split=0, 
                           batch_size=batch_size)
      logger.mcts_logger.info('NEW LOSS {}'.format(fit.history))

      self.train_overall_loss.append(fit.history['loss'][epochs - 1])
      self.train_value_loss.append(fit.history['value_head_loss'][epochs - 1]) 
      self.train_policy_loss.append(fit.history['policy_head_loss'][epochs - 1]) 
    
    plt.figure(figsize=(6.94, 5.2))
    '''
    plt.subplot(1, 2, 1)
    plt.plot(self.train_overall_loss, 'k')
    plt.title('Agent {}\'s Overall Loss'.format(self.name))
    plt.legend(['train_overall_loss'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(self.train_value_loss, 'r:')
    plt.plot(self.train_policy_loss, 'b--')
    plt.title('Agent {}\'s Value / Policy Loss'.format(self.name))
    plt.legend(['train_value_loss', 'train_policy_loss'], loc='upper right')
    '''
    plt.plot(self.train_overall_loss, 'k')
    plt.plot(self.train_value_loss, 'r:')
    plt.plot(self.train_policy_loss, 'b--')
    plt.title('Agent {}\'s Overall/Value/Policy Loss'.format(self.name))
    plt.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='upper right')
    plt.grid(True)
    plt.savefig(setting.run_folder + 'learning_curve.png')
    plt.close()

    self.model.print_weight_averages()