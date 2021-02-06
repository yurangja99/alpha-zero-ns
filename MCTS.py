import numpy as np

import config
import logger

class Node:
  '''
  Indicates each nodes in MCTS class
  '''
  def __init__(self, state):
    '''
    create new Node. 

    Parameters
      state: GameState instance indicating current state

    Fields
      self.key: key used in MCTS class.
      self.state: state
      self.turn: turn
      self.edges: connected Edges to this node.
    '''
    self.state = state
    self.key = self.state.key
    self.turn = self.state.turn
    self.edges = []

  def __len__(self):
    '''
    returns length of edges
    '''
    return len(self.edges)

  def is_leaf(self):
    '''
    Returns whether this node is leaf or not. 
    If there are no edges, this is leaf.
    '''
    return len(self.edges) == 0
  
  def add_edge(self, action, edge):
    '''
    Add new Edge instance to self.edges
    '''
    self.edges.append((action, edge))

class Edge:
  '''
  Indicates relations between Node classes
  '''
  def __init__(self, parent_node, child_node, prior_prob, action):
    '''
    create new edge

    Parameters
      parent_node: Node class. parent node.
      child_node: Node class. child node.
      prior_prob: probability to take the action
      action: int value of action

    Fields
      self.key: key used in Node class
      self.parent_node: parent node
      self.child_node: child node
      self.turn: turn
      self.action: action value of this Edge
      self.stats: statistics for this Edge (or action)
        N: number of visiting this Edge (means state and action)
        W: total value of state-action pair (total value of the next state)
        Q: value of state-action pair (mean value of the next state)
        P: prior probability to do self.action in self.parent_node.state
    '''
    self.key = parent_node.key + '|' + child_node.key
    self.parent_node = parent_node
    self.child_node = child_node
    self.turn = self.parent_node.turn
    self.action = action
    self.stats = {'N': 0, 'W': 0, 'Q': 0, 'P': prior_prob}

class MCTS:
  '''
  Monte Carlo Tree Search

  There are 3 conditions to apply MCTS
  1. There is max/min score of the game.
  2. The rule exists and observation = state.
  3. The game is episodic and quite short.
  
  There are 4 MCTS processes.
  1. Selection: start from the root node, select the best leaf (exposed node)
  2. Expansion: if game's not done, generate one or more child nodes and select one.
  3. Simulation: from selected one, simulate the game until the game ends.
  4. Backpropagation: update the value of all nodes in the path using the result of the simulation.
  '''
  def __init__(self, root, cpuct):
    '''
    create new MCTS class

    Parameters
      root: Node class of current state
      cpuct: a constant indicates exploration vs exploitation tradeoff. \
        it is used when calculating U for select child in Selection step.
    
    Fields
      self.root: root node including current state
      self.tree: dictionary of Node classes in this tree.
      self.cpuct: cpuct value to control exploration vs exploitation
    '''
    self.root = root
    self.tree = {}
    self.cpuct = cpuct
    # add root node
    self.add_node(root)

  def __len__(self):
    '''
    return number of nodes in this MCTS instance. 
    '''
    return len(self.tree)
  
  def add_node(self, node):
    '''
    add node to self.tree
    '''
    self.tree[node.key] = node

  def move_to_leaf(self):
    '''
    Function for Selection step. 
    Start from root node, move to leaf node. 
    Make sure to choose the best path according to Q + U value.

    Return
      time step: (leaf_node, value, done, breadcrumbs)
        breadcrumbs: path from root to leaf_node
    '''
    logger.mcts_logger.info('***** MOVING TO LEAF *****')
    # path from root to leaf, and some variables
    breadcrumbs = []
    # start from root
    current_node = self.root

    done = False
    value = 0

    # start to select the best childs until leaf or episode ends.
    while not current_node.is_leaf() and not done:
      logger.mcts_logger.info('PLAYER TURN...{}'.format(current_node.state.turn))
      # save max Q + U value at this variable
      max_Q_plus_U = -1e8

      # set epsilon and nu according to current_node
      # whether current_node is root or not.
      if current_node == self.root:
        epsilon = config.EPSILON
        nu = np.random.dirichlet([config.ALPHA] * len(current_node))
      else:
        epsilon = 0
        nu = [0] * len(current_node)
      
      # calculate Q + U, especially U
      # first, calculate all visit counts of current node
      N_sum = sum([edge.stats['N'] for action, edge in current_node.edges])

      # find the best child
      for idx, (action, edge) in enumerate(current_node.edges):
        # calculate Q
        Q = edge.stats['Q']
        
        # calculate U (Upper Bound Confidence)
        # if current_node is root, consider randomly initialized nu.
        # otherwise, only consider true edge.stats.
        U = self.cpuct * ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * \
          np.sqrt(N_sum) / (1 + edge.stats['N'])
        
        logger.mcts_logger.info('ACTION: {}... N = {}, P = {}, nu = {}, adjP = {}, W = {}, Q = {}, U = {}, Q + U = {}'
                                .format(action, edge.stats['N'], edge.stats['P'], nu[idx], 
                                (1 - epsilon) * edge.stats['P'] + epsilon * nu[idx], 
                                edge.stats['W'], Q, U, Q + U))

        # check max
        if max_Q_plus_U < Q + U:
          max_Q_plus_U = Q + U
          simulation_action = action
          simulation_edge = edge
      # move to the child (move forward)
      logger.mcts_logger.info('ACTION WITH HIGHEST Q + U...{}'.format(simulation_action))
      _, value, done, _ = current_node.state.step(simulation_action)
      current_node = simulation_edge.child_node
      breadcrumbs.append(simulation_edge)
    # return current node, value at the node, done at the node, and breadcrumbs (path)
    logger.mcts_logger.info('FOUND LEAF NODE...value = {}, done = {}, len_breadcrumbs = {}'
                            .format(value, done, len(breadcrumbs)))
    return (current_node, value, done, breadcrumbs)

  def back_fill(self, leaf, value, breadcrumbs):
    '''
    Function for Backpropagation step. 
    '''
    logger.mcts_logger.info('****** DOING BACKFILL *****')

    # get leaf's turn
    leaf_turn = leaf.state.turn

    # start to update the nodes in paths
    for edge in breadcrumbs:
      # set update direction (+ or -) according to 
      # leaf turn and edge turn
      edge_turn = edge.turn
      direction = 1 if edge_turn == leaf_turn else -1
      
      # update nodes in the path
      edge.stats['N'] += 1
      edge.stats['W'] += value * direction
      edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

      logger.mcts_logger.info('UPDATING EDGE WITH VALUE {} FOR PLAYER {}...N = {}, W = {}, Q = {}'
                              .format(value * direction, edge_turn, edge.stats['N'], edge.stats['W'], edge.stats['Q']))
      edge.child_node.state.render('log', logger.mcts_logger)