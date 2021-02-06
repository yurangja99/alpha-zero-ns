import numpy as np
import tensorflow as tf
from keras.utils import plot_model # to plot model

from MCTS import MCTS, Node, Edge
from game_tw import GameEnv
from model import ResidualCNNManager
from agent import User, Agent
from tournament import Tournament
from memory import Memory
import logger
import config

# Limit GPU to occupy less memory for tensorflow. 
# (if there's no external gpu, it does nothing)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
  except Exception:
    pass

def play_game_2_users():
  '''
  Test game.py through 2 user playing.
  '''
  # initialize game
  env = GameEnv()
  _, _, done, _ = env.reset()
  
  # start game
  while not done:
    env.state().render('human')
    if env.turn() == 0:
      action = int(input(">> green turn: "))
    else:
      action = int(input(">> red turn: "))
    state, value, done, _ = env.step(action)
    print('(turn_state, turn_env, value, done) = ({}, {}, {}, {})'
          .format(state.turn, env.turn(), value, done))

def mcts_init():
  '''
  Init an MCTS, and add a Node and an Edge instance. 
  Next, print some Node and Edge list. 
  '''
  # initialize env
  env = GameEnv()
  # add some states to MCTS
  state0, _, _, _ = env.reset()
  state1, _, _, _ = env.step(5)
  state2, _, _, _ = env.step(6)
  # initialize Nodes
  node0 = Node(state0)
  node1 = Node(state1)
  node2 = Node(state2)
  # initialize MCTS
  mcts = MCTS(node0, 1)
  mcts.add_node(node1)
  mcts.add_node(node2)
  # initialize Edges
  edge01 = Edge(node0, node1, 1.0, 5)
  edge12 = Edge(node1, node2, 1.0, 6)
  node0.add_edge(5, edge01)
  node1.add_edge(6, edge12)
  # print result
  print('MCTS root: {}'.format(mcts.root.key))
  print('MCTS nodes: {}'.format(len(mcts.tree)))
  for node_key, node in mcts.tree.items():
    print('Node {}'.format(node_key))
    print('\tturn: {}'.format(node.turn))
    for action, edge in node.edges:
      print('\tEdge {}'.format(edge.key))
      print('\t\tparent_node_key: {}'.format(edge.parent_node.key))
      print('\t\tchild_node_key: {}'.format(edge.child_node.key))
      print('\t\tturn: {}'.format(edge.turn))
      print('\t\taction: {} == {}'.format(action, edge.action))
      print('\t\tstats: {}'.format(str(edge.stats)))
  # return MCTS instance
  return mcts

def mcts_move_to_leaf(mcts):
  '''
  Init an MCTS, and some Node and Edge Instances. 
  Next, set some variables for the nodes and edgse. 
  Check the result leaf is intended leaf.

  Parameters
    mcts: MCTS instance
  '''
  leaf_node, value, done, breadcrumbs = mcts.move_to_leaf()
  print('Leaf Node {}'.format(leaf_node.key))
  print('\tturn: {}'.format(leaf_node.turn))
  for action, edge in leaf_node.edges:
    print('\tEdge {}'.format(edge.key))
    print('\t\tparent_node_key: {}'.format(edge.parent_node.key))
    print('\t\tchild_node_key: {}'.format(edge.child_node.key))
    print('\t\tturn: {}'.format(edge.turn))
    print('\t\taction: {} == {}'.format(action, edge.action))
    print('\t\tstats: {}'.format(str(edge.stats)))
  print('Value: {}'.format(value))
  print('Done: {}'.format(done))
  print('Breadcrumbs: {}'.format(len(breadcrumbs)))
  for edge in breadcrumbs:
    print('\tEdge {}'.format(edge.key))
    print('\t\tparent_node_key: {}'.format(edge.parent_node.key))
    print('\t\tchild_node_key: {}'.format(edge.child_node.key))
    print('\t\tturn: {}'.format(edge.turn))
    print('\t\taction: {}'.format(edge.action))
    print('\t\tstats: {}'.format(str(edge.stats)))
  return (leaf_node, value, done, breadcrumbs)

def mcts_back_fill(mcts, leaf_node, value, breadcrumbs):
  '''
  Init an MCTS, and some Nodes and Edge Instances.
  Next, set value and breadcrumbs. 
  After Backfill process, check the result is intended.

  Parameters
    mcts: MCTS instance
    leaf_node: MCTS.Node instance
    value: value by Neural Network
    breadcrumbs: list of MCTS.Edge
  '''
  # back fill
  mcts.back_fill(leaf_node, value, breadcrumbs)
  # print result
  print('MCTS root: {}'.format(mcts.root.key))
  print('MCTS nodes: {}'.format(len(mcts.tree)))
  for node_key, node in mcts.tree.items():
    print('Node {}'.format(node_key))
    print('\tturn: {}'.format(node.turn))
    for action, edge in node.edges:
      print('\tEdge {}'.format(edge.key))
      print('\t\tparent_node_key: {}'.format(edge.parent_node.key))
      print('\t\tchild_node_key: {}'.format(edge.child_node.key))
      print('\t\tturn: {}'.format(edge.turn))
      print('\t\taction: {} == {}'.format(action, edge.action))
      print('\t\tstats: {}'.format(str(edge.stats)))

def model_init():
  '''
  Test residual cnn model initialization. 
  Next, put a state and get a result. 
  '''
  # init state
  env = GameEnv()
  observation_spec = env.observation_spec()
  action_spec = env.action_spec()
  state0, _, _, _ = env.reset()
  state1, _, _, _ = env.step(6)
  # init model
  model = ResidualCNNManager(config.REG_CONST, 
                             config.LEARNING_RATE, 
                             observation_spec['shape'], 
                             np.prod(action_spec['shape']) * (action_spec['max'] - action_spec['min'] + 1), 
                             config.HIDDEN_CNN_LAYERS)
  model.summary()
  model.print_weight_averages()
  model.view_layers(game=env.name, version=123456)
  model.plot_model(to_file='model.png')
  # put state into model and print the result
  for state in [state0, state1]:
    value_head, policy_head = model.predict(np.array([state.state]))
    print('value head: {}'.format(value_head))
    print('policy head: {}'.format(policy_head))

def agent_init():
  '''
  For Connected Four game, define agent and play game. 
  This does not test training, just testing Agent's basic acting. 
  
  Listed functions of Agent class will be tested. 
  - __init__
  - act
  - simulate
  - evaluate_leaf
  - get_preds
  - get_action_values
  - choose_action
  - set_MCTS

  Listed functions will not be tested. 
  - replay
  '''
  # init env
  env = GameEnv()
  observation_spec = env.observation_spec()
  action_spec = env.action_spec()
  observation_size = observation_spec['shape']
  action_size = np.prod(action_spec['shape']) * \
                (action_spec['max'] - action_spec['min'] + 1)
  # init User agent. 
  user_agent = User('user', observation_size, action_size)
  # init Agent agent with Residual CNN. 
  model = ResidualCNNManager(config.REG_CONST, 
                             config.LEARNING_RATE, 
                             observation_size, 
                             action_size, 
                             config.HIDDEN_CNN_LAYERS)
  agent_agent = Agent('agent', observation_size, action_size, 
                      config.MCTS_SIMS, config.CPUCT, model)
  # play one episode. 
  state, _, done, _ = env.reset()
  tau = 0
  while not done:
    state.render('human')
    input(">> next turn {} ".format(state.turn))
    if state.turn == 3:
      action, pi, value, NN_value = user_agent.act(state, tau)
      print('User\'s result: ({}, {}, {}, {})'.format(action, pi, value, NN_value))
    else:
      action, pi, value, NN_value = agent_agent.act(state, tau)
      print('Agent\'s result: ({}, {}, {}, {})'.format(action, pi, value, NN_value))
    state, value, done, _ = env.step(action)
    print('result: value={}, done={}'.format(value, done))
    
  # episode ended. 
  print("End with reward {} for {}".format(value, env.turn()))
  # after the game, check MCTS of the Agent agent. 
  mcts = agent_agent.mcts
  print('MCTS root: {}'.format(mcts.root.key))
  print('MCTS nodes: {}'.format(len(mcts.tree)))
  print('MCTS edges: {}'.format(np.sum([len(node.edges) for _, node in mcts.tree.items()])))

def tournament_init():
  '''
  Test Tournament class. 
  In this functino, we use two User agents 
  to see statistics are correct. 
  '''
  # init env
  env = GameEnv()
  observation_spec = env.observation_spec()
  action_spec = env.action_spec()
  observation_size = observation_spec['shape']
  action_size = np.prod(action_spec['shape']) * \
                (action_spec['max'] - action_spec['min'] + 1)
  # init two User agents.
  user1 = User('user_one', observation_size, action_size)
  user2 = User('user_two', observation_size, action_size)
  # init memory
  memory = Memory(1000)
  # init tournament
  tournament = Tournament(env, user1, user2, 3, logger.tournament_logger, 1, memory)
  # run tournament, and check statistics and memory
  agent_scores, stone_scores = tournament.run_tournament()
  print('agent scores:', agent_scores)
  print('stone scores:', stone_scores)
  print('memory length: short {} / long {} / total {}'
        .format(len(memory.short_term_memory), len(memory.long_term_memory), memory.memory_size))

if __name__ == '__main__':
  # start test
  #play_game_2_users()
  #mcts = mcts_init()
  #leaf_node, _, _, breadcrumbs = mcts_move_to_leaf(mcts)
  #mcts_back_fill(mcts, leaf_node, -1, breadcrumbs)
  #model_init()
  #agent_init()
  #tournament_init()
  pass
