
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import os

import setting

# if setting.run_folder not exists, create it. 
if not os.path.exists(setting.run_folder):
  os.makedirs(setting.run_folder + 'logs/')
  os.makedirs(setting.run_folder + 'memory/')
  os.makedirs(setting.run_folder + 'models/')

from game import GameEnv
from agent import Agent, User
from model import ResidualCNNManager
from tournament import Tournament
import config
import logger

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

def main(include_user, render_option, plot_result):
  '''
  Play league for specific run number. 
  User can see model list, and select model numbers to participate. 
  The result will be plotted, and returned by numpy array. 

  Parameters
    include_user: bool. whether user participate in league or not. 
    render_option: str. rendering option: 'human', 'terminal', and None. 
    plot_result: bool. whether this function saves the figure of the result, or not. 
  '''
  logger.league_logger.info('******************************************')
  logger.league_logger.info('************    NEW LOG    ***************')
  logger.league_logger.info('******************************************')
  print('******************************************')
  print('************    NEW LOG    ***************')
  print('******************************************')
  
  # Initialize env
  env = GameEnv()
  observation_spec = env.observation_spec()
  action_spec = env.action_spec()
  observation_size = np.prod(observation_spec['shape'])
  action_size = np.prod(action_spec['shape']) * \
                (action_spec['max'] - action_spec['min'] + 1)
  
  # get run number. 
  run_number = input('\n>> please set run number: ')

  # get list of models. 
  archived_model_list = list(filter(lambda x: x.endswith('.h5'), 
    os.listdir(setting.run_archive_folder + env.name + '/run' + run_number.zfill(6) + '/models')))
  model_versions = input('\nModels in run number {}: \n{}\n>> please select models (ex. 0 5 10 15): '
                        .format(run_number, '\n'.join(archived_model_list))).split()
  
  # set agents. 
  agents = []
  NN = ResidualCNNManager(reg_const=config.REG_CONST, 
                          learning_rate=config.LEARNING_RATE, 
                          input_dim=observation_spec['shape'], 
                          output_dim=action_size,
                          hidden_layers=config.HIDDEN_CNN_LAYERS)
  for model_version in model_versions:
    # handle exception (the version by input doesn't exist, for example. )
    model_name = '{}_version{}.h5'.format(env.name, model_version.zfill(6))
    if model_name in archived_model_list:
      agents.append(Agent(name=model_version.zfill(6), 
                          state_size=observation_size, 
                          action_size=action_size, 
                          mcts_simulations=config.MCTS_SIMS, 
                          cpuct=config.CPUCT, 
                          model=NN.load(env.name, run_number, model_version)))
    else:
      print('{} doesn\'t exists'.format(model_name))
  
  # if include_user flag is enabled, add User at last. 
  if include_user:
    agents.insert(0, User(name='you', 
                          state_size=observation_size, 
                          action_size=action_size))
  print('\nParticipating agents: {}\n{}'.format(len(agents), '\n'.join([a.name for a in agents])))
  
  # create Tournament instance. 
  if len(agents) < 2:
    print('\nYou should choose more or equal than 2 agents including yourself.')
    return
  league = Tournament(env=env, 
                      agent1=agents[0], 
                      agent2=agents[1], 
                      episodes=2, 
                      logger=logger.league_logger, 
                      turns_until_tau0=0, 
                      memory=None, 
                      render_option=render_option)

  # start playing league
  total_agent_scores = np.zeros((len(agents), len(agents)))
  total_stone_scores = { '0': 0, 'draw': 0, '1': 0 }
  for i in range(len(agents) - 1):
    for j in range(i + 1, len(agents)):
      print('\nLeague Match of {} and {}'.format(agents[i].name, agents[j].name))
      league.reset(agent1=agents[i], agent2=agents[j])
      agent_scores, stone_scores = league.run_tournament()
      total_agent_scores[i, j] = agent_scores[agents[i].name] - agent_scores[agents[j].name]
      total_agent_scores[j, i] = agent_scores[agents[j].name] - agent_scores[agents[i].name]
      total_stone_scores['0'] += stone_scores['0']
      total_stone_scores['draw'] += stone_scores['draw']
      total_stone_scores['1'] += stone_scores['1']
      print('AGENT SCORES: {}'.format(agent_scores))
      print('STONE SCORES: {}'.format(stone_scores))
  
  y0 = total_agent_scores.tolist()[:]
  for i in range(len(y0)):
    y0[i][i] = '-'
  x1 = [a.name for a in agents]
  y1 = np.sum(total_agent_scores, axis=-1).tolist()
  x2 = ['black', 'draw', 'white']
  y2 = [total_stone_scores['0'], total_stone_scores['draw'], total_stone_scores['1']]

  # print result
  print('\nLeague Results: \n{}'.format(y0))
  print('League Points: {}'.format(y1))
  print('Black / White Win Rates: {}'.format(y2))

  # plot result
  if plot_result:
    plt.figure(figsize=(12.8, 9.6))

    plt.subplot2grid((2, 7), (0, 0), colspan=7)
    plt.table(cellText=y0, colLabels=x1, rowLabels=x1, loc='center', bbox=[0.0, 0.0, 1.0, 1.0]) # left, bottom, width, height
    plt.title('League Results')
    plt.axis('off')
    
    plt.subplot2grid((2, 7), (1, 0), colspan=5)
    plt.plot(x1, y1, 'o-', label='score')
    plt.title('League Points')
    plt.xlabel('Version')
    plt.ylabel('Score')
    plt.xticks(rotation=50)
    plt.grid(True)
    plt.legend()
    
    plt.subplot2grid((2, 7), (1, 5), colspan=2)
    plt.pie(y2, labels=x2, autopct='%.2f%%')
    plt.title('Black / White Win Rates')
    
    plt.savefig(setting.run_folder + 'league.png')
    plt.close()

if __name__ == '__main__':
  # parse arguments
  parser = argparse.ArgumentParser(description='Alpha Zero Clone League')
  parser.add_argument('--include-user', '-IU', action='store_true', 
    help='If enabled, include User (YOU) in the league. ')
  parser.add_argument('--render-option', '-R', type=str, default='terminal', 
    help='Set render option of the league. \'human\' for plotting, \
    \'terminal\' for terminal printing, and else for nothing. \
    Default setting is \'terminal\'. ')
  parser.add_argument('--plot-result', '-P', action='store_true', 
    help='If enabled, save plotted img in run folder. ')
  args = parser.parse_args()

  # play league
  main(args.include_user, args.render_option, args.plot_result)