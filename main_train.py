import numpy as np
import tensorflow as tf
import random
from keras.utils import plot_model # to plot model
from shutil import copyfile # to archive config.py file
import pickle # to archive Memory instance
from importlib import reload
import argparse # to handle training checkpoints.
import itertools
import os

import setting

# if setting.run_folder not exists, create it. 
if not os.path.exists(setting.run_folder):
  os.makedirs(setting.run_folder + 'logs/')
  os.makedirs(setting.run_folder + 'memory/')
  os.makedirs(setting.run_folder + 'models/')

from game import GameEnv
from agent import Agent, User
from memory import Memory
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

def main(initial_run_number, initial_memory_version, initial_model_version):
  '''
  Main function. 
  For training-consistence, load archived versions of memory and model 
  to train from the versions. 

  Parameters
    initial_run_number: nullable int, initial archive run number
    initial_memory_version: nullable int, initial memory version
    initial_model_version: nullable int, initial model version
  '''
  logger.main_logger.info('******************************************')
  logger.main_logger.info('************    NEW LOG    ***************')
  logger.main_logger.info('******************************************')
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
  
  # Initialize Memory and Neural Networks. 
  memory = Memory(config.MEMORY_SIZE)
  best_NN_version = 0
  best_NN = ResidualCNNManager(reg_const=config.REG_CONST, 
                               learning_rate=config.LEARNING_RATE, 
                               input_dim=observation_spec['shape'], 
                               output_dim=action_size,
                               hidden_layers=config.HIDDEN_CNN_LAYERS)
  current_NN = ResidualCNNManager(reg_const=config.REG_CONST, 
                               learning_rate=config.LEARNING_RATE, 
                               input_dim=observation_spec['shape'], 
                               output_dim=action_size,
                               hidden_layers=config.HIDDEN_CNN_LAYERS)

  # 0. if run number is not null, copy the archived config file to root. 
  if initial_run_number != None:
    print('LOAD config.py FROM run{}'.format(str(initial_run_number).zfill(6)))
    archive_path = setting.run_archive_folder + env.name + '/run' + str(initial_run_number).zfill(6)
    copyfile(archive_path + '/config.py', './config.py')  
    
    # 1. if run number and memory version is not null, 
    #    load the given version of memory from the archived run directory. 
    if initial_memory_version != None:
      print('LOAD MEMORY VERSION {}'.format(str(initial_memory_version).zfill(6)))
      memory_path = '/memory/memory' + str(initial_memory_version).zfill(6) + '.pickle'
      memory = pickle.load( open(archive_path + memory_path, 'rb'))
    
    # 2. if run number and model version is not null,
    #    load the given version of model from the archived run directory. 
    if initial_model_version != None:
      print('LOAD MODEL VERSION {}'.format(str(initial_model_version).zfill(6)))
      best_NN_version = initial_model_version
      loaded_model = best_NN.load(env.name, initial_run_number, initial_model_version)
      current_NN.model.set_weights(loaded_model.get_weights())
      best_NN.model.set_weights(loaded_model.get_weights())
    else:
      best_NN.model.set_weights(current_NN.model.get_weights())

  # archive current config.py to run_folder
  copyfile('./config.py', setting.run_folder + 'config.py')
  
  # plot current neural network model to run_folder
  current_NN.plot_model(setting.run_folder + 'models/model.png')

  # Define two agents. (for best_NN and current_NN)
  current_agent = Agent(name='current_agent', 
                        state_size=observation_size, 
                        action_size=action_size, 
                        mcts_simulations=config.MCTS_SIMS, 
                        cpuct=config.CPUCT, 
                        model=current_NN)
  best_agent = Agent(name='best_agent', 
                     state_size=observation_size, 
                     action_size=action_size, 
                     mcts_simulations=config.MCTS_SIMS, 
                     cpuct=config.CPUCT, 
                     model=best_NN)
  
  # Set Tournament instances for (1) Self-Play and (2) Tournament
  self_play = Tournament(env=env, 
                         agent1=best_agent, 
                         agent2=best_agent, 
                         episodes=config.EPISODES, 
                         logger=logger.main_logger, 
                         turns_until_tau0=config.TURNS_UNTIL_TAU0, 
                         memory=memory)
  tournament = Tournament(env=env, 
                          agent1=best_agent, 
                          agent2=current_agent, 
                          episodes=config.EVAL_EPISODES, 
                          logger=logger.tournament_logger, 
                          turns_until_tau0=0, 
                          memory=None)

  # start main loop
  t_iter = 0
  while True:
    '''
    main loop for generating better agents.
    0. self-play
    1. retraining the Neural Network
    2. Tournament: choose the best player!
    '''
    reload(logger)
    reload(config)

    # log current status
    logger.main_logger.info('BEST PLAYER VERSION: {}'.format(str(best_NN_version).zfill(6)))
    print('\nITERATION {}'.format(t_iter))
    print('BEST PLAYER VERSION: {}'.format(str(best_NN_version).zfill(6)))

    # 0. Self-play
    print('0. SELF PLAYING {} EPISODES'.format(self_play.episodes))
    self_play.reset()
    _, _ = self_play.run_tournament()

    # if memory is full, retrain the neural network
    print('1. RETRAINING: MEMORY SIZE IS {} / {}'.format(len(memory.long_term_memory), memory.memory_size))
    if len(memory.long_term_memory) == memory.memory_size:
      # backup memory every 5 interations
      if t_iter % 5 == 0:
        memory_path = 'memory/memory' + str(t_iter).zfill(6) + '.pickle'
        print('1.5. DUMP MEMORY AT {}'.format(setting.run_folder + memory_path))
        pickle.dump(memory, open(setting.run_folder + memory_path, 'wb'))
      
      # 1. Retraining current_NN
      print('DO RETRAINING')
      current_agent.replay(long_term_memory=memory.long_term_memory, 
                           training_loops=config.TRAINING_LOOPS, 
                           epochs=config.EPOCHS, 
                           batch_size=config.BATCH_SIZE)
      
      # log memory
      logger.memory_logger.info('*****************************')
      logger.memory_logger.info('**** NEW MEMORIES (100) ****')
      logger.memory_logger.info('*****************************')      
      #memory_sample = random.sample(memory.long_term_memory, min(100, len(memory.long_term_memory)))
      memory_sample = list(itertools.islice(memory.long_term_memory, 0, min(100, len(memory.long_term_memory))))
      for s in memory_sample:
        current_value, current_policy = current_agent.get_preds(s['state'])
        best_value, best_policy = best_agent.get_preds(s['state'])
        logger.memory_logger.info('MCTS VALUE FOR         {}: {}'.format(s['turn'], s['value']))
        logger.memory_logger.info('CURRENT PRED VALUE FOR {}: {}'.format(s['turn'], current_value))
        logger.memory_logger.info('BEST PRED VALUE FOR    {}: {}'.format(s['turn'], best_value))
        logger.memory_logger.info('MCTS ACTION PROBS:         {}'.format(s['action_probs']))
        logger.memory_logger.info('CURRENT PRED ACTION PROBS: {}'.format(current_policy))
        logger.memory_logger.info('BEST PRED ACTION PROBS:    {}'.format(best_policy))
        logger.memory_logger.info('STATE KEY: {}'.format(s['state'].key))
        logger.memory_logger.info('INPUT STATE: ')
        s['state'].render('log', logger.memory_logger)

      # 2. Tournament: best player vs current player
      print('2. TOURNAMENT {} EPISODES'.format(tournament.episodes))
      tournament.reset()
      agent_scores, stone_scores = tournament.run_tournament()
      print('AGENT SCORES: {}'.format(agent_scores))
      print('STONE SCORES: {}'.format(stone_scores))

      # 3. If current_agent wins a lot, replace best_agent to current_agent
      if agent_scores[current_agent.name] > agent_scores[best_agent.name] * config.SCORING_THRESHOLD:
        best_NN_version += 1
        print('NEW BEST VERSION! VERSION {}'.format(str(best_NN_version).zfill(6)))
        best_NN.model.set_weights(current_NN.model.get_weights())
        best_NN.save(env.name, best_NN_version)
        best_NN.view_layers(env.name, best_NN_version)
    else:
      print('SKIP RETRAINING AND TOURNAMENT')
    t_iter += 1

if __name__ == '__main__':
  # parse arguments
  parser = argparse.ArgumentParser(description='Alpha Zero Clone Trainer')
  parser.add_argument('--run-number', '-R', type=int, 
    help='Number of past run in run_archive directory. {root dir}/run_archive/runxxxxxx should exists. ')
  parser.add_argument('--memory-version', '-M', type=int, 
    help='If run-number is not null, can select memory version to continue with the memory. ')
  parser.add_argument('--model-version', '-D', type=int, 
    help='If run-number is not null, can select model version to continue from the model. ')
  args = parser.parse_args()

  # train
  main(args.run_number, args.memory_version, args.model_version)

