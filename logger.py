import logging

import setting

def setup_logger(name, log_file, level=logging.INFO):
  '''
  Set up new logger. 

  Parameters
    name: logger's name
    log_file: file's name where the log file will be stored. 
    level: set the logger's log level.
  '''
  formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
  handler = logging.FileHandler(log_file)
  handler.setFormatter(formatter)

  logger = logging.getLogger(name)
  logger.setLevel(level)
  if not logger.handlers:
    logger.addHandler(handler)
  return logger

# set 'disabled' option for each loggers. 
LOGGER_DISABLED = {
  'main': False, 
  'memory': False, 
  'tournament': False, 
  'mcts': True, 
  'model': False, 
  'league': False
}

# main logger
main_logger = setup_logger('main_logger', setting.run_folder + 'logs/main_logger.log')
main_logger.disabled = LOGGER_DISABLED['main']

# memory logger
memory_logger = setup_logger('memory_logger', setting.run_folder + 'logs/memory_logger.log')
memory_logger.disabled = LOGGER_DISABLED['memory']

# tournament logger
tournament_logger = setup_logger('tournament_logger', setting.run_folder + 'logs/tournament_logger.log')
tournament_logger.disabled = LOGGER_DISABLED['tournament']

# mcts logger
mcts_logger = setup_logger('mcts_logger', setting.run_folder + 'logs/mcts_logger.log')
mcts_logger.disabled = LOGGER_DISABLED['mcts']

# model logger
model_logger = setup_logger('model_logger', setting.run_folder + 'logs/model_logger.log')
model_logger.disabled = LOGGER_DISABLED['model']

# league logger
league_logger = setup_logger('league_logger', setting.run_folder + 'logs/league_logger.log')
model_logger.disabled = LOGGER_DISABLED['league']