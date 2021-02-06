# Self Play
EPISODES = 30 #75 # used in main.py. set number of self-play episodes of Tournament
MCTS_SIMS = 50 #100 # used in main.py. set number of MCTS simulation of Agent
MEMORY_SIZE = 30000 #90000 # used in main.py. set capacity of the memory
TURNS_UNTIL_TAU0 = 10 # used in main.py set Agent's threshold to act greedily
CPUCT = 1 # used in main.py. set cpuct of Agent. affects ratio of Q and U in MCTS.move_to_leaf().
EPSILON = 0.2 # used in MCTS, set epsilon if current node is root. (epsilon affects U) maybe exploration?
ALPHA = 0.8 # used in MCTS, set nu distribution if current node is root. (nu affects U) maybe exploration?

# Retraining Network
BATCH_SIZE = 256 # used in main.py. set Agent.replay()'s epoch
EPOCHS = 1 # used in main.py. set Agent.replay()'s epoch
REG_CONST = 0.0001 # used in main.py. set NN model's reg_const (maybe U -> exploration?)
LEARNING_RATE = 0.1 # used in main.py. set learning rate of NN model. 
BETA1 = 0.9  # used in model.py Adam Optimizer's momentum
BETA2 = 0.999 # used in model.py Adam Optimizer's momentum
TRAINING_LOOPS = 10 # used in main.py. set Agent.replay()'s iteration number.

HIDDEN_CNN_LAYERS = [
    {'filters':75, 'kernel_size': (4,4)}
    , {'filters':75, 'kernel_size': (4,4)}
    , {'filters':75, 'kernel_size': (4,4)}
    , {'filters':75, 'kernel_size': (4,4)}
    , {'filters':75, 'kernel_size': (4,4)}
    , {'filters':75, 'kernel_size': (4,4)}
]

# Tournament
EVAL_EPISODES = 20 # used in main.py. set number of episodes of best vs current in Tournament
SCORING_THRESHOLD = 1.5 # used in main.py. set threshold to replace best with current