import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import load_model, Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras.utils import plot_model

import config
import logger
import setting

class ModelManager:
  '''
  ModelManager class is a wrapper class to manage various models. 
  '''
  def __init__(self, learning_rate, input_dim, output_dim):
    '''
    create new ModelManager instance. 

    Parameters
      learning_rate: float. learning rate
      input_dim: input dimension
      output_dim: output dimension

    Fields
      self.learning_rate: learning rate
      self.input_dim: input dimension
      self.output_dim: output dimension
      self.model: keras.Model instance (model instance)
    '''
    self.learning_rate = learning_rate
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.model = None
  
  def summary(self):
    '''
    print summary of the model
    '''
    self.model.summary()

  def predict(self, x):
    '''
    return prediction for input

    Parameters
      x: np array
    '''
    return self.model.predict(x)
  
  def fit(self, x, y, epochs, verbose, callbacks, validation_split, batch_size):
    '''
    train the model using datasets and some options.

    Parameters
      x: input datasets (observation)
      y: output datasets (action probabilities, and estimated action value)
      epochs: int of train epoches
      verbose: 0, 1, or 2 of logging (0: silent, 1: progress bar, 2: line per epoch)
      callbacks: list of keras.callbacks.Callback. 
      validation_split: float between 0 and 1. fraction of validation data. 
      batch_size: int of batch size
    
    Return
      history: return of fit function
    '''
    return self.model.fit(x, y, 
                          epochs=epochs, 
                          verbose=verbose, 
                          callbacks=callbacks, 
                          validation_split=validation_split, 
                          batch_size=batch_size)

  def save(self, game, version):
    '''
    save current model to run_folder (see setting.py)

    Parameters
      game: str. name of the game
      version: int. version of the model about the game
    '''
    self.model.save(setting.run_folder + 'models/{}_version{}'.format(game, str(version).zfill(6)) + '.h5')

  def load(self, game, run_number, version):
    '''
    load archived model from run_archive_folder (see setting.py)

    Parameters
      game: str. name of the game
      run_number: int. run_number
      version: int. version of the model about the game

    Return
      return: loaded model object
    '''
    return load_model(setting.run_archive_folder + game + \
                      '/run' + str(run_number).zfill(6) + \
                      '/models/{}_version{}'.format(game, str(version).zfill(6)) + '.h5')
  
  def print_weight_averages(self):
    '''
    Log each layer's avg, stddev of weights. 
    Also, save visualized filters at /model 
    '''
    for i, layer in enumerate(self.model.layers):
      try:
        # log statistics of leach layer. 
        x0 = layer.get_weights()[0]
        logger.model_logger.info('WEIGHT LAYER {}: ABS_AVG = {}, STD_DEV = {}, ABS_MAX = {}, ASB_MIN = {}'
                                .format(i, np.mean(np.abs(x0)), np.std(x0), np.max(np.abs(x0)), np.min(np.abs(x0))))
      except Exception:
        pass
    logger.model_logger.info('------------------------------------------------------')
    for i, layer in enumerate(self.model.layers):
      try:
        # log statistics of each layer. 
        x1 = layer.get_weights()[1]
        logger.model_logger.info('BIAS LAYER {}: ABS_AVG = {}, STD_DEV = {}, ABS_MAX = {}, ASB_MIN = {}'
                                .format(i, np.mean(np.abs(x1)), np.std(x1), np.max(np.abs(x1)), np.min(np.abs(x1))))
      except Exception:
        pass
    logger.model_logger.info('******************************************************')

  def plot_model(self, to_file):
    '''
    Plot the model as .png file at to_file.

    Parameters
      to_file: str. relative path of target plot image. 
    '''
    plot_model(self.model, to_file=to_file, show_shapes=True)

  def view_layers(self, game, version):
    '''
    Visualize each Convolutional layers. 
    The visualized layers are stored in run_folder/models/layer{id}.png. 
    Because there are so many input channels and output channels, 
    this function limits the maximum channels to 12. 

    The output image consists of grid of filters, M x N grids. 
    M means output filter's number, and N means input channel's number. 
    For example, if input channel is 2 and output filters are 75, 
    the result will be a 75 x 2 grids image. 

    Parameters
      game: str. name of the game
      version: int. version of the model about the game
    '''
    for i, layer in enumerate(self.model.layers):
      try:
        # number of rows means number of output filters (channels), 
        # and number of columns means number of input channels. 
        x0 = layer.get_weights()[0]
        layer_shape = x0.shape
        channel_size = min(layer_shape[2], 12)
        filter_size = min(layer_shape[3], 12)
        fig = plt.figure(figsize=(channel_size, filter_size))
        for f in range(filter_size):
          for ch in range(channel_size):
            sub = fig.add_subplot(filter_size, channel_size, f * channel_size + ch + 1)
            sub.imshow(x0[:, :, ch, f], cmap='coolwarm', clim=(-1, 1), aspect='auto')
        fig.savefig(setting.run_folder + 'models/{}_version{}_filter{}.png'.format(game, str(version).zfill(6), i))
        plt.close(fig)
      except Exception:
        pass

class ResidualCNNManager(ModelManager):
  '''
  Child of ModelManager. This uses multiple residual blocks. 
  '''
  def __init__(self, reg_const, learning_rate, input_dim, output_dim, hidden_layers):
    '''
    create new ResidualCNNManager instance, and build model for it. 

    Parameters
      reg_const: float. kernel regularizer constant
      learning_rate: float. learning rate
      input_dim: input dimension
      output_dim: output dimension
      hidden_layers: list of dictionaries. info about hidden layers
        filters: int, number of output filters
        kernel_size: int or tuple, size of kernel for that layer
    
    Fields
      self.hidden_layers: info about hidden layers (list of dict)
      self.reg_const: regularization constant of kernels
      self.model: generated Residual CNN Keras model
    '''
    ModelManager.__init__(self, learning_rate, input_dim, output_dim)
    self.hidden_layers = hidden_layers
    self.reg_const = reg_const
    self.model = self.__build_model()
  
  def load(self, game, run_number, version):
    '''
    override ModelManager's load() function. 
    (with softmax_cross_entropy_with_logits_v2)
    '''
    return load_model(setting.run_archive_folder + game + \
                      '/run' + str(run_number).zfill(6) + \
                      '/models/{}_version{}'.format(game, str(version).zfill(6)) + '.h5', 
                      custom_objects={'softmax_cross_entropy_with_logits_v2': tf.nn.softmax_cross_entropy_with_logits})

  def __build_model(self):
    '''
    build a residual model

    0. convolution block
    1. some residual blocks
    2. policy head (softmax) and value head (tanh)
    '''
    # define input layer
    main_input = Input(shape=self.input_dim, name='main_input')

    # first, add convolution block
    x = self.__conv_block(main_input, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])

    # if there are more than 2 hidden layers, add them as residual blocks.
    if len(self.hidden_layers) > 1:
      for h in self.hidden_layers[1:]:
        x = self.__residual_block(x, h['filters'], h['kernel_size'])
    
    # get policy head and value head
    value_head = self.__value_head(x)
    policy_head = self.__policy_head(x)
    
    # compile and return model
    model = Model(inputs=[main_input], outputs=[value_head, policy_head])
    model.compile(loss={ 'value_head': 'mean_squared_error', 'policy_head': tf.nn.softmax_cross_entropy_with_logits }, 
                  optimizer=Adam(learning_rate=self.learning_rate, beta_1=config.BETA1, beta_2=config.BETA2), 
                  loss_weights={'value_head': 0.5, 'policy_head': 0.5})
    return model
  
  def __conv_block(self, x, filters, kernel_size):
    '''
    returns x following by 1x1 convolution block

    Parameters
      x: input layer
      filters: int. number of filters
      kernel_size: int or tuple. size of kernel
    
    Return
      return: output layers with convolution block.
    '''
    x = Conv2D(filters=filters, 
               kernel_size=kernel_size, 
               data_format='channels_last', 
               padding='same', 
               use_bias=False, 
               activation='linear', 
               kernel_regularizer=regularizers.l2(self.reg_const)
              )(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU()(x)
    
    return (x)
  
  def __residual_block(self, x, filters, kernel_size):
    '''
    returns residual block after input layers.

    Parameters
      x: input layers with 'filters' channels
      filters: int. number of filters
      kernel_size: int or tuple. size of the kernel
    '''
    x_res = self.__conv_block(x, filters, kernel_size)
    x_res = Conv2D(filters=filters, 
                   kernel_size=kernel_size, 
                   data_format='channels_last', 
                   padding='same', 
                   use_bias=False, 
                   activation='linear', 
                   kernel_regularizer=regularizers.l2(self.reg_const)
                  )(x_res)
    x_res = BatchNormalization(axis=-1)(x_res)
    x = add([x, x_res])
    x = LeakyReLU()(x)
    return (x)

  def __value_head(self, x):
    '''
    return value head for this model.

    Parameters
      x: input layers
    
    Return
      return: output layer for value (tanh activation)
    '''
    x = Conv2D(filters=1, 
               kernel_size=(1, 1), 
               data_format='channels_last', 
               padding='same', 
               use_bias=False, 
               activation='linear', 
               kernel_regularizer=regularizers.l2(self.reg_const)
              )(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    x = Dense(20, use_bias=False, 
              activation='linear', 
              kernel_regularizer=regularizers.l2(self.reg_const)
             )(x)
    x = LeakyReLU()(x)
    x = Dense(1, use_bias=False, 
              activation='tanh', 
              kernel_regularizer=regularizers.l2(self.reg_const),
              name='value_head'
             )(x)
    return (x)

  def __policy_head(self, x):
    '''
    return policy head for this model.

    Parameters
      x: input layers
    
    Return
      return: output layer for policy (linear activation)
    '''
    x = Conv2D(filters=2, 
               kernel_size=(1, 1), 
               data_format='channels_last', 
               padding='same', 
               use_bias=False, 
               activation='linear', 
               kernel_regularizer=regularizers.l2(self.reg_const)
              )(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    x = Dense(self.output_dim, use_bias=False, 
              activation='linear', 
              kernel_regularizer=regularizers.l2(self.reg_const),
              name='policy_head'
             )(x)
    return (x)

