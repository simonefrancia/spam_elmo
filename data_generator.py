import numpy as np
import os
import keras.backend as K
from keras.utils import np_utils
import keras
global_path = ""

class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, n_ch = 1, batch_size = 32, shuffle = False, n_classes=1, filename=None, labelstoInt = None):

      'Initialization'

      self.n_classes = n_classes
      self.n_ch = n_ch
      self.batch_size = batch_size
      self.shuffle = shuffle
      self.filename = filename
      self.labelstoInt = labelstoInt


  def generate(self, list_IDs):
      'Generates batches of samples'

      # Infinite loop
      while 1:

          # Generate order of exploration of dataset
          indexes = self.__get_exploration_order(list_IDs)

          # Generate batches
          imax = int(len(indexes)/self.batch_size)

          for i in range(imax):

              # Find list of IDs
              list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

              # Generate data
              X, y = self.__data_generation(list_IDs_temp)

              yield X, y

  def __get_exploration_order(self, list_IDs):
      'Generates order of exploration'

      # Find exploration order
      indexes = np.arange(len(list_IDs))
      if self.shuffle == True:
          np.random.shuffle(indexes)
      return indexes

  def __data_generation(self,  list_IDs_temp):

      'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)

      labels = []
      sentences = []

      for i, ID in enumerate(list_IDs_temp):
          with open(self.filename) as fp:
              for i, line in enumerate(fp):
                  if i == ID:
                      sentences.append(line.split("\t")[1])
                      l = line.split("\t")[0].lower()
                      if l == 'spam':
                          labels.append(self.labelstoInt['spam'])
                      elif l == 'verified':
                          labels.append(self.labelstoInt['verified'])
                      else:
                          print("errore")


      return np.array(sentences), keras.utils.to_categorical(labels,num_classes=self.n_classes)
