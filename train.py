import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn import preprocessing
import keras
import numpy as np
from data_generator import DataGenerator

url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url)


def encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)

def decode(le, one_hot):
    dec = np.argmax(one_hot, axis=1)
    return le.inverse_transform(dec)


filename = 'spam_dataset.csv'

with open(filename) as f:
    size=len([0 for _ in f])


print("Read " + str(size) + " lines.")

labelstoInt = {'spam' : 0, 'verified' : 1}

params = {  'batch_size': 16,
            'n_classes': 2,
             'filename' : filename,
             'labelstoInt' : labelstoInt }


partition = {}
partition['train'] = range(1,size)

training_generator = DataGenerator(**params).generate(partition['train'])

from keras.layers import Input, Lambda, Dense
from keras.models import Model
import keras.backend as K


def ELMoEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

input_text = Input(shape=(1,), dtype=tf.string)
embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
dense = Dense(256, activation='relu')(embedding)
pred = Dense(2, activation='softmax')(dense)
model = Model(inputs=[input_text], outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    history = model.fit_generator(training_generator,
                                  steps_per_epoch=len(partition['train'])//params['batch_size'],
                                  epochs=1,
                                  verbose=1,
                                  callbacks=None,
                                  validation_data=None,
                                  validation_steps=None,
                                  class_weight=None,
                                  max_queue_size=10,
                                  workers=1,
                                  use_multiprocessing=False,
                                  shuffle=False,
                                  initial_epoch=0)

    model.save_weights('./elmo-model.h5')





