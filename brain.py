
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from tensorflow.keras.optimizers import Adam

class Brain(object):
    def __init__(self, learning_rate = 0.001, num_actions = 5):
        self.learning_rate = learning_rate
        
        # Inputs: Server temperature, number of users, rate of data
        states = Input(shape = (3, ))
        x = Dense(units = 64, activation = 'sigmoid')(states)
        x = Dropout(rate = 0.1)(x)
        y = Dense(units = 32, activation = 'sigmoid')(x)
        y = Dropout(rate = 0.1)(y)
        
        q_values = Dense(units = num_actions, activation = 'softmax')(y)
        
        self.model = Model(inputs = states, outputs = q_values)
        self.model.compile(loss = 'mse', optimizer = Adam(learning_rate = learning_rate))
        