import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, concatenate
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, Conv2DTranspose, ZeroPadding2D
from keras.optimizers import Adam, Adadelta, Adamax, Nadam, Adagrad, SGD, RMSprop

SMOOTH = 1  # constant value for internal dice coefficient calculations

def dice_coef(y_true, y_pred, smooth=1):
    """Dice coefficient function to use outside of the U-Net model for calculations."""
    side = len(y_true[0])
    y_true_f = y_true.reshape(side * side)
    y_pred_f = y_pred.reshape(side * side)
    intersection = sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)

def _dice_coef(y_true, y_pred, smooth=1):
    """Dice coefficient for use with Tensorflow fitting. For INTERNAL use only."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def _dice_coef_loss(y_true, y_pred, smooth=1):
    """Dice coefficient loss. Simply just the negative dice coefficient. For INTERNAL use only."""
    return -_dice_coef(y_true, y_pred, smooth)

class UNet:
    def __init__(self, optimizer=Adam, sample_width=128, sample_height=128,
                 learning_rate=1e-3, weight_decay=1e-3, momentum=0.8):
        self.optimizer = optimizer
        self.sample_width = sample_width
        self.sample_height = sample_height
        self.loss_metric = _dice_coef_loss
        self.metrics = [_dice_coef]
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay    # Used only for Adam
        self.momentum = momentum            # Used only for SGD
        self.history = None
        self.model = None; self._initializeArchitecture()
    
    def fit(self, train_x: np.array, train_y: np.array, batch_size=None, 
            epochs=150, verbose=1, shuffle=True, validation_split=0.2):
        """Fit the model and set the Keras history."""
        if not batch_size:
            batch_size = int(len(train_x) / 4)  # Initialize here, since we can't do it in parameters
        self.history = self.model.fit(train_x, train_y, batch_size=batch_size, 
                                      epochs=epochs, verbose=verbose, shuffle=shuffle, 
                                      validation_split=validation_split)
    
    def predict(self, test_x: np.array, verbose=1) -> np.array:
        """Predict and return a mask for the test image."""
        return self.model.predict(test_x, verbose=verbose)
        
    def plot(self, title='Model Accuracy'):
        """Summarize the network history for accuracy."""
        plt.plot(self.history.history['_dice_coef'])
        plt.plot(-np.array(self.history.history['val_loss']))
        plt.title(title)
        plt.ylabel('Dice coefficient')
        plt.xlabel('Epochs')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.show()
        
    def _initializeArchitecture(self):
        """Initialize the architecture for the U-Net."""
        inputs = Input((self.sample_width, self.sample_height, 1))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        drop1 = Dropout(0.5)(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(drop1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        drop2 = Dropout(0.5)(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(drop2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        drop3 = Dropout(0.3)(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(drop3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        drop4 = Dropout(0.3)(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(drop4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), 
                                           padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), 
                                           padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), 
                                           padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), 
                                           padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
        model = Model(inputs=[inputs], outputs=[conv10])
        # Compile the network based off of its optimizer. 
        if self.optimizer is Adam:
            model.compile(optimizer=self.optimizer(lr=self.learning_rate, decay=self.weight_decay), 
                          loss=self.loss_metric, metrics=self.metrics)
        elif self.optimizer in [Adamax, Nadam, Adadelta]:
            model.compile(optimizer=self.optimizer(lr=self.learning_rate), 
                          loss=self.loss_metric, metrics=self.metrics)
        elif self.optimizer is SGD:
            model.compile(optimizer=self.optimizer(lr=self.learning_rate, momentum=self.momentum), 
                          loss=self.loss_metric, metrics=self.metrics)
        self.model = model