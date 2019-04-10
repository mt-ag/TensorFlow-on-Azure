#tensorflow
import tensorflow as tf
from tensorflow.python.client import device_lib

#keras
import keras
from keras.backend.tensorflow_backend import set_session
from keras.datasets import mnist
from keras import models
from keras import layers
from keras import regularizers
from keras.utils import to_categorical
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras import callbacks

#numpy
import numpy as np

#sklearn
from sklearn.metrics import confusion_matrix

#itertools
import itertools

#matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#time
import time

#os
import os

# Azure Machine Learning
from azureml.core.run import Run
run = Run.get_context()

# Callback to keep track of training progress
from computeMetrics import ComputeMetrics

# parsing arguments
import argparse

def main():
    
    # get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256, help='size of each batch')
    parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs to train')
    
    args = parser.parse_args()

    #Force GPU support with growing memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)

    #keras / tensorflow has already the full MNIST dataset
    (train_images_raw, train_labels_raw), (test_images_raw, test_labels_raw) = mnist.load_data()

    #fraktion of the training to the validation samples
    a_train = int(train_images_raw.shape[0] * 0.9)

    images = train_images_raw.reshape((train_images_raw.shape[0], 28, 28, 1))
    images = images.astype('float32') / 255 #255 different gray scales

    train_images = images[ : a_train]
    valid_images = images[a_train : ]
    print("Amount of all images:{}".format(images.shape))
    print("Amount of all training images:{}".format(train_images.shape))
    print("Amount of all validation images:{}".format(valid_images.shape))

    #convert labels into one hot representation
    labels = to_categorical(train_labels_raw)
    train_labels = labels[ : a_train]
    valid_labels = labels[a_train : ]
    print("Amount of all labels:{}".format(labels.shape))
    print("Amount of all training labels:{}".format(train_labels.shape))
    print("Amount of all validation labels:{}".format(valid_labels.shape))


    test_images = test_images_raw.reshape((test_images_raw.shape[0], 28, 28, 1))
    test_images = test_images.astype('float32') / 255 #255 different gray scales
    test_labels = to_categorical(test_labels_raw)

    model = models.Sequential()
    # Convolution Layers
    model.add(layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape = (28, 28, 1)))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(32, (5, 5), padding = 'same', activation = 'relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu', input_shape = (28, 28, 1)))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (5, 5), padding = 'same', activation = 'relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu'))
    model.add(layers.Dropout(0.4))

    # Fully connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(10 , activation='softmax'))

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(optimizer = optimizer , loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    datagen = ImageDataGenerator(
        # Wieviel Grad [0, 180] darf sich das Bild hin und her drehen
        rotation_range = 10, 
        # Um wieviel darf das Bild nach links und rechts bewegt werden
        width_shift_range=0.1, 
        # Um wieviel darf das Bild nach Oben und Unten bewegt werden
        height_shift_range=0.1, 
        # Die Varianz von dunkel zu hell 
        #brightness_range = (0.1, 2), 
        # In welchem Bereich darf das Bild abgeschert werden
        shear_range = 0.1, 
        # In welchem Bereich dar das Bild die größe ändern
        zoom_range = 0.1, 
        # Bei Zahlen sollte man auf das Spiegeln eher verzichten :)
        horizontal_flip = False, 
        # Bei Zahlen sollte man auf das Spiegeln eher verzichten :)
        vertical_flip = False)


    learning_rate_reduction = callbacks.ReduceLROnPlateau(
                                            # val_acc zeigt die letzten verlusraten
                                            monitor='val_acc', 
                                            # wie lange halten wir die Fuesse still, bevor wir die Lernrate anpassen
                                            patience=3, 
                                            verbose=1,
                                            # Die lernrate wird halbiert
                                            factor=0.5, 
                                            # ab wann brechen wir ab
                                            min_lr=0.00001)

    early_stopping = callbacks.EarlyStopping(
                                                # über welchen Monitor beobachtet man die Änderung der letzten Iterationen?
                                                monitor = 'acc',
                                                # wie lange halten wir die Fuesse still, bis wir das Lernen abbrechen
                                                patience = 3)


    model_checkpoint = callbacks.ModelCheckpoint(
                                                # Dateiname mit Pfad relativ zu dieserm Code
                                                filepath = './outputs/augmenting_cnn_MNIST_model.h5',
                                                # Welcher Monitor soll beobachtet werden, um über die äbderung der Qualität zu entscheiden
                                                monitor = 'val_loss',
                                                # Nur speciern, wenn sich auch was verbessert hat
                                                save_best_only = True)


    os.makedirs('./outputs', exist_ok=True)

    start = time.time()

    history  = model.fit_generator(
        datagen.flow(train_images, train_labels, batch_size = args.batch_size), 
        steps_per_epoch = train_images.shape[0] // args.batch_size, 
        epochs = args.num_epochs,
        validation_data = (valid_images, valid_labels),
        callbacks=[learning_rate_reduction, early_stopping, model_checkpoint, ComputeMetrics(run)]
    )
    
    print("It took :", time.time() - start)

    # serialize model to JSON
    model.save('./outputs/final_MNIST_model.h5')

    (test_loss, test_acc) = model.evaluate(test_images, test_labels)

    print("Loss: ", test_loss)
    print("Accuracy: ", test_acc)

    run.log('best_val_acc', np.float(test_acc))

if __name__ == "__main__":
    main()