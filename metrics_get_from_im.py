from utils import *
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose
# from keras.callbacks import ModelCheckpoint , EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import keras

import numpy as np
from sklearn.utils import shuffle
import cv2
from skimage.util import view_as_windows
import argparse
import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import metrics

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report
# Local

from metrics import fmeasure,categorical_accuracy
import deb
from keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy
from main import NetObject,Dataset

data=Dataset()
#name='im_reconstructed_rgb_test_predictionplen64_3.png'
#name='im_reconstructed_rgb_test_predictionskip.png'
#name='im_reconstructed_rgb_test_predictionno_skip.png'
#name='im_reconstructed_rgb_test_predictiondropout_0_5.png'
name='im_reconstructed_rgb_test_predictiondropout_0_2.png'

metrics=data.metrics_per_class_from_im_get(name=name,average=None,folder='../results/final_results/')


