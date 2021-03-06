from numpy import *
import argparse
import os.path as path
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
#import Extract_Features as features
from keras.models import load_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras_self_attention import SeqWeightedAttention

import functions
from my_classes import DataGenerator
import dotenv

import tensorflow as tf
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)

continue_from_last_trained_model=True
dotenv.load_dotenv(verbose=True)

#np.set_printoptions(threshold=sys.maxsize)

def main(opt):

 if opt.train == '1':
   print('|             Training a CNN based Speaker Verification System                           |')
   print(' ******************************************************************************************\n')

   training_filename = ('training_labels.lst')
   training_list = open(training_filename, "r")
   show_names, show_labels = functions.read_file(training_list)

   # To encode target labels with value between 0 and n_classes-1
   label_encoder = LabelEncoder()
   data_labels = label_encoder.fit_transform(show_labels)
   opt.n_classes = len(np.unique(data_labels))
   print('Number of classes',len(np.unique(data_labels)))

   n_frames = opt.window_size
   n_features1 = 80 #
   n_features2 = 9 #
   n_channels = 1
   optm = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

   input_shape1 = (n_frames,n_features1,n_channels)
   input_shape2 = (n_frames,n_features2,n_channels)

   # Partitions
   train_names, val_names = train_test_split(show_names, test_size=0.20, random_state=4)
   partition = {'train':train_names, 'validation':val_names}

   zipObj = zip(show_names,data_labels)
   labels = dict(zipObj) # dictionary looks like this {'id10278/QOq66XogW3Q/00005': 8, ...}

   # Parameters
   params = {'dim1': (n_frames, n_features1),
                       'dim2': (n_frames, n_features2),
                       'n_frames': n_frames,
                       'batch_size': opt.batch_size,
                       'n_classes': opt.n_classes,
                       'n_channels': n_channels,
                       'shuffle': True,
                       'suffixes': ['.mel2','.xls3']}
   print('DataGenerator Params', params)

   # Generators
   training_generator = DataGenerator(partition['train'], labels, **params)
   validation_generator = DataGenerator(partition['validation'], labels, **params)

   # comment out below if loading an existing model instead...
   #  model = functions.cnn(opt, 3, n_filters=[128,256,256], input_shape=input_shape1)
   

   model_name = 'cnn_pulse.h5'
   if(continue_from_last_trained_model):
     print('Continuing from a saved model...')
     model = load_model(model_name)
     #  model = load_model(model_name, custom_objects=SeqWeightedAttention.get_custom_objects())
     #model.compile(optimizer=optm, loss='categorical_crossentropy', metrics = ['accuracy'])
     #model.set_weights(last_model.get_weights())
   else:
	   model = functions.cnn_concat(opt, 3, n_filters=[128,256,256], input_shape1=input_shape1, input_shape2=input_shape2)
	  #  model = functions.cnn(opt, 3, n_filters=[128,256,256], input_shape=input_shape2)
	   model.compile(optimizer=optm, loss='categorical_crossentropy', metrics = ['accuracy'])
   
   model.summary()

   checkpoint = ModelCheckpoint(model_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
   callbacks_list = [checkpoint]

   model.fit_generator(generator=training_generator,
                      epochs=opt.max_epochs,
                      validation_data=validation_generator,
                      verbose=1,
                      shuffle=True,
                      #workers=2,
                      #use_multiprocessing=True,
                      callbacks=callbacks_list)
   print('.... Saving model \n')
  #  model.save(opt.save_dir + model_name, overwrite=True)

 if opt.predict == '1':
   print(' -------------------------------------------------')
   print('|          Prediciting using trained CNN based Speaker Verification Model                            |')
   print('******************************************************************************************************\n')

   validation_trials = 'VoxCeleb-1_validation_trials.txt'
   validation_list = open(validation_trials, "r")
   validation_names = functions.read_trials(validation_list)
   #print(validation_names)
   #exit(1)

   model_name = 'cnn_conv1d3-b3-k11.h5'
   model = load_model('./models/triton/07-09/' + model_name)
  #  model = load_model(opt.save_dir + model_name)
  #  model = load_model(opt.save_dir + model_name, custom_objects=SeqWeightedAttention.get_custom_objects())
   model.summary()
   print('Model %s loaded' %model_name)

   score_file = './scores/' + model_name[:-3]
   functions.predict_by_model(opt, model, validation_names, score_file, 'Embedding') #  concatenate_1 _mel
   print('.... Done prediction with model : %s' %model_name)

if __name__=="__main__":

   parser = argparse.ArgumentParser(description='A CNN based Speaker Verification System.')

   parser.add_argument('--train', default = 1, help='1 for trainning, 0 for predicting')
   parser.add_argument('--predict', default = 0, help='0 for trainning, 1 for predicting')

   #paths
   parser.add_argument('--spec_path', type=str, default =os.getenv("SOUND_FILE_PATH"), help='spectrograms path')
   parser.add_argument('--save_dir', type=str, default='./models/', help='where model is saved')

   #optmization:
   parser.add_argument('--window_size', type=int, default=350, help='Number of frames in a sample')
   parser.add_argument('--batch_size', type=int, default=16, help='number of sequences to train on in parallel')
   parser.add_argument('--max_epochs', type=int, default=100, help='number of full passes through the training data')
   parser.add_argument('--activation_function', type=str, default='relu', help='Activation function')
   parser.add_argument('--n_classes',  type=int, help='Number of classes')
   parser.add_argument('--seed', type=int, default=3435, help='random number generator seed')

   params=parser.parse_args()
   np.random.seed(params.seed)

   main(params)
