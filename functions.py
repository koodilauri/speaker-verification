import numpy as np
import argparse
from numpy import *
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
import pickle
import warnings
import random

from sklearn.metrics.pairwise import cosine_similarity

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, Flatten, MaxPooling2D, Reshape, concatenate, BatchNormalization, Dropout, Activation
from keras_self_attention import SeqWeightedAttention 

from keras import optimizers
from keras import initializers

def cnn_concat(opt, n_blocks, n_filters, input_shape1, input_shape2):

    # the first branch operates on the first input
    x = create_cnn(opt, n_blocks, n_filters=n_filters, input_shape=input_shape1, tag='mel')
    # the second branch opreates on the second input
    y = create_cnn(opt, n_blocks, n_filters=n_filters, input_shape=input_shape2, tag='jit')

    # combine the output of the two branches
    combineInput = concatenate([x.output, y.output])

    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Dense(400, activation=opt.activation_function, name='Embedding')(combineInput)
    z = Dense(opt.n_classes, activation='softmax', name='Classification')(z)

    # our model will accept the inputs of the two branches
    model = Model(inputs=[x.input, y.input], outputs=z)

    return model

def create_cnn(opt, n_blocks, n_filters, input_shape, tag):
    print('.... Constructing CNN')
    init=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
    inputs = Input(shape=(input_shape))

    if len(n_filters) < n_blocks:
       print('..... Define filter size for every CNN block')
    else:
       for i in range(n_blocks):
         name = ('block%d_conv_1_%s'%(i+1, tag))
         if i == 0:
            x = Conv2D(n_filters[i], kernel_size=(3,3), padding="same", activation=opt.activation_function, name=name)(inputs)
         else:
            x = Conv2D(n_filters[i], kernel_size=(3,3), padding="same", activation=opt.activation_function, name=name)(x)
         name = ('block%d_conv_2_%s'%(i+1, tag))
         x = Conv2D(n_filters[i], kernel_size=(3,3), padding="same", activation=opt.activation_function, name=name)(x)
         name = ('block%d_pool_%s'%(i+1, tag))
         x = MaxPooling2D(pool_size=(2,2), name=name)(x)
         
    x = Reshape((-1,(np.int(x.shape[2]*x.shape[3]))))(x)
    x = SeqWeightedAttention(name='attention_layer_%s'%(tag), return_attention=True)(x)
    x = Dense(1000, activation=opt.activation_function)(x[0])
    # x = Flatten()(x)
    # x = Dense(1000, activation=opt.activation_function)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(500, activation=opt.activation_function)(x)

    model = Model(inputs, x)
    return model

def cnn(opt, n_blocks, n_filters, input_shape):
    print('.... Constructing CNN')
    init=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
    inputs = Input(shape=(input_shape))

    if len(n_filters) < n_blocks:
       print('..... Define filter size for every CNN block')
    else:
       for i in range(n_blocks):
         name = ('block%d_conv_1'%(i+1))
         if i == 0:
            x = Conv2D(n_filters[i], kernel_size=(3,3), padding="same", activation=opt.activation_function, name=name)(inputs)
         else:
            x = Conv2D(n_filters[i], kernel_size=(3,3), padding="same", activation=opt.activation_function, name=name)(x)
         name = ('block%d_conv_2'%(i+1))
         x = Conv2D(n_filters[i], kernel_size=(3,3), padding="same", activation=opt.activation_function, name=name)(x)
         name = ('block%d_pool'%(i+1))
         x = MaxPooling2D(pool_size=(2,2), name=name)(x)
         
    # x = Reshape((-1,(np.int(x.shape[2]*x.shape[3]))))(x)
    # x = SeqWeightedAttention(name='attention_layer', return_attention=True)(x)
    # x = Dense(1000, activation=opt.activation_function)(x[0])
    x = Flatten()(x)
    x = Dense(1000, activation=opt.activation_function)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(400, activation=opt.activation_function, name='Embedding')(x)
    x = Dense(opt.n_classes, activation='softmax', name='Classification')(x)
    model = Model(inputs, x)
    return model

def normalize(features):
    #np.seterr(divide='ignore', invalid='ignore')
    #mean = np.mean(features,axis=0)
    #std = np.std(features, axis=0)
    #features = (features-mean)/std
    features = scale(features, axis=0)
    return features

def get_vector(Features, window_size, seed):
    # random.seed(seed)
    Features = normalize(Features)
    file_size = Features.shape[0]
    index = random.randint(0, max(0,file_size-window_size-1))
    a = np.array(range(min(file_size, window_size)))+index
    new_vec = Features[a,:]
    return np.expand_dims(new_vec, axis=-1)

def read_file(list):
    name = []
    label = []
    lines = list.readlines()
    for x in lines:
        name.append(x.split()[0])
        label.append(x.split()[1])
    return name, label

def read_trials(list):
    name1 = []
    name2 = []
    label = []
    lines = list.readlines()
    for x in lines:
        name1.append(x.split()[0])
        name2.append(x.split()[1])
        label.append(x.split()[2])
    return name1, name2, label

# def load_data(opt, data_names):
#     print('.... Loading data')
#     data = []
#     for i in range(len(data_names)):
#         with open(opt.spec_path + data_names[i] + '.xls1', 'r') as f:
#              dat = read_xls(f)
#              data.append(dat, opt.window_size, data_names[i])
#     return data

def predict_by_model(opt, dnn, val_names, score_file, layer_name):
    model = Model(inputs=dnn.input, outputs=dnn.get_layer(layer_name).output)
    # print(model.summary())
    scores = []
    for i in range(len(val_names[0])):
        # mel spectrogram
        with open(opt.spec_path + val_names[0][i] + '.mel2', 'rb') as f:
             sample11 = get_vector(np.transpose(pickle.load(f)), opt.window_size, opt.spec_path + val_names[0][i])
             sample11 = np.expand_dims(sample11,axis=0)
        # jitter & shimmer
        # with open(opt.spec_path + val_names[0][i] + '.xls1', 'r') as f:
            #  dat = read_xls(f)
            #  sample12 = get_vector(dat, opt.window_size, opt.spec_path + val_names[0][i])
            #  sample12 = np.expand_dims(sample12,axis=0)
        with open(opt.spec_path + val_names[1][i] + '.mel2', 'rb') as f:
             sample21 = get_vector(np.transpose(pickle.load(f)), opt.window_size, opt.spec_path + val_names[1][i])
             sample21 = np.expand_dims(sample21,axis=0)
        # with open(opt.spec_path + val_names[1][i] + '.xls1', 'r') as f:
            #  dat = read_xls(f)
            #  sample22 = get_vector(dat, opt.window_size, opt.spec_path + val_names[1][i])
            #  sample22 = np.expand_dims(sample22,axis=0)
        # sample1 = model.predict([sample11,sample12])
        # sample2 = model.predict([sample21,sample22])
        sample1 = model.predict([sample11])
        sample2 = model.predict([sample21])
        scores.append(np.squeeze(cosine_similarity(sample1, sample2)))
        #print('%d %s' %(i,scores[i]))
    scores = np.array(scores)
    out = open(score_file, 'w')
    for a in scores:
        out.write('%s\n'%a)
    out.close()
    print('..... Scores are written in: %s' %score_file)

def read_xls(f):
    lines = f.readlines()
    r = []
    for x in lines:
        x = x.rstrip().split('\t')
        r.append(x[:-1]) # append all features except last one (shim apq11)
    ar = np.array(r).astype(np.float)
    return ar