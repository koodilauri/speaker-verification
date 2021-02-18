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
from keras.layers import Input, Dense, Flatten, Conv1D, Conv2D, Flatten, MaxPooling1D, MaxPooling2D, Reshape, concatenate, BatchNormalization, Dropout, Activation
from keras_self_attention import SeqWeightedAttention 

from keras import optimizers
from keras import initializers

def cnn_concat(opt, n_blocks, n_filters, input_shape1, input_shape2):

    # the first branch operates on the first input
    x = create_cnn(opt, n_blocks, n_filters=n_filters, input_shape=input_shape1, tag='mel')
    # the second branch opreates on the second input
    y = create_jitter_shimmer(opt, n_blocks, n_filters=[32,64,126], input_shape=input_shape2, tag='jit')

    # combine the output of the two branches
    combineInput = concatenate([x.output, y.output], name='Embedding')

    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Dense(opt.n_classes, activation='softmax', name='Classification')(combineInput)

    # our model will accept the inputs of the two branches
    model = Model(inputs=[x.input, y.input], outputs=z)

    return model

def create_jitter_shimmer(opt, n_blocks, n_filters, input_shape, tag):
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
         x = MaxPooling2D(pool_size=(2,1), name=name)(x)
        #  if i == 2:
        #     x = MaxPooling2D(pool_size=(3,2), name=name)(x)
        #  else:
        #     x = MaxPooling2D(pool_size=(3,1), name=name)(x)
    # x = Reshape((-1,(np.int(x.shape[2]*x.shape[3]))))(x)
    # x = SeqWeightedAttention(name='attention_layer_%s'%(tag), return_attention=True)(x)
    # x = Dense(1000, activation=opt.activation_function)(x[0])
    x = Flatten()(x)
    x = Dense(500, activation=opt.activation_function)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(400, activation=opt.activation_function, name='Embedding_%s'%(tag))(x)

    model = Model(inputs, x)
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
         
    # x = Reshape((-1,(np.int(x.shape[2]*x.shape[3]))))(x)
    # x = SeqWeightedAttention(name='attention_layer_%s'%(tag), return_attention=True)(x)
    # x = Dense(1000, activation=opt.activation_function)(x[0])
    x = Flatten()(x)
    x = Dense(500, activation=opt.activation_function)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(400, activation=opt.activation_function, name='Embedding_%s'%(tag))(x)

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
         x = MaxPooling2D(pool_size=(2,1), name=name)(x)
         
    # x = Reshape((-1,(np.int(x.shape[2]*x.shape[3]))))(x)
    # x = SeqWeightedAttention(name='attention_layer', return_attention=True)(x)
    # x = Dense(1000, activation=opt.activation_function)(x[0])
    x = Flatten()(x)
    x = Dense(500, activation=opt.activation_function)(x)
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

def get_vector(Features, window_size, seed, index=-1):
    # random.seed(seed)
    Features = normalize(Features)
    file_size = Features.shape[0]
    # if index is not provided (deafult=-1), randomly choose the index
    if index < 0:
        index = random.randint(0, max(0,file_size-window_size-1))
    a = np.array(range(min(file_size, window_size)))+index
    new_vec = Features[a,:]
    # print(file_size, new_vec.shape, seed, index)
    return np.expand_dims(new_vec, axis=-1), index

def get_vector2(Features, window_size, seed, index=-1):
    # random.seed(seed)
    file_size = Features.shape[0]
    # if index is not provided (deafult=-1), randomly choose the index
    if index < 0:
        index = random.randint(0, max(0,file_size-window_size-1))
    a = np.array(range(min(file_size, window_size)))+index
    new_vec = Features[a,:]
    # print(file_size, new_vec.shape, seed, index)
    return np.expand_dims(new_vec, axis=-1), index

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
def predict_by_model2(opt, dnn, val_names, score_file, layer_name):
    model = Model(inputs=dnn.input, outputs=dnn.get_layer(layer_name).output)
    
    scores = []
    for i in range(len(val_names[0])):
        with open(opt.spec_path + val_names[0][i] + '.xls3', 'r') as f:
             dat = read_xls(f)
             sample1, i1 = get_vector2(dat, opt.window_size, opt.spec_path + val_names[0][i])
             sample1 = np.expand_dims(sample1,axis=0)
        with open(opt.spec_path + val_names[1][i] + '.xls3', 'r') as f:
             dat = read_xls(f)
             sample2, i2 = get_vector2(dat, opt.window_size, opt.spec_path + val_names[1][i])
             sample2 = np.expand_dims(sample2,axis=0)
        sample1 = model.predict([sample1])
        sample2 = model.predict([sample2])
        scores.append(np.squeeze(cosine_similarity(sample1, sample2)))
        #print('%d %s' %(i,scores[i]))
    scores = np.array(scores)
    out = open('%s_2'%(score_file), 'w')
    for a in scores:
        out.write('%s\n'%a)
    out.close()
    print('..... Scores are written in: %s' %score_file)

def predict_by_model(opt, dnn, val_names, score_file, layer_name):
    model = Model(inputs=dnn.input, outputs=dnn.get_layer(layer_name).output) # 'Embedding_mel
    model_jit = Model(inputs=dnn.input, outputs=dnn.get_layer('Embedding_jit').output)
    # print(model.summary())
    scores = []
    scores2 = []
    weights = np.arange(0.75,1,0.01) #0.705,0.95,0.01
    for i in range(len(val_names[0])):
        # mel spectrogram
        with open(opt.spec_path + val_names[0][i] + '.mel2', 'rb') as f:
             sample11, i11 = get_vector(np.transpose(pickle.load(f)), opt.window_size, opt.spec_path + val_names[0][i])
             sample11 = np.expand_dims(sample11,axis=0)
        # jitter & shimmer
        with open(opt.spec_path + val_names[0][i] + '.xls3', 'r') as f:
             dat = read_xls(f)
             sample12, i12 = get_vector2(dat, opt.window_size, opt.spec_path + val_names[0][i], i11)
             sample12 = np.expand_dims(sample12,axis=0)
        # 2nd sample
        with open(opt.spec_path + val_names[1][i] + '.mel2', 'rb') as f:
             sample21, i21 = get_vector(np.transpose(pickle.load(f)), opt.window_size, opt.spec_path + val_names[1][i])
             sample21 = np.expand_dims(sample21,axis=0)
        with open(opt.spec_path + val_names[1][i] + '.xls3', 'r') as f:
             dat = read_xls(f)
             sample22, i22 = get_vector2(dat, opt.window_size, opt.spec_path + val_names[1][i], i21)
             sample22 = np.expand_dims(sample22,axis=0)
        #sample1 = model.predict([sample11,sample12])
        #sample2 = model.predict([sample21,sample22])
        sample1 = model.predict([sample11,sample12]) # mel
        sample1_jit = model_jit.predict([sample11,sample12])
        sample2 = model.predict([sample21,sample22]) # mel
        sample2_jit = model_jit.predict([sample21,sample22])
        # scores.append(np.squeeze(cosine_similarity(sample1, sample2)))
        s = []
        for w in weights:
            s.append(weighted_predict2(sample1, sample2, sample1_jit, sample2_jit, w))
        scores.append(s)
        s = []
        # for w in weights:
        #     s.append(weighted_predict1(sample1, sample2, sample1_jit, sample2_jit, w))
        # scores2.append(s)
        #print('%d %s' %(i,scores[i]))
    scores = np.array(scores)
    # scores2 = np.array(scores2)
    for i in range(len(weights)):
        out = open('%s_1_%s'%(score_file, round(weights[i],3)), 'w')
        for a in scores:
            out.write('%s\n'%a[i])
        out.close()
        print('..... Scores are written in: %s_%s' %(score_file, round(weights[i],3)))
    # for i in range(len(weights)):
    #     out = open('%s_w_%s'%(score_file, round(weights[i],3)), 'w')
    #     for a in scores2:
    #         out.write('%s\n'%a[i])
    #     out.close()
    #     print('..... Scores are written in: %s_w_%s' %(score_file, round(weights[i],3)))

def weighted_predict1(mel_sample1, mel_sample2, jit_sample1, jit_sample2, w):
    sample1 = mel_sample1 * w + jit_sample1 * (1-w)
    sample2 = mel_sample2 * w + jit_sample2 * (1-w)
    return np.squeeze(cosine_similarity(sample1,sample2))

def weighted_predict2(mel_sample1, mel_sample2, jit_sample1, jit_sample2, w):
    mel = np.squeeze(cosine_similarity(mel_sample1, mel_sample2))
    jit = np.squeeze(cosine_similarity(jit_sample1, jit_sample2))
    return mel * w + jit * (1-w)


def read_xls(f):
    lines = f.readlines()
    r = []
    for x in lines:
        x = x.rstrip().split('\t')
        # r.append( + [x[13]] + [x[14]] + [x[16]])# append the 9 first features (shim apq11 skipped)
        r.append([x[1]] + [x[6]] + [x[7]])# pulse [x[1]] + [x[6]] + [x[7]]  x[:9]
    ar = np.array(r).astype(np.float)
    return ar