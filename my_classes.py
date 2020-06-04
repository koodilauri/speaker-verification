import numpy as np
import keras
import pickle
import os
import functions
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim1=(100,128),dim2=(100,9), n_channels=1,
                 n_classes=10, shuffle=True, n_frames=100, suffixes=[None,None]):
        'Initialization'
        self.dim1 = dim1
        self.dim2 = dim2
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_frames = n_frames
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.suffixes = suffixes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X_mel = np.empty((self.batch_size, *self.dim1, self.n_channels))
        X_jittershimmer = np.empty((self.batch_size, *self.dim2, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample

            # get mel sample
            with open(os.getenv("SOUND_FILE_PATH") + ID + self.suffixes[0], 'rb') as f:
                X_mel[i,], index1 = functions.get_vector(np.transpose(pickle.load(f)), self.n_frames, ID)
            #X[i,] = np.load()

            # get jitter & shimmer samples
            with open(os.getenv("SOUND_FILE_PATH") + ID + self.suffixes[1], 'r') as f:
                lines = f.readlines()
                r = []
                for x in lines:
                    # remove all \t and \n from a line
                    x = x.rstrip().split('\t')
                    r.append(x[:9]) # append the 9 first features (shim apq11 skipped)
                # change the numbers to float
                ar = np.array(r).astype(np.float)
                # also provide the index1 from mel, so the data points match 
                # print(ar.shape, ID)
                X_jittershimmer[i,], index2 = functions.get_vector(ar, self.n_frames, ID, index1)
            
            # Store class
            y[i] = self.labels[ID]
            X = [X_mel, X_jittershimmer]
            # X = X_mel
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)