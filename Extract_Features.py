import librosa
import pickle
import glob
import pickle
import math
import os.path

total = 148642
print('script started...')
count = 0
for filename in glob.iglob('E:/dippa/voxceleb1/wav/**/*/*.wav',recursive=True):
    if not os.path.isfile(filename[:-3] + 'mel'):
        count+=1
        if (count % 1000 == 0):
            print(count/total, ' processed.')
        duration = int(librosa.get_duration(filename=filename))
        y, sr = librosa.load(filename, sr=None)
        data = librosa.feature.melspectrogram(y=y[:(sr*duration-1)], sr=sr, n_fft=int(sr*0.03), hop_length=int(0.01*sr))
        pickle_file = open(filename[:-3] + 'mel', 'wb')
        pickle.dump(data, pickle_file)
        pickle_file.close()
    else:
        count +=1
        if (count % 1000 == 0):
            print(count/total, ' processed.')

print('finished')
