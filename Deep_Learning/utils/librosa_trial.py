#!/usr/bin/env python

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import skimage

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

#filename = '../../../Raw_Audio_Data/ElectronicUnprocessedChoppedWAV/MixedRenderedElectronic-00.wav'

filename = '/home/dg777/Documents/Yale Coursework/Semester 1/Building Interactive Machines/Project/Raw_Audio_Data/ElectronicUnprocessedChoppedWAV/MixedRenderedElectronic-02.wav'

#print(filename)
audio_sample, sr = librosa.load(filename, sr=None)
#print(y)
#print(sr)
# trim silent edges
#trimmed_audio, _ = librosa.effects.trim(y)
#librosa.display.waveplot(audio_sample, sr=sr);

#plt.show()


###################################################### Linear Scaled Spectrogram ################################################
hop_length = 512
n_fft = 2048
D = np.abs(librosa.stft(audio_sample, n_fft=n_fft,  
                        hop_length=hop_length))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear');
#plt.show()
#plt.colorbar();

##################################################### Log Scaled Spectrogram #####################################################
DB = librosa.amplitude_to_db(D, ref=np.max)
librosa.display.specshow(DB, sr=sr, hop_length=hop_length, 
                         x_axis='time', y_axis='log');
#plt.colorbar(format='%+2.0f dB');
#plt.savefig('sampleNaturalLogSpectrogram.png')

##################################################### Mel Spectrogram #############################################################
n_mels = 128
mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
S = librosa.feature.melspectrogram(audio_sample, sr=sr, n_fft=n_fft, 
                                   hop_length=hop_length, 
                                   n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
print(S_DB.shape)

#fig = plt.figure(figsize=(9, 11))
fig,ax = plt.subplots(1)

fig.subplots_adjust(left=0,right=10,bottom=0,top=10)
ax.axis('tight')
ax.axis('off')



librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, 
                         x_axis='time', y_axis='mel');



#plt.figure()
#ax = plt.axes()
#ax.set_axis_off()
#plt.set_cmap('hot')
#plt.colorbar(format='%+2.0f dB');
fig.savefig('sampleMelElectronicLogSpectrogram.png', bbox_inches='tight', transparent=True, pad_inches=0.0, frameon=True)
#plt.show()

'''
# min-max scale to fit inside 8-bit range
img = scale_minmax(S, 0, 255).astype(np.uint8)
img = np.flip(img, axis=0) # put low frequencies at the bottom in image
img = 255-img # invert. make black==more energy

# save as PNG
skimage.io.imsave('out.png', img)
'''