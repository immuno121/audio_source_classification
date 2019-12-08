#!/usr/bin/env python

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import skimage
import glob
import os
import soundfile as sf


directory = '../../../Original_Data/Electronic/'
print(directory)

count = 1

save_path = '../../../Raw_Audio_Data/Electronic'
new_sr = 44100



for filename in sorted(glob.glob(os.path.join(directory,'*.mp3'))):
		#print(filename)
		audio_sample, sr = librosa.load(filename, sr=None)
		resampled_audio = librosa.core.resample(audio_sample, sr, new_sr)
		trimmed_audio = librosa.util.fix_length(resampled_audio, 5*new_sr)
		print(trimmed_audio.shape)
		file_to_write = os.path.join(save_path,'Electronic_'+str(count)+'.wav')
		sf.write(file_to_write, trimmed_audio, new_sr)
		count = count+1


		#print('done')

print('end')
