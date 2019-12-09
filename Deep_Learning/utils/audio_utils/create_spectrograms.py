#!/usr/bin/env python

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse

#Reference: https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0

def create_log_spectrogram(audio, hop_length, n_fft, sampling_rate):
	D = np.abs(librosa.stft(audio, n_fft=n_fft,  hop_length=hop_length))
	DB = librosa.amplitude_to_db(D, ref=np.max)
	librosa.display.specshow(DB, sr=sampling_rate, hop_length=hop_length, 
                         	x_axis='time', y_axis='log')


def create_mel_spectrogram(audio, hop_length, n_fft, sampling_rate, n_mels):
	mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
	S = librosa.feature.melspectrogram(audio, sr=sampling_rate, n_fft=n_fft, 
                                   		hop_length=hop_length, 
                                   		n_mels=n_mels)
	S_DB = librosa.power_to_db(S, ref=np.max)
	librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, 
                         	x_axis='time', y_axis='mel')


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--audio_type', help='natural or electronic audio', 
										default='Natural')
	parser.add_argument('--nfft', help='number of fft points', 
									default = 2048)
	parser.add_argument('--hop_length', help='number of fft points to skip',
										default=512)
	parser.add_argument('--n_mels', help='number of mels to consider for mel spectrogram',
									default=128)
	args = parser.parse_args()
	count = 1
	fig,ax = plt.subplots(1)
	fig.subplots_adjust(left=0,right=10,bottom=0,top=10)
	ax.axis('tight')
	ax.axis('off')
	directory = os.path.join('../../../Log_Spectrogram', args.audio_type)
	if not os.path.exists(directory):
		os.makedirs(directory)
	for filename in sorted(glob.glob(os.path.join('../../../Raw_Audio_Data/', args.audio_type, '*.wav'))):
		#TODO: save files in the same order as the audio files: currently it is reading as 1, 10, 11,..., 2
		print(filename)
		audio_sample, sr = librosa.load(filename, sr=None)
		print('hop_length = ', args.hop_length)
		print('n_fft = ', args.nfft)
		print('audio_type = ', args.audio_type)
		create_log_spectrogram(audio_sample, args.hop_length, args.nfft, sr)
		#create_mel_spectrogram(audio_sample, args.hop_length, args.nfft, sr, args.n_mels)
		filename = args.audio_type+'_'+ str(count)+'.png'
		image = os.path.join(directory, filename)
		print(image)
		fig.savefig(image, bbox_inches='tight', transparent=False, pad_inches=0.0, frameon=True)
		count = count + 1