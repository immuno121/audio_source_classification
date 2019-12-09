#!/usr/bin/env python

from scipy.io.wavfile import write
from playsound import playsound
from threading import Thread

import sounddevice as sd
import winsound

import time
import argparse
import numpy as np
import os


def record_audio_natural(length, ID):
	fs = 44100  # Sample rate
	seconds = length  # Duration of recording
	print("Speak Now!!")
	myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
	sd.wait()  # Wait until recording is finished
	directory = '../../../Recording/Natural/'
	if not os.path.exists(directory):
		os.makedirs(directory)
		print("making directory")
	audio_file_name = os.path.join(directory, 'Natural_'+ str(ID)+'.wav')
	write(audio_file_name, fs, myrecording)  # Save as WAV file 
	print("Done Recording! Re-recording the same audio file now!")

def playback_audio(ID):
	directory = '../../../Recording/Natural/'
	audio_to_play = os.path.join(directory, 'Natural_'+str(ID)+'.wav')
	print("Playing the audio we just recorded")
	#### For windows, use winsound.PlaySound(sound_file, flag). playsound doesn't work
	
	'''
	from here: https://stackoverflow.com/questions/25357054/error-when-i-try-to-play-a-sound/25357106
	while flags is a bitwise-OR'd combination of winsound.SND_FILENAME
	parameter is the path to a .wav file), winsound.SND_ALIAS
	(the sound parameter is a name for a builtin Windows sound, see the docs), 
	winsound.SND_LOOP (play the sound in a loop), 
	winsound.SND_MEMORY (the sound parameter is a memory image of a .wav file), 
	winsound.SND_PURGE (stop all playing instances of the specified sound, not supported on modern Windows), 
	winsound.SND_ASYNC (return immediately, allowing sounds to play asynchronously),
	winsound.SND_NODEFAULT (do not play default sound if the sound cannot be found), 
	winsound.SND_NOSTOP (do not interrupt other sounds currently playing) 
	and winsound.SND_NOWAIT (return immediately if the sound driver is busy)	
	'''

	winsound.PlaySound(audio_to_play, winsound.SND_FILENAME)
	####playsound(audio_to_play) # uncomment this for Linux
	


	##################################################
	time.sleep(1)

def record_audio_electronic(length, ID):
	fs = 44100  # Sample rate
	seconds = length  # Duration of recording
	print("Rerecoding natural audio!!!!")
	myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
	sd.wait()  # Wait until recording is finished
	directory = '../../../Recording/Electronic/'
	if not os.path.exists(directory):
		os.makedirs(directory)
		print("making directory")
	audio_file_name = os.path.join(directory, 'Electronic_'+ str(ID)+'.wav')
	write(audio_file_name, fs, myrecording)  # Save as WAV file 
	print("Done Recording! Please remember to update the ID for the next recording!!!")
	time.sleep(1)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--seconds', help='length of the audio', 
										default = 5)
	parser.add_argument('--ID', help='ID of the recording', 
									default = 1)
	args = parser.parse_args()

	record_audio_natural(args.seconds, args.ID)
	
	t1 = Thread(target = playback_audio, args = [args.ID])
	t2 = Thread(target = record_audio_electronic, args = [args.seconds, args.ID])

	t1.start()
	t2.start()