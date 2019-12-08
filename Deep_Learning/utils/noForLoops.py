#!/usr/bin/env python

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import skimage
import soundfile as sf
import os
# from pydub import AudioSegment

"""
# Helper function to get files in a single directory
def __get_files(UnprocessedImmediated, mp3):

# Expand out the directory
UnprocessedImmediated = os.path.abspath(/Users/emmanueltoksadeniran/Documents/MATLAB/(dir_name))
trimmedW = set()
for sub_ext in mp3:
    globstr = os.path.join(UnprocessedImmediated, '*' + os.path.extsep + sub_ext)
    trimmedW |= set(glob.glob(globstr))
return trimmedW
"""

a = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000001.mp3'
b = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000002.mp3'
c = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000003.mp3'
d = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000004.mp3'
e = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000005.mp3'
f = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000006.mp3'
g = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000007.mp3'
h = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000008.mp3'
i = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000009.mp3'
j = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000010.mp3'
k = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000011.mp3'
l = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000012.mp3'
m = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000013.mp3'
n = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000014.mp3'
o = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000015.mp3'
p = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000016.mp3'
q = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000017.mp3'
r = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000018.mp3'
s = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000019.mp3'
t = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000020.mp3'
u = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000021.mp3'
v = '/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/000022.mp3'

a, sr = librosa.load(a)
trimmedA, _ = librosa.effects.trim(a, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedA, _ = librosa.effects.trim(a, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
b, sr = librosa.load(b)
trimmedB, _ = librosa.effects.trim(b, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedB, _ = librosa.effects.trim(b, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
c, sr = librosa.load(c)
trimmedC, _ = librosa.effects.trim(c, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedC, _ = librosa.effects.trim(c, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
d, sr = librosa.load(d)
trimmedD, _ = librosa.effects.trim(d, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedD, _ = librosa.effects.trim(d, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
e, sr = librosa.load(e)
trimmedE, _ = librosa.effects.trim(e, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedE, _ = librosa.effects.trim(e, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
f, sr = librosa.load(f)
trimmedF, _ = librosa.effects.trim(f, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedF, _ = librosa.effects.trim(f, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
g, sr = librosa.load(g)
trimmedG, _ = librosa.effects.trim(g, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedG, _ = librosa.effects.trim(g, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
h, sr = librosa.load(h)
trimmedH, _ = librosa.effects.trim(h, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedH, _ = librosa.effects.trim(h, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
i, sr = librosa.load(i)
trimmedI, _ = librosa.effects.trim(i, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedI, _ = librosa.effects.trim(i, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
j, sr = librosa.load(j)
trimmedJ, _ = librosa.effects.trim(j, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedJ, _ = librosa.effects.trim(j, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
k, sr = librosa.load(k)
trimmedK, _ = librosa.effects.trim(k, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedK, _ = librosa.effects.trim(k, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
l, sr = librosa.load(l)
trimmedL, _ = librosa.effects.trim(l, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedL, _ = librosa.effects.trim(l, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
m, sr = librosa.load(m)
trimmedM, _ = librosa.effects.trim(m, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedM, _ = librosa.effects.trim(m, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
n, sr = librosa.load(n)
trimmedN, _ = librosa.effects.trim(n, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedN, _ = librosa.effects.trim(n, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
o, sr = librosa.load(o)
trimmedO, _ = librosa.effects.trim(o, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedO, _ = librosa.effects.trim(o, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
p, sr = librosa.load(p)
trimmedP, _ = librosa.effects.trim(p, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedP, _ = librosa.effects.trim(p, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
q, sr = librosa.load(q)
trimmedQ, _ = librosa.effects.trim(q, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedQ, _ = librosa.effects.trim(q, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
r, sr = librosa.load(r)
trimmedR, _ = librosa.effects.trim(r, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedR, _ = librosa.effects.trim(r, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
s, sr = librosa.load(s)
trimmedS, _ = librosa.effects.trim(s, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedS, _ = librosa.effects.trim(s, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
t, sr = librosa.load(t)
trimmedT, _ = librosa.effects.trim(t, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedT, _ = librosa.effects.trim(t, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
u, sr = librosa.load(u)
trimmedU, _ = librosa.effects.trim(u, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedU, _ = librosa.effects.trim(u, top_db=60, ref=np.max, frame_length=2048, hop_length=512)
v, sr = librosa.load(v)
trimmedV, _ = librosa.effects.trim(v, top_db=30, ref=np.max, frame_length=2048, hop_length=512)
#trimmedV, _ = librosa.effects.trim(v, top_db=60, ref=np.max, frame_length=2048, hop_length=512)

trimmedB = np.append(trimmedA,trimmedB)
trimmedC = np.append(trimmedB,trimmedC)
trimmedD = np.append(trimmedC,trimmedD)
trimmedE = np.append(trimmedD,trimmedE)
trimmedF = np.append(trimmedE,trimmedF)
trimmedG = np.append(trimmedF,trimmedG)
trimmedH = np.append(trimmedG,trimmedH)
trimmedI = np.append(trimmedH,trimmedI)
trimmedJ = np.append(trimmedI,trimmedJ)
trimmedK = np.append(trimmedJ,trimmedK)
trimmedL = np.append(trimmedK,trimmedL)
trimmedM = np.append(trimmedL,trimmedM)
trimmedN = np.append(trimmedM,trimmedN)
trimmedO = np.append(trimmedN,trimmedO)
trimmedP = np.append(trimmedO,trimmedP)
trimmedQ = np.append(trimmedP,trimmedQ)
trimmedR = np.append(trimmedQ,trimmedR)
trimmedS = np.append(trimmedR,trimmedS)
trimmedT = np.append(trimmedS,trimmedT)
trimmedU = np.append(trimmedT,trimmedU)
trimmedV = np.append(trimmedU,trimmedV)

"""
trimmedW = np.append(trimmedA,trimmedB,trimmedC,trimmedD,trimmedE,trimmedF,trimmedG,trimmedH,trimmedI,trimmedJ,trimmedK,trimmedL,trimmedM,trimmedN,trimmedO,trimmedP,trimmedQ,trimmedR,trimmedS,trimmedT,trimmedU,trimmedV)
"""

trimmedW = librosa.output.write_wav('/Users/emmanueltoksadeniran/Documents/MATLAB/UnprocessedImmediated/concatenatedAudioFile.wav', trimmedV, sr)

ffmpeg -i concatenatedAudioFile.wav -map 0 -f segment -segment_time 5 -c copy out%03d.mp3
#ffmpeg -i somefile.mp3 -f segment -segment_time 3 -c copy out%03d.mp3

# trimmedX = librosa.output.write_wav('concatenatedFile.wav', trimmedW, sr)
#trimmedY, sr = librosa.load(trimmedX)


"""
# Let's consider anything that is 30 decibels quieter than
# the average volume of the podcast to be silence
average_loudness = podcast.rms
silence_threshold = average_loudness * db_to_float(-30)
"""

"""
    # Calculating the sum of the numbers from 1 to 100:
    reduce(lambda x, y: x+y, range(1,101))
    result = 5050
"""

# sf.write('concatenatedFile.wav', data, samplerate, subtype='PCM_24')


