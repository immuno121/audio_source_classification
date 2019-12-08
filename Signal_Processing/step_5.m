%put the file in the same directory as the script
file = 'hellotest.wav';

%y is samples and Fs is sample rates
[y, Fs] = audioread(file);

%plays the audio
sound(y,Fs);

%disp(Fs);

y=y(:,1);
L = size(y);
%disp(size(y));
t = (0:L-1)/Fs;
%information about the audio file
info = audioinfo(file);

%disp(info.Duration);
audioTime = info.Duration;

%disp(audioTime*Fs);

%t = 1:.1*Fs-1:audioTime*Fs;

%disp(t);

%disp(44100*0.1);
%disp("size: " + size(y));
%oneWindow = y(1:4410);


%disp(y(1:2));
%disp(y(1));
from = 1;

%for x = .1*Fs:.1*Fs:audioTime*Fs




%X = fft(y, 4096);

%f = Fs/4096*(0:2048);

%plot(f,X)

%data_fft = fft(y);
%plot(abs(data_fft(20:250)));
%title('1khz-fft');
plot(t(1:Fs*3), y(1:Fs*3));
speedyFourier = fft(y);
%disp(speedyFourier);
doubleSided = abs(speedyFourier/(audioTime*Fs));
singleSided = doubleSided(1:(audioTime*Fs)/2+1);
singleSided(2:end-1) =2*singleSided(2:end-1);
domainFrequency = Fs*(0:((audioTime*Fs)/2))/(audioTime*Fs);
ydb = mag2db(singleSided);
%disp(domainFrequency);
%plot(domainFrequency,ydb);
%title('singles side');
%xlabel('freq');
%ylabel('magnitute by freqs');

Pyy = singleSided.*conj(singleSided)/(audioTime*Fs);
plot(domainFrequency,Pyy(1:((audioTime*Fs)/2)+1));
title('power spectral density');
xlabel('Frequency (Hz)');
ylabel('something by freqs');
xlim([20 250]);
intergrated = trapz(domainFrequency,Pyy(1:audioTime*Fs/2+1));
disp(intergrated);
y_normalized=Pyy(1:((audioTime*Fs)/2)+1)./intergrated;
%disp(trapz(domainFrequency,y_normalized));
disp(domainFrequency(1:100*audioTime));
%disp(y_normalized(1:1000));
plot(domainFrequency,y_normalized);
title('normalized power spectral density');
xlabel('Frequency (Hz)');
ylabel('something by freqs');
xlim([20 1050]);
disp(domainFrequency(997)-domainFrequency(996));
disp(1/(audioTime*Fs));

%X = 0:0.3265:150;
disp(trapz(domainFrequency(1:1000), y_normalized(1:1000)));
disp(1/audioTime);
%freq* 1/audiotime)
%cutoff = 100*audiotime
lowerBandFreq = 20*audioTime;
upperBandFreq = 80*audioTime;
subBassRegion = (trapz(domainFrequency(lowerBandFreq:upperBandFreq), y_normalized(lowerBandFreq:upperBandFreq)));
disp(subBassRegion);
upperBandFreqEval = 250*audioTime;
totalEvaluatedRegion = (trapz(domainFrequency(lowerBandFreq:upperBandFreqEval), y_normalized(lowerBandFreq:upperBandFreqEval)));
disp(totalEvaluatedRegion);
energyBalanceMetric = subBassRegion/totalEvaluatedRegion;
disp(energyBalanceMetric);

