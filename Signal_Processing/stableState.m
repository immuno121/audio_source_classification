%put the file in the same directory as the script
file = 'GoogleTextJohnNoiseReduced.mp3';

%y is samples and Fs is sample rates
[y, Fs] = audioread(file);

%plays the audio
% sound(y,Fs);

%disp(Fs);
%takes in the last channel of the recording
y=y(:,1);
%assigns the number of samples
L = size(y);
%disp(size(y));
% steps from zero to last sample, and divides by sampling freq
t = (0:L-1)/Fs;
%information about the audio file
info = audioinfo(file);

%disp(info.Duration);
%extracts time in seconds frm attributes of audio file
audioTime = info.Duration;

%plot(t(1:Fs*3), y(1:Fs*3));
% creats an FFT of audio file
speedyFourier = fft(y);
%disp(speedyFourier);
%creates absolute values for the sinulsoidal wave
doubleSided = abs(speedyFourier/(audioTime*Fs));
%makes it singlesided by removing negative values
singleSided = doubleSided(1:(audioTime*Fs)/2+1);
% superimposes the flipped up waves
singleSided(2:end-1) =2*singleSided(2:end-1);
% WTF (We''l figure it out
domainFrequency = Fs*(0:((audioTime*Fs)/2))/(audioTime*Fs);

ydb = mag2db(singleSided);
disp(domainFrequency);
subplot(2,1,1);
plot(domainFrequency,ydb);
title('singles side');
xlabel('freq');
ylabel('magnitute by freqs');
xlim([20 250])

Pyy = singleSided.*conj(singleSided)/(audioTime*Fs);
%plot(domainFrequency,Pyy(1:((audioTime*Fs)/2)+1));
%title('power spectral density');
%xlabel('Frequency (Hz)');
%ylabel('something by freqs');
%xlim([20 250]);
intergrated = trapz(domainFrequency(20*audioTime:250*audioTime),Pyy(20*audioTime:250*audioTime));
%disp(intergrated);
y_normalized=Pyy(20*audioTime:250*audioTime)./intergrated;
%disp(trapz(domainFrequency,y_normalized));
%disp(domainFrequency(1:100*audioTime));
%disp(y_normalized(1:1000));
subplot(2,1,2);
plot(domainFrequency(20*audioTime:250*audioTime),y_normalized);
title('normalized power spectral density');
xlabel('Frequency (Hz)');
ylabel('something by freqs');
%xlim([20 250]);
%disp(domainFrequency(997)-domainFrequency(996));
%disp(1/(audioTime*Fs));

%X = 0:0.3265:150;
%disp(trapz(domainFrequency(1:1000), y_normalized(1:1000)));
%disp(1/audioTime);
%freq* 1/audiotime)
%cutoff = 100*audiotime
%lowerBandFreq = 20*audioTime;
%upperBandFreq = 80*audioTime;
%subBassRegion = (trapz(domainFrequency(lowerBandFreq:upperBandFreq), y_normalized(lowerBandFreq:upperBandFreq)));
%disp(subBassRegion);
%upperBandFreqEval = 250*audioTime;
%totalEvaluatedRegion = (trapz(domainFrequency(lowerBandFreq:upperBandFreqEval), y_normalized(lowerBandFreq:upperBandFreqEval)));
%disp(totalEvaluatedRegion);
%energyBalanceMetric = subBassRegion/totalEvaluatedRegion;
%disp(energyBalanceMetric);

from = 1;

count = 0;

%for x = .1*Fs:.1*Fs+1:audioTime*Fs

numWindows = fix(audioTime/.1);

disp('numW: ')
disp(numWindows);
EBMs = zeros(1,numWindows);

%EBMs = zeros(1,2);


for x = .1*Fs:.1*Fs+1:audioTime*Fs
    
    currSamples = y(from:x);
    %disp("from: " + from + " to " + x);
    %disp(x-from);
    
    currFFT = fft(currSamples);
    currDoubleSided = abs(currFFT/(0.1*Fs));
    
    currSingleSided = currDoubleSided(1:(0.1*Fs)/2+1);
    currSingleSided(2:end-1) = 2*currSingleSided(2:end-1);
    
    currDomainFrequency = Fs*(0:((0.1*Fs)/2))/(0.1*Fs);
    
    currYdB = mag2db(currSingleSided);
    
    
    
    %disp(currDomainFrequency(20*.1+1:250*.1+1));
    %{
    plot(currDomainFrequency,currYdB);
    title('singles side');
    xlabel('freq');
    ylabel('dB');
    xlim([20 250]);
    %}
    currPyy = currSingleSided.*conj(currSingleSided)/(.1*Fs);
    
    %plot(currDomainFrequency,currPyy(1:((.1*Fs)/2)+1));
    %title('power spectral density');
    %xlim([20 250]);
    
    %disp(size(currDomainFrequency(20*.1+1:250*.1+1)));
    %disp(size(currPyy(20*.1+1:250*.1+1)));
    
    
    %currIntegrate = trapz(currDomainFrequency, currPyy(1:((.1*Fs)/2+1)));
    currIntegrate = trapz(currDomainFrequency(20*.1+1:250*.1+1), currPyy(20*.1+1:250*.1+1));
    %disp(currIntegrate);
    
    %currNorm = currPyy(1:((.1*Fs)/2+1))./currIntegrate;
    currNorm = currPyy(20*.1+1:250*.1+1)./currIntegrate;
    %disp('here');
    %disp(trapz(currDomainFrequency(20*.1+1:250*.1+1), currNorm));
    
    %plot(currDomainFrequency(20*.1+1:250*.1+1),currNorm);
    %xlim([20 250]);
    
    
    
    
    
    %disp(size(currNorm));
    %disp(currNorm);
    
    upperSBR = 100;
    upperTER = 250;
    
    
    lowerBand = 20*.1+1;
    upperBand = upperSBR*.1+1;
    upperBound = upperTER*.1+1;
    
    currSBR = trapz(currDomainFrequency(lowerBand:upperBand),currNorm((20-10)/10:(upperSBR-10)/10));
    %disp("currSBR: "+ currSBR);
    
    currTER = trapz(currDomainFrequency(lowerBand:upperBound),currNorm((20-10)/10:(upperTER-10)/10));
    %disp("currTER: "+ currTER);
    
    currEBM = currSBR/currTER;
    %disp("currEBM: " + currEBM);
 
    %EBMs = [EBMs currEBM];
    count = count+1;
    
    EBMs(count) = currEBM;
    from = x+1;
    %disp('all values: ' + EBMs);
    
end

%disp('all values: ');
%disp(EBMs);

[nonSkewed, outliers] = rmoutliers(EBMs, 'mean');

%disp('outliers: ');
%disp(outliers);


M = median(nonSkewed);

disp("EBM: "+ M);

%disp(audioTime);
%disp(count);

%disp(domainFrequency(200*audioTime:250*audioTime));
