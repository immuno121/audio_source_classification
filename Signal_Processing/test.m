%put the file in the same directory as the script
file = 'hellotest.wav';

%y is samples and Fs is sample rates
[y, Fs] = audioread(file);

%plays the audio
sound(y,Fs);

%disp(Fs);

%information about the audio file
info = audioinfo(file);

%disp(info.Duration);
audioTime = info.Duration;

%disp(audioTime);

%t = 1:.1*Fs-1:audioTime*Fs;

%disp(t);

%disp(44100*0.1);
%disp("size: " + size(y));
%oneWindow = y(1:4410);


%disp(y(1:2));
%disp(y(1));
from = 1;

%for x = .1*Fs:.1*Fs:audioTime*Fs
for x = .1*Fs:.1*Fs:4410
    currSamples = y(from:x);
    disp("from: " + from + " to " + x);
    
    %disp(currSamples);
    
    currFFT = fft(currSamples,4096);
    
    f = Fs/4096*(0:2048);
    
    plot(f(20:250), abs(currFFT(20:250)));
    title("current FFT");
    xlabel("Frequency (Hz)");
        
    
    
    
    from = x+1;
    
    
    %disp(x);
    %disp(currSamples);
end




%X = fft(y, 4096);

%f = Fs/4096*(0:2048);

%plot(f,X)

%data_fft = fft(y);
%plot(abs(data_fft(20:250)));
%title('1khz-fft');