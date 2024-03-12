function [y] = midFilter(x,fs)
win = floor(fs * 0.8);
if mod(win,2) == 0
    win = win + 1;
end
y = medfilt1(x,win);
len = length(y);
d = (win-1)/2;
y(1:d) = zeros(1,d);
y(len-d+1:len) = zeros(1,d);
end

