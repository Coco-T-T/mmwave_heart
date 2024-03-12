function data_test(bin_pth)

%% setting 1
B=3871.53e6;       %调频带宽
K=43.017e12;       %调频斜率
Tc=90e-6;     %chirp总周期
fs=6e6;       %采样率
numsample=470; %采样点数/脉冲
numframe = 400; %注意与mmWave Studio保持一致 5s->200 10s->400
TFrame = 0.025; % Frame周期 25ms
numchirp=255;  %每帧脉冲数

%% other setting
c=3.0e8; 
f0=77e9;       %初始频率
lambda=c/f0;   %雷达信号波长
d=lambda/2;    %天线阵列间距
T=B/K;         %采样时间
NFFT=2^nextpow2(numsample);  %距离向FFT点数
M=2^nextpow2(numchirp); %多普勒向FFT点数
n_RX=1;        %RX天线通道数
% n_RX=4;        %RX天线通道数

duration = numframe*TFrame; % 总时长
Fs = numchirp/TFrame; % 等效采样率
Fupper = Fs/2;

oo = 200;
Rangedf = c/2/B;           %距离分辨率 单位米 c/2B 
lim_num = ceil(1/Rangedf); %寻找人的上界索引

%% 数据加载 
raw_data = readDCA1000(bin_pth,numsample);

%取出第一个雷达的数据
%numsample行，numchirp*numframe
data = [];
for antenna = 1:n_RX
    adc_data_antenna = raw_data(antenna,:);
    adc_data_antenna = reshape(adc_data_antenna,[numsample,numchirp*numframe]);
    data(:,:) = adc_data_antenna;
end

%% rangefft
win = hamming(numsample);  %汉明窗
all_profile = zeros(NFFT, numchirp*numframe);
%对所有的chirp做range fft，求相位角
for i = 1:1*numchirp*numframe
    %对每一个chirp进行range-FFT
    temp = data(:,i).*win;
    temp_fft = fft(temp,NFFT);
    all_profile(:,i) = temp_fft;
end
%range-bin tracking 找出能量最大的
absdata = abs(all_profile);
energydata = sum(absdata,2);  %对每行分别求和
%每一列都是一个chirp的range-FFT结果
%功率高就表示这里是一个峰，存在目标

[~,max_num] = max(energydata(1:lim_num));
%最高功率所在的索引
disp(max_num)

%% DACM 计算相位角
%     %------------------ 传统方法计算相位角--------------------
%     %计算相位角
%     angledata = atan2(imag(all_profile), real(all_profile));
%     anglemax = angledata(max_num,1:numchirp:end);
%     %相位解缠
%     anglemax = unwrap(anglemax);
%     %------------------ 传统方法计算相位角--------------------
%每一帧取出一个最大功率处的range-FFT结果
angledata = all_profile(max_num,1:oo:end);
Phase_length = length(angledata);
Q = imag(angledata);%Q
I = real(angledata);%I
phi = zeros(1, Phase_length);
for n = 2:Phase_length
    phi(n) = 0;
    for m =2:n
        if I(m)==0 && Q(m)==0
            %h = warndlg(sprintf('雷达数据丢失，请检查网线连接！'),'Program wrong');
            continue;
        else
            phi(n) = phi(n) + (I(m)*(Q(m) - Q(m-1))-Q(m)*(I(m)-I(m-1)))/(I(m).^2 + Q(m).^2);
        end
    end
end

%% ------------------sgolay滤波器---------------
order = 2;
framelen = 11;
phi = sgolayfilt(phi, order, framelen);  %滤波后结果

%% 计算相位差
diff_phase = diff(phi);
figure;
plot(diff_phase);

end