%编写提取有关振动的相位信号
%77g雷达：身份认证的预处理，提取心跳信号
%
clear 
close all;
%% setting 1
B=3871.53e6;       %调频带宽
K=43.017e12;       %调频斜率
Tc=90e-6;     %chirp总周期
fs=6e6;       %采样率
numsample=470; %采样点数/脉冲
numframe=400;
TFrame=0.025; % Frame周期 25ms
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

oo = 40;  % 相位采样率 10200/oo
Rangedf = c/2/B;           %距离分辨率 单位米 c/2B 
lim_num = ceil(1/Rangedf); %寻找人的上界索引

%% 数据加载
Path = 'D:\ZJU\good_data\';
savePath = '.\Heartdata\';

File = dir(fullfile(Path,'4_106_Raw_0.bin'));  % 显示文件夹下所有符合后缀名为_Raw_0.bin文件的完整信息
FileNames = {File.name}';            % 提取符合后缀名为_Raw_0.bin的所有文件的文件名，转换为n行1列
[~,ind] = natsort(FileNames);
FileNames = FileNames(ind);
Length_Names = size(FileNames,1);    % 获取所提取数据文件的个数

for z = 1 : Length_Names    
    bin_pth = strcat(Path, FileNames{z,1});   
    raw_data = readDCA1000(bin_pth,numsample);
    
    %取出第一个雷达的数据
    %numsample行，numchirp*numframe
    Rxdata = [];
    for antenna = 1:n_RX
        adc_data_antenna = raw_data(antenna,:);
        adc_data_antenna = reshape(adc_data_antenna,[numsample,numchirp*numframe]);
        Rxdata(:,:) = adc_data_antenna;
    end

    %% 恢复数据
    Rxdata_real=real(Rxdata); Rxdata_real(find(Rxdata_real==0))=nan; Rxdata_real=fillmissing(Rxdata_real,'nearest',2);
    Rxdata_imag=imag(Rxdata); Rxdata_imag(find(Rxdata_imag==0))=nan; Rxdata_imag=fillmissing(Rxdata_imag,'nearest',2);
    Rxdata = complex(Rxdata_real, Rxdata_imag);
    
    %% rangefft
    win = hamming(numsample);  %汉明窗
    all_profile = zeros(NFFT, numchirp*numframe);
    %对所有的chirp做range fft，求相位角
    for i = 1:1*numchirp*numframe
        %对每一个chirp进行range-FFT
        temp = Rxdata(:,i).*win;
        temp_fft = fft(temp,NFFT);
        all_profile(:,i) = temp_fft;
    end

    %range-bin tracking 找出能量最大的
    absdata = abs(all_profile);
    energydata = sum(absdata,2);  %对每行分别求和
    %每一列都是一个chirp的range-FFT结果
    %功率高就表示这里是一个峰，存在目标

%     figure()
%     pcolor(absdata);
%     shading interp; 
%     colorbar; colormap(jet);
%     xlabel('Time Samples');ylabel('FFT Samples');

%     figure()
%     pcolor(absdata(1:30,:));
%     shading interp; 
%     colorbar; colormap(jet);
%     xlabel('Time Samples');ylabel('FFT Samples');
    
    [~,max_num] = max(energydata(1:lim_num));
    %最高功率所在的索引
    disp(max_num) 

    %% 最简单椭圆拟合算法 修正圆心 补偿偏移
    m_temp = all_profile(max_num,:);%得到一行
    
    %极坐标I/Q通道
    xdata = double(real(m_temp));
    ydata = double(imag(m_temp));
    [center_x, center_y] = Ellipse_Fitting(xdata, ydata);

    disp(center_x);
    disp(center_y);

    reg_xdata = xdata-center_x;
    reg_ydata = ydata-center_y;
    %重新得到序列 
    %修正直流偏置
    reg_temp = complex(reg_xdata, reg_ydata);

    %% DACM 计算相位角
    %     %------------------ 传统方法计算相位角--------------------
    %     %计算相位角
    %     angledata = atan2(imag(all_profile), real(all_profile));
    %     anglemax = angledata(max_num,1:numchirp:end);
    %     %相位解缠
    %     anglemax = unwrap(anglemax);
    %     %------------------ 传统方法计算相位角--------------------
    %每一帧取出一个最大功率处的range-FFT结果
    angledata = reg_temp(1:oo:end);
    Phase_length = length(angledata);
    Q = imag(angledata);%Q
    I = real(angledata);%I
    phi = zeros(1, Phase_length);
    for n = 2:Phase_length
        phi(n) = 0;
        for m =2:n
            phi(n) = phi(n) + (I(m)*(Q(m) - Q(m-1))-Q(m)*(I(m)-I(m-1)))/(I(m).^2 + Q(m).^2);
        end
    end
    
    %% ------------------sgolay滤波器---------------
    %DACM提取后的相位，先去除滤波器再跑一边所有分解类代码
    order = 2;
    framelen = 11;
    phi_1 = sgolayfilt(phi, order, framelen);  %滤波后结果
    phi = phi_1;

    %% 计算相位差
    diff_phase = diff(phi);
    base_0 = midFilter(diff_phase,length(phi)/10);
    dd = diff_phase - base_0;
    
    %% 计算二阶导数
    h = 10 * oo / numchirp / numframe;  %time interval
    phi_2 = zeros(1,Phase_length);
    for n = 4:Phase_length-3
        phi_2(n) = 4*phi(n) + (phi(n+1)+phi(n-1)) - 2*(phi(n+2)+phi(n-2)) - (phi(n+3)+phi(n-3));
        phi_2(n) = phi_2(n) / (16*h*h);
    end

    %% 基线漂移
    base_1 = midFilter(phi_2,length(phi)/10);
    pp = phi_2 - base_1;

end