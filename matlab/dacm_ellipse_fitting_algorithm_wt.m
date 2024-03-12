%��д��ȡ�й��񶯵���λ�ź�
%77g�״�����֤��Ԥ������ȡ�����ź�
%
clear 
close all;
%% setting 1
B=3871.53e6;       %��Ƶ����
K=43.017e12;       %��Ƶб��
Tc=90e-6;     %chirp������
fs=6e6;       %������
numsample=470; %��������/����
numframe=400;
TFrame=0.025; % Frame���� 25ms
numchirp=255;  %ÿ֡������

%% other setting
c=3.0e8; 
f0=77e9;       %��ʼƵ��
lambda=c/f0;   %�״��źŲ���
d=lambda/2;    %�������м��
T=B/K;         %����ʱ��
NFFT=2^nextpow2(numsample);  %������FFT����
M=2^nextpow2(numchirp); %��������FFT����
n_RX=1;        %RX����ͨ����
% n_RX=4;        %RX����ͨ����

duration = numframe*TFrame; % ��ʱ��
Fs = numchirp/TFrame; % ��Ч������
Fupper = Fs/2;

oo = 40;  % ��λ������ 10200/oo
Rangedf = c/2/B;           %����ֱ��� ��λ�� c/2B 
lim_num = ceil(1/Rangedf); %Ѱ���˵��Ͻ�����

%% ���ݼ���
Path = 'D:\ZJU\good_data\';
savePath = '.\Heartdata\';

File = dir(fullfile(Path,'4_106_Raw_0.bin'));  % ��ʾ�ļ��������з��Ϻ�׺��Ϊ_Raw_0.bin�ļ���������Ϣ
FileNames = {File.name}';            % ��ȡ���Ϻ�׺��Ϊ_Raw_0.bin�������ļ����ļ�����ת��Ϊn��1��
[~,ind] = natsort(FileNames);
FileNames = FileNames(ind);
Length_Names = size(FileNames,1);    % ��ȡ����ȡ�����ļ��ĸ���

for z = 1 : Length_Names    
    bin_pth = strcat(Path, FileNames{z,1});   
    raw_data = readDCA1000(bin_pth,numsample);
    
    %ȡ����һ���״������
    %numsample�У�numchirp*numframe
    Rxdata = [];
    for antenna = 1:n_RX
        adc_data_antenna = raw_data(antenna,:);
        adc_data_antenna = reshape(adc_data_antenna,[numsample,numchirp*numframe]);
        Rxdata(:,:) = adc_data_antenna;
    end

    %% �ָ�����
    Rxdata_real=real(Rxdata); Rxdata_real(find(Rxdata_real==0))=nan; Rxdata_real=fillmissing(Rxdata_real,'nearest',2);
    Rxdata_imag=imag(Rxdata); Rxdata_imag(find(Rxdata_imag==0))=nan; Rxdata_imag=fillmissing(Rxdata_imag,'nearest',2);
    Rxdata = complex(Rxdata_real, Rxdata_imag);
    
    %% rangefft
    win = hamming(numsample);  %������
    all_profile = zeros(NFFT, numchirp*numframe);
    %�����е�chirp��range fft������λ��
    for i = 1:1*numchirp*numframe
        %��ÿһ��chirp����range-FFT
        temp = Rxdata(:,i).*win;
        temp_fft = fft(temp,NFFT);
        all_profile(:,i) = temp_fft;
    end

    %range-bin tracking �ҳ���������
    absdata = abs(all_profile);
    energydata = sum(absdata,2);  %��ÿ�зֱ����
    %ÿһ�ж���һ��chirp��range-FFT���
    %���ʸ߾ͱ�ʾ������һ���壬����Ŀ��

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
    %��߹������ڵ�����
    disp(max_num) 

    %% �����Բ����㷨 ����Բ�� ����ƫ��
    m_temp = all_profile(max_num,:);%�õ�һ��
    
    %������I/Qͨ��
    xdata = double(real(m_temp));
    ydata = double(imag(m_temp));
    [center_x, center_y] = Ellipse_Fitting(xdata, ydata);

    disp(center_x);
    disp(center_y);

    reg_xdata = xdata-center_x;
    reg_ydata = ydata-center_y;
    %���µõ����� 
    %����ֱ��ƫ��
    reg_temp = complex(reg_xdata, reg_ydata);

    %% DACM ������λ��
    %     %------------------ ��ͳ����������λ��--------------------
    %     %������λ��
    %     angledata = atan2(imag(all_profile), real(all_profile));
    %     anglemax = angledata(max_num,1:numchirp:end);
    %     %��λ���
    %     anglemax = unwrap(anglemax);
    %     %------------------ ��ͳ����������λ��--------------------
    %ÿһ֡ȡ��һ������ʴ���range-FFT���
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
    
    %% ------------------sgolay�˲���---------------
    %DACM��ȡ�����λ����ȥ���˲�������һ�����зֽ������
    order = 2;
    framelen = 11;
    phi_1 = sgolayfilt(phi, order, framelen);  %�˲�����
    phi = phi_1;

    %% ������λ��
    diff_phase = diff(phi);
    base_0 = midFilter(diff_phase,length(phi)/10);
    dd = diff_phase - base_0;
    
    %% ������׵���
    h = 10 * oo / numchirp / numframe;  %time interval
    phi_2 = zeros(1,Phase_length);
    for n = 4:Phase_length-3
        phi_2(n) = 4*phi(n) + (phi(n+1)+phi(n-1)) - 2*(phi(n+2)+phi(n-2)) - (phi(n+3)+phi(n-3));
        phi_2(n) = phi_2(n) / (16*h*h);
    end

    %% ����Ư��
    base_1 = midFilter(phi_2,length(phi)/10);
    pp = phi_2 - base_1;

end