clear 
close all;

%% 参数
data_time_path = 'D:\\mmwave_data\\';
adc_path = 'D:\\mmwave_data\\';
person = 'a';
value_capture = 1;

%% Connect
RSTD_DLL_Path = 'D:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\Clients\RtttNetClientController\RtttNetClientAPI.dll';
ErrStatus = Init_RSTD_Connection(RSTD_DLL_Path);

if (ErrStatus ~= 30000)
    disp('Error inside Init_RSTD_Connection');
    return;
else
    disp('Connect successfully');
end

%% Capturewhile
fid = fopen([data_time_path 'data_time.txt'],'a');
fprintf(fid, ['\nStart-> ' datestr(now,'mmmm dd, yyyy HH:MM:SS.FFF') '\n']);

while(1) % 采集数量  
    file_name = [person '_' num2str(value_capture)]; 
    adc_file = [adc_path file_name '.bin'];

    % start record the ADC data from RF capture card
    Lua_path_config = sprintf('ar1.CaptureCardConfig_StartRecord("%s", 1)', adc_file);
    RtttNetClientAPI.RtttNetClient.SendCommand(Lua_path_config);
    RtttNetClientAPI.RtttNetClient.SendCommand('RSTD.Sleep(1000)'); %miliseconds
    
    RtttNetClientAPI.RtttNetClient.SendCommand('ar1.StartFrame()');
    fprintf(fid, [file_name '-> ' datestr(now,'mmmm dd, yyyy HH:MM:SS.FFF') '\n']);
    
    % pause时间太短数据传输不完会出错
    % 5s pause(15)
    % 10s pause(25)
    pause(26);
    % next file name
    value_capture = value_capture + 1;
end

% fclose all