clear;clc;
for t=60:61
fileFolder=['D:\research\ntuRGB\ske_f\',num2str(t),'\'];
dirOutput=dir(fullfile(fileFolder,'*.skeleton'));
savepath=['D:\research\ntuRGB\mat_f\',num2str(t),'\'];
fileNames={dirOutput.name};
LengthFiles = length(fileNames)
for n = 864:LengthFiles;
% if(exist([savepath,num2str(n),'-points.mat'],'file') ==2)  
%     continue;
% else
fileName=char(fileNames(n))
savetomat(fileFolder,fileName,n,savepath);
end
end
