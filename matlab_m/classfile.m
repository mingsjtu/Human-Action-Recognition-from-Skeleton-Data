%% move_img  
% Summary of example objective  
clear;clc;
% DST_PATH_t = [ 'D:\科研\ntuRGB\mat_f\'];%目的文件目录
for i= 21:22
SOURCE_PATH_t =[ 'D:\科研\ntuRGB\mat_f\',num2str(i),'\'];%源文件目录  
DST_PATH_t1 = [ 'D:\科研\ntuRGB\mat_f\',num2str(i),'\test'];%目的文件目录  
DST_PATH_t2 = [ 'D:\科研\ntuRGB\mat_f\',num2str(i),'\train'];%目的文件目录  
dirOutput=dir(fullfile(SOURCE_PATH_t,'*.mat'));%如果存在不同类型的文件，用‘*’读取所有，如果读取特定类型文件，'.'加上文件类型，例如用‘.jpg’
fileNames={dirOutput.name};
LengthFiles = length(fileNames)
for n = 1:LengthFiles
fileName=char(fileNames(n))
fileName(8)
if(fileName(8)=='1')
movefile([SOURCE_PATH_t,fileName],DST_PATH_t1);%%save to test
else
movefile([SOURCE_PATH_t,fileName],DST_PATH_t2);%%save to train
end
end
end
% for i=21:22
% 
% new_folder =[DST_PATH_t,num2str(i),'\train']; % new_folder 保存要创建的文件夹，是绝对路径+文件夹名称
% mkdir(new_folder);  % mkdir()函数创建文件夹
% new_folder =[DST_PATH_t,num2str(i),'\test'];
% mkdir(new_folder);  % mkdir()函数创建文件夹
% if(fileName(18:20)==num2str(i,'%03d'))
% movefile(['C:\Users\aygsg\Downloads\nturgbd_skeletons\nturgb+d_skeletons\',fileName],new_folder); 
% end
% end

%
