import numpy as np
import scipy.io
from PIL import Image 
import cv2
import os,os.path,shutil
import re

from scipy.interpolate import interp1d

##save file size（3,60,25)
timestep_size=60


def find_martrix_min_value(data_matrix):
    '''
    功能：找到矩阵最小值
    '''
    new_data=[]
    for i in range(len(data_matrix)):
        new_data.append(min(data_matrix[i]))
    print ('data_matrix 最小值为：', min(new_data))
    return min(new_data)


def find_martrix_max_value(data_matrix):
    '''
    功能：找到矩阵最大值
    '''
    new_data=[]
    for i in range(len(data_matrix)):
        new_data.append(max(data_matrix[i]))
    print ('data_matrix 最大值为：', max(new_data))
    return  max(new_data)
#get whole joints place
#transfor .mat(all joints) into wanted point and reference(.npz)
def mtop( filename,savepath):
	point= scipy.io.loadmat(filename)  # 读取mat文件
	#point=np.load("whole1.npz")
	wx=point['x']##whole joints point
	wy=point['y']
	wz=point['z']
	w=np.vstack((wx,wy,wz)).reshape(3,-1,25) #left arm, right arm,torso, left leg, right leg
	center=w[:,:,0]
	center=center.repeat(25)
	center=center.reshape(3,-1,25)
	#print(center)
	w=w-center
	if w.shape[1]>60 :
		file_new=filename[filename.find('S'):filename.find('.mat')]
		#print(file_new)
		np.save(savepath+file_new,w)

def eachFile(folder):
    allFile = os.listdir(folder)
    fileNames = []
    for file in allFile:
        fullPath = os.path.join(folder, file)
        fileNames.append(fullPath)
    return fileNames


# main part
for i in range(60,61):
	srcFolder='./mat_f/'+str(i)
	savepath='./CV_40/'
	fileNames =eachFile(srcFolder)
	for fileName in fileNames:
		print(fileName)
		#print(int(fileName.find('C')))
		if(int(fileName[fileName.find('C')+1:fileName.find('C')+4])==1):
			savepath='./CV_40/test/'
		else:
			savepath='./CV_40/train/'
		mtop(fileName,savepath)

srcFolder='./CV/train'
fileNames =eachFile(srcFolder)
trainpath='./CS/train/'
testpath='./CS/test/'
for fileName in fileNames:
	subject=int(fileName[fileName.find('S')+1:fileName.find('S')+4])
	a=[1, 2, 4, 5, 8, 9, 13, 14, 15,
	16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
	if subject in a:
		newname=trainpath+fileName[fileName.find('S'):]

	else:
		newname=testpath+fileName[fileName.find('S'):]
	shutil.copyfile(fileName,newname)





