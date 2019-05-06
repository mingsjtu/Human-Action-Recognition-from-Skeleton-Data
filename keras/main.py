from __future__ import division
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential,load_model
from keras.layers import Embedding, LSTM, Dense, Activation,Dropout
from keras.optimizers import SGD,Adam



train_file='CS/train'
test_file='CS/test'

num_ske = 25      
hidden_size = 100    # hidden layer num
layer_num = 4        # LSTM layer num
class_num = 60        # last class number
cell_type = "lstm"   # lstm or block_lstm
NUM_SKE=25
batch_size =128  # tf.int32, batch_size = 128
timestep_size=150
learning_rate=1e-3
epoch_size=1000
N=10

def eachFile(folder):
    allFile = os.listdir(folder)
    fileNames = []
    for file in allFile:
        
        fileNames.append(file)
    return fileNames

def getlist(filefolder):
  print(filefolder+' load start')
  fileNames=eachFile(filefolder)
  num=len(fileNames)
  #print(len(fileNames))
  listp=[[]for i in range (num)]
  i=0
  classname=[]#list can only use .append

  for fileName in fileNames:
    p=np.load(filefolder+'/'+fileName)
    #p.astype(np.float32)
    listp[i].append(p)
    classname.append(fileName[-5]) #filename example='S001C001P001R001A001.mat'
    
    #print(listp[0][0].shape) (3,103,25)
    i=i+1
  return listp,classname

def get_all(listp,classname):
  #print(type(listp[1])[0])
  imdata=np.zeros((len(listp),timestep_size,
            num_ske*3
            ),dtype=float)
  currentlen=np.zeros((len(listp)
    ))
  imlabel=np.zeros((len(listp),
           class_num),dtype=float)
  j=0
  i=0
  print('len',len(listp))
  for i in range (len(listp)-1):
    tmp=np.array(listp[i][0])    
    tmp=np.swapaxes(tmp,1,2)#(timestep_size,3,num_ske)
    tmp=np.reshape(tmp,[-1,3*num_ske])#(timestep_size,3*num_ske,)
    tmp_len=int(tmp.shape[0])
    currentlen[i]=tmp_len
    if tmp_len<timestep_size:
      tmp=np.vstack((tmp, np.zeros((timestep_size-tmp_len,3*num_ske))))
      #print('tmp2',tmp.shape)      
    else:
      tmp=tmp[0:timestep_size,:]
      currentlen[i]=timestep_size
    imdata[i]=tmp

  for j in range (0,len(listp)):
    currentclass=int(classname[j])
    imlabel[j,currentclass-1]=1#right=1 others=0
  return imdata,imlabel,currentlen

'''
load model
'''

#model=load_model('ke_model/model1.h5')


'''
     first step :choose model
'''
model = Sequential()
'''
   second step :build model
'''

model.add(LSTM(hidden_size,return_sequences=True))
model.add(Dropout(0.5))

model.add(LSTM(hidden_size, return_sequences=True))
model.add(Dropout(0.5))

model.add(LSTM(hidden_size, return_sequences=False))
model.add(Dropout(0.5))

model.add(Dense(class_num)) # 
model.add(Activation('softmax')) # 

'''
   third step:complie

'''
adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#model.summary()
model.compile(loss='categorical_crossentropy',  optimizer=adam,
              metrics=['accuracy']) # use cross loss

'''
   train model
'''
#get data
listp_train,classname_train=getlist(train_file)
listp_test,classname_test=getlist(test_file)

X_train, Y_train,len_train= get_all(listp_train,classname_train)
X_test, Y_test,len_test= get_all(listp_test,classname_test)


# evaluate model

loss,accuracy = model.evaluate(X_test, Y_test)

print('\nold model_s test loss',loss)
print('old model_s accuracy',accuracy)

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

for i in range(N):
  history=model.fit(X_train,Y_train,batch_size=batch_size,epochs=epoch_size,shuffle=True,verbose=2,validation_split=0.05)

  score, acc=model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)
  print('Test accuracy:', acc)


  model.save('ke_model/model1.h5')


'''
    5st step :output
'''
# print("test set")
# result = model.predict(X_test,batch_size=batch_size,verbose=0)

# result_max = np.argmax(result, axis = 1)
# test_max = np.argmax(Y_test, axis = 1)
# print(result_max[0:10],test_max[0:10])
# result_bool = np.equal(result_max, test_max)
# print(result_bool[0:10])
# true_num = np.sum(result_bool)
# print("")
# print("The accuracy of the model is %f" % (true_num/len(result_bool)))
# model.save('ke_model/model2.h5')


fig = plt.figure()

plt.plot(history.history['acc'])


plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')


plt.plot(history.history['loss'])


plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

fig.savefig('performance.png')
