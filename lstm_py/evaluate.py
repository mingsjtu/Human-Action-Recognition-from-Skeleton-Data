import warnings
warnings.filterwarnings('ignore')  # no printing  warning 

import tensorflow as tf
from tensorflow.contrib import rnn
import os
import numpy as np
#import pandas as pd
import time

# arrange GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True





#####################arrage parameter########################
train_file='CV_20/train'
test_file='CV_20/test'

num_ske = 25      
hidden_size = 100    # hidden layer num
layer_num = 4        # LSTM layer num
class_num = 60        # last class number
cell_type = "lstm"   # lstm or block_lstm
NUM_SKE=25
batch_size =128  # tf.int32, batch_size = 128
timestep_size=150
learning_rate=1e-3
train_time=1000000
N=100
X_input = tf.placeholder(tf.float32, [batch_size, 3*num_ske,timestep_size])
y_input = tf.placeholder(tf.float32, [batch_size, class_num])
len_input = tf.placeholder(tf.float32, [batch_size])
keep_prob = tf.placeholder(tf.float32, [])

def eachFile(folder):
    allFile = os.listdir(folder)
    fileNames = []
    for file in allFile:
        
        fileNames.append(file)
    return fileNames
#build lstm
def lstm_cell(num_nodes, keep_prob):
	cell = tf.contrib.rnn.BasicLSTMCell(num_nodes,reuse=tf.get_variable_scope().reuse)
	cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
	return cell

outputsx = list()
outputsx_=list()

def read_txt(txtfile):
	file=open(txtfile, 'r')
	list_read = file.readlines()
	return list_read
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



def get_batch(batch_size,listp,classname):
	#print(type(listp[1])[0])
	sta=np.random.randint(1, len(listp), size=batch_size) 
	imdata=np.zeros((batch_size,num_ske*3,
            timestep_size
            ),dtype=float)
	currentlen=np.zeros((batch_size
		))
	imlabel=np.zeros((batch_size,
           class_num),dtype=float)
	j=0
	i=0
	for i in range (0,batch_size):
		tmp=np.array(listp[sta[i]][0])		
		tmp=np.swapaxes(tmp,1,2)#(timestep_size,3,num_ske)
		tmp=np.reshape(tmp,[3*num_ske,-1])#(3*num_ske,timestep_size)
		tmp_len=int(tmp.shape[1])
		currentlen[i]=tmp_len
		if tmp_len<timestep_size:
			tmp=np.hstack((tmp, np.zeros((3*num_ske,timestep_size-tmp_len))))
			#print('tmp2',tmp.shape)			
		else:
			tmp=tmp[:,0:timestep_size]
			currentlen[i]=timestep_size
		imdata[i]=tmp

	for j in range (0,batch_size):
		currentclass=int(classname[sta[j]])
		imlabel[j,currentclass-1]=1#right=1 others=0
	return imdata,imlabel,currentlen

def model(X_input,len_input,W, B, lstm_size):
    # X, input shape: (batch_size, 3*num_ske,time_step_size )
    X_input= tf.transpose(X_input, [2,0,1])#(batch_size,time_step_size ,3*num_ske)
    XR = tf.reshape(X_input, [-1, 3*num_ske]) # each row has input for each lstm cell (lstm_size=input_vec_size)
    X_split = tf.split(XR, timestep_size, 0) # split them to time_step_size (28 arrays),shape = [(128, 28),(128, 28)...]
    X_split = tf.split(XR, timestep_size, 0)
    mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(hidden_size, keep_prob) for _ in range(layer_num)], state_is_tuple = True)

    # Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
    # rnn..static_rnn()的输出对应于每一个timestep，如果只关心最后一步的输出，取outputs[-1]即可
    outputs, _states = rnn.static_rnn(mlstm_cell, X_split, dtype=tf.float32)  # 时间序列上每个Cell的输出:[... shape=(128, 28)..]
    # Linear activation
    # Get the last output
    return tf.matmul(outputs[-1],W) + B,mlstm_cell.state_size # State size to initialize the stat
def load_model():
	print('load model start')
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph('model/my-model.meta')
		saver.restore(sess, tf.train.latest_checkpoint("model/"))
##train start


W_m = tf.Variable(tf.truncated_normal([hidden_size,class_num], stddev=0.1), dtype=tf.float32)
B_m = tf.Variable(tf.random_normal([batch_size,class_num]), dtype=tf.float32)

pre,state_size= model(X_input,len_input,W_m,B_m,hidden_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pre, labels=y_input))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
predict_op = tf.argmax(pre, 1)

correct_prediction = tf.equal(tf.argmax(pre,1), tf.argmax( y_input,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



time0 = time.time()
listp_train,classname_train=getlist(train_file)
listp_test,classname_test=getlist(test_file)
tmp=np.array(listp_train[1][0])
saver = tf.train.Saver()
load_model()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	graph = tf.get_default_graph()
	writer_test=tf.summary.FileWriter('logs/load',sess.graph)
	test_acc = 0.0
	test_cost = 0.0      
	for j in range(N):
	    X_batch, y_batch,len_batch = get_batch(batch_size,listp_test,classname_test)
	    _cost, _acc = sess.run([cost, accuracy], feed_dict={X_input: X_batch, y_input: y_batch,len_input:len_batch,keep_prob:1})
	    test_acc += _acc
	    test_cost += _cost
	print("test acc={:.6f}; pass {}s".format(test_acc/N, time.time() - time0))
	#saver.save(sess, "model/my-model")
	time0 = time.time()