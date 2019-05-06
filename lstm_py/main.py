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
train_time=100000

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

#def model(X_input,len_input,W_h,B_h,W, B, lstm_size):
def model(X_input,len_input,W, B, lstm_size):
    # X, input shape: (batch_size, 3*num_ske,time_step_size )
    X_input= tf.transpose(X_input, [2,0,1])#(batch_size,time_step_size ,3*num_ske)
    XR = tf.reshape(X_input, [-1, 3*num_ske]) # each row has input for each lstm cell (lstm_size=input_vec_size)
    # XR = tf.nn.relu(tf.matmul(XR, W_h) + B_h)
    # Each array shape: (batch_size, input_vec_size)
    X_split = tf.split(XR, timestep_size, 0) # split them to time_step_size (28 arrays),shape = [(128, 28),(128, 28)...]
    # Make lstm with lstm_size (each input vector size). num_units=lstm_size; forget_bias=1.0
    ###################################layer for b##########################################
    # lstm_cell1 = lstm_cell(hidden_size, 0.5)
    # init_state = lstm_cell1.zero_state(batch_size, dtype=tf.float32)
    # out_b,state_b=outputs, _states = rnn.static_rnn(lstm_cell1, X_split, dtype=tf.float32)
    # # lstm_cell.state_size=_states
    # out_B=tf.matmul(out_b[-1],W_b)+ B_b
    # out_B=tf.tile(out_B, [timestep_size,num_ske])
    # out_B=tf.reshape(out_B,[-1,3*num_ske])

    ###################################layer for R##########################################
    # lstm_cell2 = lstm_cell(hidden_size, 0.5)
    # init_state = lstm_cell2.zero_state(batch_size, dtype=tf.float32)
    # # outputs, _states = rnn.static_rnn(mlstm_cell, X_split, dtype=tf.float32)  
    # out_r,state_r= rnn.static_rnn(lstm_cell2, X_split, dtype=tf.float32)
    # # lstm_cell.state_size=_states
    # out_R=tf.matmul(out_r[-1],W_r)+ B_r
    # out_R=tf.tile(out_R, [timestep_size,num_ske])
    # out_R=tf.reshape(out_R,[-1,3*num_ske])
    # ##################################main layer############################################
    # XR=out_R*XR
    X_split = tf.split(XR, timestep_size, 0)
    mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(hidden_size, keep_prob) for _ in range(layer_num)], state_is_tuple = True)

    # Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
    outputs, _states = rnn.static_rnn(mlstm_cell, X_split, dtype=tf.float32)  
    # Linear activation
    # Get the last output
    return tf.matmul(outputs[-1],W) + B,mlstm_cell.state_size # State size to initialize the stat
def load_model():
	print('load model start')
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph('model/my-model.meta')
		saver.restore(sess, tf.train.latest_checkpoint("model/"))
##train start
# W_h = tf.Variable(tf.truncated_normal([3*num_ske,hidden_size], stddev=0.1), dtype=tf.float32)
# B_h = tf.Variable(tf.random_normal([hidden_size]), dtype=tf.float32)

W_m = tf.Variable(tf.truncated_normal([hidden_size,class_num], stddev=0.1), dtype=tf.float32)
B_m = tf.Variable(tf.random_normal([batch_size,class_num]), dtype=tf.float32)

# W_r = tf.Variable(tf.truncated_normal([hidden_size,3], stddev=0.1), dtype=tf.float32)
# B_r = tf.Variable(tf.random_normal([batch_size,3]), dtype=tf.float32)

# W_b = tf.Variable(tf.truncated_normal([hidden_size,3], stddev=0.1), dtype=tf.float32)
# B_b = tf.Variable(tf.random_normal([batch_size,3]), dtype=tf.float32)

#pre,state_size= model(X_input,len_input, W_b,B_b,W_r,B_r,W_m, B_m, hidden_size)
pre,state_size= model(X_input,len_input,W_m,B_m,hidden_size)

#print(pre.shape)
# cross_entropy = -tf.reduce_mean(y_input * tf.log(pre))
# train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pre, labels=y_input))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
predict_op = tf.argmax(pre, 1)

correct_prediction = tf.equal(tf.argmax(pre,1), tf.argmax( y_input,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



time0 = time.time()
listp_train,classname_train=getlist(train_file)
listp_test,classname_test=getlist(test_file)
tmp=np.array(listp_train[1][0])
print("lstm start")

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	load_model()      
	for i in range(train_time):
		X_batch, y_batch,len_batch= get_batch(batch_size,listp_train,classname_train)
		cost1, acc,  _ = sess.run([cost,accuracy, train_op], feed_dict={X_input: X_batch, y_input: y_batch,len_input:len_batch,keep_prob:0.5})
		
		if (i+1) % 500 == 0:
		    # distribe into 100 batches
		    print('step',i) 
		    test_acc = 0.0
		    test_cost = 0.0
		    N = 100
		    for j in range(N):
		        X_batch, y_batch,len_batch = get_batch(batch_size,listp_test,classname_test)
		        _cost, _acc = sess.run([cost, accuracy], feed_dict={X_input: X_batch, y_input: y_batch,len_input:len_batch,keep_prob:1})
		        test_acc += _acc
		        test_cost += _cost
		    print("step {}, train cost={:.6f}, acc={:.6f}; test cost={:.6f}, acc={:.6f}; pass {}s".format(i+1, cost1, acc, test_cost/N, test_acc/N, time.time() - time0))
		    time0 = time.time()
	saver = tf.train.Saver()
	saver.save(sess, "model/my-model")
		    
