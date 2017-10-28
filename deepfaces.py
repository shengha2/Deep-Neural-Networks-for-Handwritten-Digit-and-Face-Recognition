################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random


import tensorflow as tf

from caffe_classes import class_names
def readdata(filename):
    try:
        img = imread('cropped/'+filename+'.jpg')   
        result= []
        for line in img:
            result=np.append(result, line)
        return result
    except:
        try:
            img = imread('cropped/'+filename+'.png')    
            result= []
            for line in img:
                result=np.append(result, line)
            return result
        except:
            try:
                img = imread('cropped/'+filename+'.jpeg')   
                result= []
                for line in img:
                    result=np.append(result, line)
                return result
            except:
                img = imread('cropped/'+filename+'.JPG')   
                result= []
                for line in img:
                    result=np.append(result, line)
                return result                
        


name_list=['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth','Alec Baldwin', 'Steve Carell', 'Bill Hader']
trainning_size=2
validation_size=15
#M={"act0":np.empty([0,0]),"act1":np.empty([0,0]),"act2":np.empty([0,0]),"act3":np.empty([0,0]),"act4":np.empty([0,0]),"act5":np.empty([0,0]),"test0":np.empty([0,0]),"test1":np.empty([0,0]),"test2":np.empty([0,0]),"test3":np.empty([0,0]),"test4":np.empty([0,0]),"test5":np.empty([0,0])}

################################################
train_x = []
test_x = []
num_image=0
trainning_size = 50
train_y=np.zeros((len(name_list),trainning_size*len(name_list)))
testing_size = 30
test_y=np.zeros((len(name_list),testing_size*len(name_list)))
count = 0
for name in name_list:    
    num_image=0
    index = 0
    # print (name)
    while num_image<trainning_size:
        try:
          filename= name.split()[1]+str(index)
          train_x=np.append(train_x,readdata(filename))
          train_y[count,int(trainning_size*(count)+num_image)]+=1
          num_image+=1
          index +=1
        except:
            index +=1
            continue
    num_image = 0
    while num_image<testing_size:
        try:
          filename= name.split()[1]+str(index)
          test_x=np.append(test_x,readdata(filename))
          test_y[count,int(testing_size*(count)+num_image)]+=1
          num_image+=1
          index +=1
        except:
            index +=1
            continue
    count +=1
train_y = train_y.T 
train_x=np.resize(train_x,(trainning_size*count,227,227,3))
test_y = test_y.T 
test_x=np.resize(test_x,(testing_size*count,227,227,3))   



print('%%%%%%%%%')   

xdim = train_x.shape[1:]
ydim = train_y.shape[1]


################################################################################
#Read Image, and change to BGR


im1 = (imread("laska.png")[:,:,:3]).astype(float32)
im1 = im1 - mean(im1)
im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]

im2 = (imread("poodle.png")[:,:,:3]).astype(float32)
im2[:, :, 0], im2[:, :, 2] = im2[:, :, 2], im2[:, :, 0]


################################################################################


#In Python 3.5, change this to:
net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
#net_data = load("bvlc_alexnet.npy").item()

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



x = tf.placeholder(tf.float32, (None,) + xdim)


#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.Variable(net_data["conv5"][0])
conv5b = tf.Variable(net_data["conv5"][1])
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6W = tf.Variable(net_data["fc6"][0])
fc6b = tf.Variable(net_data["fc6"][1])
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
fc7W = tf.Variable(net_data["fc7"][0])
fc7b = tf.Variable(net_data["fc7"][1])
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
#fc(1000, relu=False, name='fc8')
fc8W = tf.Variable(net_data["fc8"][0])
fc8b = tf.Variable(net_data["fc8"][1])
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


#prob
#softmax(name='prob'))
prob = tf.nn.softmax(fc8)



t = time.time()
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
output = sess.run(conv4, feed_dict = {x:train_x})
output_test = sess.run(conv4, feed_dict = {x:test_x})


conv1W_array = sess.run(conv1W, feed_dict = {x:train_x})
################################################################################

#Output:
print (output.shape)

input_conv4=np.resize(output,(output.shape[0], output.shape[1]*output.shape[2]*output.shape[3]))
input_conv4_test=np.resize(output_test,(output_test.shape[0], output_test.shape[1]*output_test.shape[2]*output_test.shape[3]))

print (train_y.shape)
print (input_conv4.shape)

################################################################################
size=input_conv4.shape[1]
x = tf.placeholder(tf.float32, [None, size])



nhid = 1000
W0 = tf.Variable(tf.random_normal([size, nhid], stddev=0.01))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
b1 = tf.Variable(tf.random_normal([6], stddev=0.01))


layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1


y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, 6])



lam = 0.0
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_x, test_y = input_conv4_test,test_y

y_train=[]
y_validation=[]
y_test=[]
length = 1000
x_axis=range(0,length,10)
for i in range(length):
  batch_xs, batch_ys = input_conv4, train_y
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
  
  if i % 10 == 0:
    print ("i=",i)
    temp1=sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
    print ("Test:", temp1)
    y_test.append(temp1)
    batch_xs, batch_ys = input_conv4, train_y
    temp2=sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    print ("Train:", temp2)
    y_train.append(temp2)
    print ("Penalty:", sess.run(decay_penalty))

# print (tf.Session().run(W0).shape)
# print (tf.Session().run(W0))

plt.suptitle('Performance Using AlexNet Feature Activations')
plt.plot(x_axis, y_train, '-',label="Training set")
plt.plot(x_axis,y_test,'-',label="Testing set")
plt.axis([0, length, 0.0, 1.1])
plt.legend(loc='lower right')
plt.xlabel('Number of iterations', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.show()
