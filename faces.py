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
import pickle

import os
from scipy.io import loadmat

from scipy.misc import imsave as save

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result
testfile = urllib.URLopener()            


#Note: you need to create the uncropped folder first in order 
#for this to work

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

filenames = ["subset_actresses.txt", "subset_actors.txt"]
for subset in range(len(filenames)):
    act = list(set([a.split("\t")[0] for a in open(filenames[subset]).readlines()]))
    for a in act:
        name = a.split()[1].lower()
        i = 0
        j = 0
        for line in open(filenames[subset]):
            if a in line:
                sentence = line.split()
                filename = sentence[1]+str(i)+'.jpg'
                dimensions = sentence[-2]
                dimensions = dimensions.replace(",", " ")
                dimensions = dimensions.split()
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                # urllib.urlretrieve(line.split()[4], "uncropped/"+filename)
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                if not os.path.isfile("uncropped/"+filename):
                    continue
                try:
                    im = imread("uncropped/"+filename)
                    im = im[int(dimensions[1]):int(dimensions[3]),int(dimensions[0]):int(dimensions[2])]
                    im = imresize(im, (227,227))
                    im = rgb2gray(im)
                    save("cropped/"+filename, im)
                    print ("success:", filename)
                    print ("number of corrupt images:", j )                   
                    i += 1
                except:
                    j+=1
                    print ("read failed:", filename)
                    print ("number of corrupt images:", j)
                    i+=1
                    continue

t = int(time.time())
#t = 1454219613
print ("t=", t)
random.seed(t)
dimension=227
size=dimension*dimension

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
M={"act0":np.empty([0,0]),"act1":np.empty([0,0]),"act2":np.empty([0,0]),"act3":np.empty([0,0]),"act4":np.empty([0,0]),"act5":np.empty([0,0]),"test0":np.empty([0,0]),"test1":np.empty([0,0]),"test2":np.empty([0,0]),"test3":np.empty([0,0]),"test4":np.empty([0,0]),"test5":np.empty([0,0])}

count=0
for name in name_list:    
    num_image=0
    for i in range(31,230):
        try:
            filename= name.split()[1].lower()+str(i)
            M["act"+str(count)]=np.append(M["act"+str(count)],readdata(filename))
            num_image+=1
        except:
            continue
       
    M["act"+str(count)].resize((num_image,size))
    print(M["act"+str(count)].shape)   
    count+=1
print('%%%%%%%%%')   
count=0   
for name in name_list:    
    num_image=0
    for i in range(0,30):
        #try:
        filename= name.split()[1].lower()+str(i)
        M["test"+str(count)]=np.append(M["test"+str(count)],readdata(filename))
        num_image+=1
        #except:
            #continue
        
    M["test"+str(count)].resize((num_image,size))
    print(M["test"+str(count)].shape)
    count+=1   




def get_train_batch(M, N):
    n = N/10
    batch_xs = zeros((0, dimension*dimension))
    batch_y_s = zeros( (0, 6))
    
    train_k =  ["act"+str(i) for i in range(10)]

    train_size = len(M[train_k[0]])
    #train_size = 5000
    
    for k in range(6):
        train_size = len(M[train_k[k]])
        idx = array(random.permutation(train_size)[:n])
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[idx])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (n, 1))   ))
    return batch_xs, batch_y_s
    

def get_test(M):
    batch_xs = zeros((0, dimension*dimension))
    batch_y_s = zeros( (0, 6))
    
    test_k =  ["test"+str(i) for i in range(10)]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s


def get_train(M):
    batch_xs = zeros((0, dimension*dimension))
    batch_y_s = zeros( (0, 6))
    
    train_k =  ["act"+str(i) for i in range(10)]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s
        



x = tf.placeholder(tf.float32, [None, size])


nhid = 200
W0 = tf.Variable(tf.random_normal([size, nhid], stddev=0.01))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
b1 = tf.Variable(tf.random_normal([6], stddev=0.01))

#snapshot = pickle.load(open("snapshot50.pkl","rb"))
#W0 = tf.Variable(snapshot["W0"])
#b0 = tf.Variable(snapshot["b0"])
#W1 = tf.Variable(snapshot["W1"])
#b1 = tf.Variable(snapshot["b1"])


layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1


y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, 6])



lam = 0.1
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_x, test_y = get_test(M)

y_training=[]
y_validation=[]
y_test=[]
x=[]
for i in range(200):
  #print i  
  batch_xs, batch_ys = get_train_batch(M, 180)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  x=x.append(i)
  
  if i % 50 == 0:
    print ("i=",i)
    temp=sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
    print ("Test:", temp)
    y_test.append(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
    batch_xs, batch_ys = get_train(M)
    temp=sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    print ("Train:", temp)
    y_test.append(temp)
    print ("Penalty:", sess.run(decay_penalty))
   
#for i in range(6):
    #print(sess.run(W1).shape)   
    #temp=sess.run(W1)[:,i].argmax()
    #print (temp)
    #print(sess.run(W0).shape)
    #im=sess.run(W0)[:,temp]
    #im=np.resize(im,(dimension,dimension))
    #plt.figure(i+4)
    #plt.imshow(im, interpolation='none', cmap='coolwarm')
    #plt.title(name_list[i])
    #plt.show()
plt.plot(x, y_training, '-',label="Training set")
plt.plot(x, y_validation, '-',label="Validation set")
plt.plot(x,y_test,'-',label="test set")
plt.axis([0, 5000, 0.0, 1.1])
plt.legend()
plt.xlabel('Number of iterations', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
    #snapshot = {}
    #snapshot["W0"] = sess.run(W0)
    #snapshot["W1"] = sess.run(W1)
    #snapshot["b0"] = sess.run(b0)
    #snapshot["b1"] = sess.run(b1)
    #cPickle.dump(snapshot,  open("new_snapshot"+str(i)+".pkl", "w"))