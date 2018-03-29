import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib

import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Convolution2D, Activation, Flatten, MaxPooling2D,Input,Dropout,GlobalAveragePooling2D
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.engine.topology import Layer
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.layers.normalization import BatchNormalization
from tensorflow.python.framework import ops
#from multi_gpu import make_parallel

def binarize(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    clipped = K.clip(x,-1,1)
    rounded = K.sign(clipped)
    return clipped + K.stop_gradient(rounded - clipped)

class Residual_sign(Layer):
    def __init__(self, levels=1,**kwargs):
        self.levels=levels
        super(Residual_sign, self).__init__(**kwargs)
    def build(self, input_shape):
        ars=np.arange(self.levels)+1.0
        ars=ars[::-1]
        means=ars/np.sum(ars)
        self.means=[K.variable(m) for m in means]
        self.trainable_weights=self.means
    def call(self, x, mask=None):
        resid = x
        out_bin=0
        for l in range(self.levels):
            out=binarize(resid)*K.abs(self.means[l])
            out_bin=out_bin+out
            resid=resid-out
        return out_bin

    def get_output_shape_for(self,input_shape):
        return input_shape
	def compute_output_shape(self,input_shape):
		return input_shape
    def set_means(self,X):
        means=np.zeros((self.levels))
        means[0]=1
        resid=np.clip(X,-1,1)
        approx=0
        for l in range(self.levels):
            m=np.mean(np.absolute(resid))
            out=np.sign(resid)*m
            approx=approx+out
            resid=resid-out
            means[l]=m
            err=np.mean((approx-np.clip(X,-1,1))**2)

        means=means/np.sum(means)
        sess=K.get_session()
        sess.run(self.means.assign(means))

class binary_conv(Layer):
	def __init__(self,nfilters,ch_in,k,padding,strides=(1,1),**kwargs):
		self.nfilters=nfilters
		self.ch_in=ch_in
		self.k=k
		self.padding=padding
		self.strides=strides
		super(binary_conv,self).__init__(**kwargs)
	def build(self, input_shape):
		stdv=1/np.sqrt(self.k*self.k*self.ch_in)
		w = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
		if keras.backend._backend=="mxnet":
			w=w.transpose(3,2,0,1)
		self.w=K.variable(w)
		self.gamma=K.variable(1.0)
		self.trainable_weights=[self.w,self.gamma]


	def call(self, x,mask=None):
		constraint_gamma=K.abs(self.gamma)#K.clip(self.gamma,0.01,10)
		self.clamped_w=constraint_gamma*binarize(self.w)
		if keras.__version__[0]=='2':
			self.out=K.conv2d(x, kernel=self.clamped_w, padding=self.padding,strides=self.strides )
		if keras.__version__[0]=='1':
			self.out=K.conv2d(x,self.clamped_w,border_mode=self.padding,strides=self.strides)
		self.output_dim=self.out.get_shape()
		return self.out
	def  get_output_shape_for(self,input_shape):
		return (input_shape[0], self.output_dim[1],self.output_dim[2],self.output_dim[3])
	def compute_output_shape(self,input_shape):
		return (input_shape[0], self.output_dim[1],self.output_dim[2],self.output_dim[3])

class binary_dense(Layer):
	def __init__(self,n_in,n_out,**kwargs):
		self.n_in=n_in
		self.n_out=n_out
		super(binary_dense,self).__init__(**kwargs)
	def build(self, input_shape):
		stdv=1/np.sqrt(self.n_in)
		w = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
		self.w=K.variable(w)
		self.gamma=K.variable(1.0)
		self.trainable_weights=[self.w,self.gamma]

	def call(self, x,mask=None):
		constraint_gamma=K.abs(self.gamma)#K.clip(self.gamma,0.01,10)
		self.clamped_w=constraint_gamma*binarize(self.w)
		self.out=K.dot(x,self.clamped_w)
		return self.out
	def  get_output_shape_for(self,input_shape):
		return (input_shape[0], self.n_out)
	def compute_output_shape(self,input_shape):
		return (input_shape[0], self.n_out)



"""
def binarize(x):
    #Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    g = tf.get_default_graph()
    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}):
            x=tf.clip_by_value(x,-1,1)
            return tf.sign(x)

class Residual_sign(Layer):
	def __init__(self, levels=1,**kwargs):
		self.levels=levels
		super(Residual_sign, self).__init__(**kwargs)
	def build(self, input_shape):
		ars=np.arange(self.levels)+1.0
		ars=ars[::-1]
		self.means=ars/np.sum(ars)
		self.means=tf.Variable(self.means,dtype=tf.float32)
		K.get_session().run(tf.variables_initializer([self.means]))
		self.trainable_weights=[self.means]

	def call(self, x,mask=None):
		resid = x
		out_bin=0
		for l in range(self.levels):
			out=binarize(resid)*K.abs(self.means[l])
			out_bin=out_bin+out
			resid=resid-out
		return out_bin

	def compute_output_shape(self,input_shape):
		return input_shape
	def set_means(self,X):
		means=np.zeros((self.levels))
		means[0]=1
		resid=np.clip(X,-1,1)
		approx=0
		for l in range(self.levels):
			m=np.mean(np.absolute(resid))
			out=np.sign(resid)*m
			approx=approx+out
			resid=resid-out
			means[l]=m
			err=np.mean((approx-np.clip(X,-1,1))**2)

		means=means/np.sum(means)
		sess=K.get_session()
		sess.run(self.means.assign(means))

class binary_conv(Layer):
	def __init__(self,nfilters,ch_in,k,padding,**kwargs):
		self.nfilters=nfilters
		self.ch_in=ch_in
		self.k=k
		self.padding=padding
		super(binary_conv,self).__init__(**kwargs)
	def build(self, input_shape):
		stdv=1/np.sqrt(self.k*self.k*self.ch_in)
		w = tf.random_normal(shape=[self.k,self.k,self.ch_in,self.nfilters], mean=0.0, stddev=stdv, dtype=tf.float32)
		self.w=K.variable(w)
		self.gamma=K.variable([1.0])
		self.trainable_weights=[self.w,self.gamma]
	def call(self, x,mask=None):
		constraint_gamma=K.abs(self.gamma)
		self.clamped_w=constraint_gamma*binarize(self.w)
		self.out=K.conv2d(x, kernel=self.clamped_w, padding=self.padding)#tf.nn.convolution(x, filter=self.clamped_w , padding=self.padding)
		self.output_dim=self.out.get_shape()
		#self.out=Convolution2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding='valid', use_bias=False)(x)
		return self.out
	def  compute_output_shape(self,input_shape):
		return (input_shape[0], self.output_dim[1],self.output_dim[2],self.output_dim[3])

class binary_dense(Layer):
	def __init__(self,n_in,n_out,**kwargs):
		self.n_in=n_in
		self.n_out=n_out
		super(binary_dense,self).__init__(**kwargs)
	def build(self, input_shape):
		stdv=1/np.sqrt(self.n_in)
		w = tf.random_normal(shape=[self.n_in,self.n_out], mean=0.0, stddev=stdv, dtype=tf.float32)
		self.w=K.variable(w)
		self.gamma=K.variable([1.0])
		self.trainable_weights=[self.w,self.gamma]
	def call(self, x, mask=None):
		constraint_gamma=K.abs(self.gamma)
		self.clamped_w=constraint_gamma*binarize(self.w)
		self.out=K.dot(x, self.clamped_w)
		self.output_dim=self.out.get_shape()
		return self.out
	def  compute_output_shape(self,input_shape):
		return (input_shape[0], self.output_dim[1])
"""
class my_flat(Layer):
	def __init__(self,**kwargs):
		super(my_flat,self).__init__(**kwargs)
	def build(self, input_shape):
		return

	def call(self, x, mask=None):
		self.out=tf.reshape(x,[-1,np.prod(x.get_shape().as_list()[1:])])
		return self.out
	def  compute_output_shape(self,input_shape):
		shpe=(input_shape[0],int(np.prod(input_shape[1:])))
		return shpe
