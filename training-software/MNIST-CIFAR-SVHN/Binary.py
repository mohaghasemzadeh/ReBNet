import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import keras
from keras.datasets import cifar10,mnist
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
import sys
sys.path.insert(0, '..')
from binarization_utils import *
from model_architectures import get_model

dataset='CIFAR-10'
Train=True
Evaluate=False
batch_size=100
epochs=200

def load_svhn(path_to_dataset):
	import scipy.io as sio
	train=sio.loadmat(path_to_dataset+'/train.mat')
	test=sio.loadmat(path_to_dataset+'/test.mat')
	extra=sio.loadmat(path_to_dataset+'/extra.mat')
	X_train=np.transpose(train['X'],[3,0,1,2])
	y_train=train['y']-1

	X_test=np.transpose(test['X'],[3,0,1,2])
	y_test=test['y']-1

	X_extra=np.transpose(extra['X'],[3,0,1,2])
	y_extra=extra['y']-1

	X_train=np.concatenate((X_train,X_extra),axis=0)
	y_train=np.concatenate((y_train,y_extra),axis=0)

	return (X_train,y_train),(X_test,y_test)

if dataset=="MNIST":
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	# convert class vectors to binary class matrices
	X_train = X_train.reshape(-1,784)
	X_test = X_test.reshape(-1,784)
	use_generator=False
elif dataset=="CIFAR-10":
	use_generator=True
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
elif dataset=="SVHN":
	use_generator=True
	(X_train, y_train), (X_test, y_test) = load_svhn('./svhn_data')
else:
	raise("dataset should be one of the following: [MNIST, CIFAR-10, SVHN].")

X_train=X_train.astype(np.float32)
X_test=X_test.astype(np.float32)
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
X_train /= 255
X_test /= 255
X_train=2*X_train-1
X_test=2*X_test-1


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')





if Train:
	if not(os.path.exists('models')):
		os.mkdir('models')
	if not(os.path.exists('models/'+dataset)):
		os.mkdir('models/'+dataset)
	for resid_levels in range(1,4):
		print 'training with', resid_levels,'levels'
		sess=K.get_session()
		model=get_model(dataset,resid_levels)
		#model.summary()

		#gather all binary dense and binary convolution layers:
		binary_layers=[]
		for l in model.layers:
			if isinstance(l,binary_dense) or isinstance(l,binary_conv):
				binary_layers.append(l)

		#gather all residual binary activation layers:
		resid_bin_layers=[]
		for l in model.layers:
			if isinstance(l,Residual_sign):
				resid_bin_layers.append(l)
		lr=0.01
		opt = keras.optimizers.Adam(lr=lr,decay=1e-6)#SGD(lr=lr,momentum=0.9,decay=1e-5)
		model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


		weights_path='models/'+dataset+'/'+str(resid_levels)+'_residuals.h5'
		cback=keras.callbacks.ModelCheckpoint(weights_path, monitor='val_acc', save_best_only=True)
		if use_generator:
			if dataset=="CIFAR-10":
				horizontal_flip=True
			if dataset=="SVHN":
				horizontal_flip=False
			datagen = ImageDataGenerator(
				width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
				height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
				horizontal_flip=horizontal_flip)  # randomly flip images
			if keras.__version__[0]=='2':
				history=model.fit_generator(datagen.flow(X_train, y_train,batch_size=batch_size),steps_per_epoch=X_train.shape[0]/batch_size,
				nb_epoch=epochs,validation_data=(X_test, y_test),verbose=2,callbacks=[cback])
			if keras.__version__[0]=='1':
				history=model.fit_generator(datagen.flow(X_train, y_train,batch_size=batch_size), samples_per_epoch=X_train.shape[0], 
				nb_epoch=epochs, verbose=2,validation_data=(X_test,y_test),callbacks=[cback])

		else:
			if keras.__version__[0]=='2':
				history=model.fit(X_train, y_train,batch_size=batch_size,validation_data=(X_test, y_test), verbose=2,epochs=epochs,callbacks=[cback])
			if keras.__version__[0]=='1':
				history=model.fit(X_train, y_train,batch_size=batch_size,validation_data=(X_test, y_test), verbose=2,nb_epoch=epochs,callbacks=[cback])
		dic={'hard':history.history}
		foo=open('models/'+dataset+'/history_'+str(resid_levels)+'_residuals.pkl','wb')
		pickle.dump(dic,foo)
		foo.close()

if Evaluate:
	for resid_levels in range(1,4):
		weights_path='models/'+dataset+'/'+str(resid_levels)+'_residuals.h5'
		model=get_model(dataset,resid_levels)
		model.load_weights(weights_path)
		opt = keras.optimizers.Adam()
		model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
		#model.summary()
		score=model.evaluate(X_test,Y_test,verbose=0)
		print "with %d residuals, test loss was %0.4f, test accuracy was %0.4f"%(resid_levels,score[0],score[1])



"""if hardware_debug:
	inp=X_train[0:1]
	sess=K.get_session()
	resid_levels=3
	model=Sequential()
	model.add(binary_dense(n_in=784,n_out=256,change='clip',input_shape=[784]))
	#model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(Residual_sign(levels=resid_levels))

	for i,l in enumerate(model.layers):
		if isinstance(l,binary_dense) or isinstance(l,binary_conv):
			l.hard_binarize()

	w,gamma_w=model.layers[0].get_weights()
	gamma_act=np.absolute(sess.run(model.layers[-1].means))
	gamma=gamma_act*gamma_w
	w=(w>0).astype(np.int32)
	w=w.transpose()

	b=np.zeros(256)
	alpha=np.ones(256)
	print w.shape,b.shape,gamma
	out=model.predict(inp)


	inp=np.squeeze(inp)
	inp=inp*2**(8)
	inp=inp.astype(np.uint64)
	out=np.squeeze(out)
	out=np.maximum(out,0)
	print out
	out=out*2**16
	out=out.astype(np.uint64)

	import struct
	outFile=open('params/input.bin','wb')
	for i in range(inp.shape[0]):
		outFile.write(struct.pack('Q', inp[i]))
	outFile.close()

	outFile=open('params/output.bin','wb')
	for i in range(out.shape[0]):
		outFile.write(struct.pack('Q', out[i]))
	outFile.close()

	print out

	store_layer_weights(w,b,peCount=16,simdCount=64,alpha=alpha,targetDirBin='params',layer_num=0,gamma=gamma,target_config_file=None)"""
