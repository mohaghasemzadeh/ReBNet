# Copyright 2016 Google Inc. All Rights Reserved.
# Copyright 2017 NVIDIA Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse

import keras
from keras.applications import ResNet50, VGG16, imagenet_utils,VGG19
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
import math
import multiprocessing
import pickle

import mxnet as mx
import numpy as np
from time import time
from keras.models import load_model

import data
from keras import backend as K
import sys
sys.path.insert(0, '..')
from binarization_utils import *
from model_architectures import get_model

dataset='Imagenet'
Train=True
Evaluate=False

def load_from_pickle(model,file_dir):
    foo=open(file_dir,'rb')
    dic=pickle.load(foo)
    foo.close()
    #dic={'convs':convs,'denses':denses,'activation_gammas':activation_gammas,'batch_norms':batch_norms}
    convs=dic['convs']
    denses=dic['denses']
    activation_gammas=dic['activation_gammas']
    batch_norms=dic['batch_norms']
    for l in model.layers:
        if isinstance(l,binary_dense):


            l.trainable_weights=[]
            val=denses[0]
            l.non_trainable_weights=[l.w]
            l.set_weights([val[0]])

            l.non_trainable_weights=[l.gamma]
            gamma=l.get_weights()[0]
            gamma=gamma*0+val[1]
            print 'gamma:',gamma
            l.set_weights([gamma])

            l.non_trainable_weights=[l.alpha]
            alpha=l.get_weights()[0]
            alpha=alpha*0+val[2]
            l.set_weights([alpha])
            print 'alpha:',alpha
            l.trainable_weights=[l.w]
            non_trainable_weights=[l.gamma,l.alpha]
            denses.remove(denses[0])

        if isinstance(l,BatchNormalization):
            l.set_weights([batch_norms[0][1],batch_norms[0][0],batch_norms[0][2],batch_norms[0][3]])
            batch_norms.remove(batch_norms[0])
            #print l.trainable_weights,l.non_trainable_weights
        if isinstance(l,binary_conv):
            l.trainable_weights=[]
            val=convs[0]
            l.non_trainable_weights=[l.w]
            l.set_weights([val[0]])

            l.non_trainable_weights=[l.gamma]
            gamma=l.get_weights()[0]
            gamma=gamma*0+val[1]
            print 'gamma:',gamma
            l.set_weights([gamma])

            l.non_trainable_weights=[l.alpha]
            alpha=l.get_weights()[0]
            alpha=alpha*0+val[2]
            l.set_weights([alpha])
            print 'alpha:',alpha
            l.trainable_weights=[l.w]
            non_trainable_weights=[l.gamma,l.alpha]
            #l.set_weights([val[0],val[1],val[2]])
            convs.remove(convs[0])
        if isinstance(l,Residual_sign):
            gammas=activation_gammas[0]
            l.trainable_weights=[m for m in l.means]
            l.set_weights(gammas)
            """l.trainable_weights=[]
            l.non_trainable_weights=[l.means[0]]
            m=l.get_weights()[0]*0+val
            l.set_weights([m])
            l.trainable_weights=l.means
            l.non_trainable_weights=[]"""
            activation_gammas.remove(activation_gammas[0])

def backend_agnostic_compile(model, loss, optimizer, metrics, args):
    if keras.backend._backend == 'mxnet':
        gpu_list = ["gpu(%d)" % i for i in range(args.num_gpus)]
        model.compile(loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            context = gpu_list)
    else:
        if args.num_gpus > 1:
            print("Warning: num_gpus > 1 but not using MxNet backend")
        model.compile(loss=loss,
            optimizer=optimizer,
            metrics=metrics)

# Get data from iterator and report samples per second if desired
def get_data(it, batch_size, report_speed=False, warm_batches_up_for_reporting=100):
    ctr = 0
    warm_up_done = False
    last_time = None

    # Need to feed data as NumPy arrays to Keras
    def get_arrays(db):
        return db.data[0].asnumpy(), to_categorical(db.label[0].asnumpy(), args.num_classes)#db.label[0].asnumpy()

    # repeat for as long as training is to proceed, reset iterator if need be
    while True:
        try:
            ctr += 1
            db = it.next()

            # Skip this if samples/second reporting is not desired
            if report_speed:

                # Report only after warm-up is done to prevent downward bias
                if warm_up_done:
                    curr_time = time()
                    elapsed =  curr_time - last_time
                    ss = float(batch_size * ctr) / elapsed
                    print(" Batch: %d, Samples per sec: %.2f" % (ctr, ss))

                if ctr > warm_batches_up_for_reporting and not warm_up_done:
                   ctr = 0
                   last_time = time()
                   warm_up_done = True

        except StopIteration as e:
            print("get_data exception due to end of data - resetting iterator")
            it.reset()
            db = it.next()

        finally:
            yield get_arrays(db)

class lr_schedule(keras.callbacks.Callback):
    def __init__(self,model):
        self.counter=0
        self.levels= [0.0001,0.00008,0.00005,0.00003,0.00002,0.00001]
        self.model=model
        self.best_score = -1.
        self.decay = 0.01


    def on_epoch_end(self, epoch, logs={}):
        current_score = logs.get('loss')
        if current_score < self.best_score:
            self.best_score = current_score
        else:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr,lr*0.5)

    '''def on_epoch_begin(self,batch,logs={}):
        self.counter=self.counter+1
        lr = K.get_value(self.model.optimizer.lr)
        if self.counter==1:
            K.set_value(self.model.optimizer.lr, self.levels[0])
        elif self.counter==5:
            K.set_value(self.model.optimizer.lr, self.levels[1])
        elif self.counter==10:
            K.set_value(self.model.optimizer.lr, self.levels[2])
        elif self.counter==17:
            K.set_value(self.model.optimizer.lr, self.levels[3])
        elif self.counter==30:
            K.set_value(self.model.optimizer.lr, self.levels[4])
        elif self.counter==45:
            K.set_value(self.model.optimizer.lr, self.levels[5])'''


parser = argparse.ArgumentParser(description="Train_Binary_Imagenet",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

data.add_data_args(parser)
data.add_data_aug_args(parser)
parser.add_argument_group('gpu_config', 'gpu config')
parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs to use during training')
parser.add_argument('--batch-per-gpu', type=int, default=64, help='Batch size per GPU')

# Use a large augmentation level (needed by ResNet-50)
data.set_data_aug_level(parser, 3)

parser.set_defaults(
    # data
    num_classes      = 1000,
    num_examples     = 1281167,
    image_shape      = '3,224,224',
    min_random_scale = 1,
    # Assume HyperThreading on x86, only count physical cores
    data_nthreads    = multiprocessing.cpu_count() // 2,
    validation_ex    = 50099
)

args = parser.parse_args()
args.data_train='/home/hamid/imagenet/train.rec'
args.data_val='/home/hamid/imagenet/val.rec'

global_batch_size = args.num_gpus * args.batch_per_gpu
args.batch_size = global_batch_size


num_epoch=100



# Get training and validation iterators
train_iter, val_iter = data.get_rec_iter(args)
train_gen = get_data(train_iter, batch_size=global_batch_size, report_speed=True)
val_gen = get_data(val_iter, batch_size=global_batch_size, report_speed=True)

it_per_epoch = int(math.ceil(1.0 * args.num_examples / global_batch_size))

print("Number of iterations per epoch: %d" % it_per_epoch)
print("Using %d GPUs, batch size per GPU: %d, total batch size: %d" % (args.num_gpus, args.batch_per_gpu, global_batch_size))


if Train:
    if not(os.path.exists('models')):
        os.mkdir('models')
    if not(os.path.exists('models/'+dataset)):
        os.mkdir('models/'+dataset)
    for resid_levels in range(1,4):
        print 'training with', resid_levels,'levels'
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
        lr=0.0005
        opt = keras.optimizers.Adam(lr=lr,decay=1e-6)#SGD(lr=lr,momentum=0.9,decay=1e-5)
        backend_agnostic_compile(model=model, loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'], args=args)

        weights_path='models/'+dataset+'/'+str(resid_levels)+'_residuals.h5'
        logs_path='models/'+dataset+'/'+str(resid_levels)+'_residuals.log'
        model_saver=keras.callbacks.ModelCheckpoint(weights_path, monitor='val_acc', save_best_only=True)
        csv_logger = keras.callbacks.CSVLogger(logs_path)

        history=model.fit_generator(generator=train_gen, samples_per_epoch=args.num_examples, nb_epoch=num_epoch, verbose=True,validation_data=val_gen,
                                 nb_val_samples=50000, class_weight=None, max_q_size=2, nb_worker=1, pickle_safe=False, initial_epoch=0,callbacks=[model_saver,csv_logger])
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
