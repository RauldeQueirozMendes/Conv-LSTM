from __future__ import absolute_import, division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Do other imports now...

#IMPORT
import skimage.io as io
import cv2
import numpy as np
import glob
import sys, os
import tensorflow as tf
import h5py
from tensorflow import keras
import imageio
import matplotlib.pyplot as plt
import os
import time
import math
import warnings
from scipy import misc
import argparse
import matplotlib as mpl
import argparse
import random
import itertools
from keras_preprocessing.image import apply_brightness_shift,apply_channel_shift,img_to_array,load_img,apply_affine_transform
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default


warnings.filterwarnings("ignore")
saveModel = False

parser = argparse.ArgumentParser()

parser.add_argument('-f','--foo', required=True)
parser.add_argument('-b','--bar', required=True)

print(tf.__version__)

LOG_INITIAL_VALUE = 1

def load_and_scale_image(filepath):
    image_input = img_to_array(load_img(filepath, target_size=(128,416), interpolation='lanczos'))
    image_input = image_input.astype(np.float32)
    image_input = np.expand_dims(image_input,axis=0)
    return image_input/255.0


def load_and_scale_depth(filepath):
    depth_gt = img_to_array(load_img(filepath, grayscale=True, color_mode='grayscale', target_size=(128,416), interpolation='lanczos'))/3.0
    depth_gt = depth_gt.astype(np.float32)
    depth_gt = np.expand_dims(depth_gt,axis=0)
    return depth_gt/90.0    #TODO: Change it to 90.0 and see the sigmoid x limits before saturation


def imageLoader(img_filenames,depth_filenames,depth_skip,seq_len=3):

    assert len(img_filenames) == len(depth_filenames)

    numSamples = len(img_filenames)

    while True:

        batch_start = 0
        batch_end = seq_len

        while batch_start < numSamples:

            flag = 0

            limit = min(batch_end,numSamples)

            img_batch = np.concatenate(list(map(load_and_scale_image,img_filenames[batch_start:limit])),0)
            depth_batch = np.concatenate(list(map(load_and_scale_depth,depth_filenames[batch_start:limit])),0)

            img_final = np.expand_dims(img_batch,0)
            depth_final = np.expand_dims(depth_batch,0)

            # input("Pause")

            yield (img_final,depth_final)  # A tuple with two numpy arrays with batch_size samples

            if((limit + 1) <= numSamples):

                for i in range(len(depth_skip)):

                    if (depth_filenames[limit] == depth_skip[i]): #adjust input every first file image

                        flag = 1

                if(flag == 1):

                    batch_start += seq_len
                    batch_end += seq_len


                else:

                    batch_start += 1
                    batch_end += 1

            else:

                del img_batch
                del depth_batch
                del img_final
                del depth_final
                batch_start = 0
                batch_end = seq_len


if not (os.path.exists('kitti_continuous_train (2).txt') and os.path.exists('kitti_continuous_test (2).txt')):

    timer1 = -time.time()

    bad_words = ['image_03',
                 '2011_09_28_drive_0053_sync',
                 '2011_09_28_drive_0054_sync',
                 '2011_09_28_drive_0057_sync',
                 '2011_09_28_drive_0065_sync',
                 '2011_09_28_drive_0066_sync',
                 '2011_09_28_drive_0068_sync',
                 '2011_09_28_drive_0070_sync',
                 '2011_09_28_drive_0071_sync',
                 '2011_09_28_drive_0075_sync',
                 '2011_09_28_drive_0077_sync',
                 '2011_09_28_drive_0078_sync',
                 '2011_09_28_drive_0080_sync',
                 '2011_09_28_drive_0082_sync',
                 '2011_09_28_drive_0086_sync',
                 '2011_09_28_drive_0087_sync',
                 '2011_09_28_drive_0089_sync',
                 '2011_09_28_drive_0090_sync',
                 '2011_09_28_drive_0094_sync',
                 '2011_09_28_drive_0095_sync',
                 '2011_09_28_drive_0096_sync',
                 '2011_09_28_drive_0098_sync',
                 '2011_09_28_drive_0100_sync',
                 '2011_09_28_drive_0102_sync',
                 '2011_09_28_drive_0103_sync',
                 '2011_09_28_drive_0104_sync',
                 '2011_09_28_drive_0106_sync',
                 '2011_09_28_drive_0108_sync',
                 '2011_09_28_drive_0110_sync',
                 '2011_09_28_drive_0113_sync',
                 '2011_09_28_drive_0117_sync',
                 '2011_09_28_drive_0119_sync',
                 '2011_09_28_drive_0121_sync',
                 '2011_09_28_drive_0122_sync',
                 '2011_09_28_drive_0125_sync',
                 '2011_09_28_drive_0126_sync',
                 '2011_09_28_drive_0128_sync',
                 '2011_09_28_drive_0132_sync',
                 '2011_09_28_drive_0134_sync',
                 '2011_09_28_drive_0135_sync',
                 '2011_09_28_drive_0136_sync',
                 '2011_09_28_drive_0138_sync',
                 '2011_09_28_drive_0141_sync',
                 '2011_09_28_drive_0143_sync',
                 '2011_09_28_drive_0145_sync',
                 '2011_09_28_drive_0146_sync',
                 '2011_09_28_drive_0149_sync',
                 '2011_09_28_drive_0153_sync',
                 '2011_09_28_drive_0154_sync',
                 '2011_09_28_drive_0155_sync',
                 '2011_09_28_drive_0156_sync',
                 '2011_09_28_drive_0160_sync',
                 '2011_09_28_drive_0161_sync',
                 '2011_09_28_drive_0162_sync',
                 '2011_09_28_drive_0165_sync',
                 '2011_09_28_drive_0166_sync',
                 '2011_09_28_drive_0167_sync',
                 '2011_09_28_drive_0168_sync',
                 '2011_09_28_drive_0171_sync',
                 '2011_09_28_drive_0174_sync',
                 '2011_09_28_drive_0177_sync',
                 '2011_09_28_drive_0179_sync',
                 '2011_09_28_drive_0183_sync',
                 '2011_09_28_drive_0184_sync',
                 '2011_09_28_drive_0185_sync',
                 '2011_09_28_drive_0186_sync',
                 '2011_09_28_drive_0187_sync',
                 '2011_09_28_drive_0191_sync',
                 '2011_09_28_drive_0192_sync',
                 '2011_09_28_drive_0195_sync',
                 '2011_09_28_drive_0198_sync',
                 '2011_09_28_drive_0199_sync',
                 '2011_09_28_drive_0201_sync',
                 '2011_09_28_drive_0204_sync',
                 '2011_09_28_drive_0205_sync',
                 '2011_09_28_drive_0208_sync',
                 '2011_09_28_drive_0209_sync',
                 '2011_09_28_drive_0214_sync',
                 '2011_09_28_drive_0216_sync',
                 '2011_09_28_drive_0220_sync',
                 '2011_09_28_drive_0222_sync']

    with open('kitti_continuous_train (1).txt') as oldfile,open('kitti_continuous_train (2).txt','w') as newfile:
        for line in oldfile:
            if not any(bad_word in line for bad_word in bad_words):
                newfile.write(line)

    with open('kitti_continuous_test (1).txt') as oldfile,open('kitti_continuous_test (2).txt','w') as newfile:
        for line in oldfile:
            if not any(bad_word in line for bad_word in bad_words):
                newfile.write(line)

    timer1 += time.time()

else:

    timer1 = -time.time()

    try:

        def read_text_file(filename,dataset_path):
            print("\n[Dataloader] Loading '%s'..." % filename)
            try:
                data = np.genfromtxt(filename,dtype='str',delimiter='\t')
                # print(data.shape)

                # Parsing Data
                image_filenames = list(data[:,0])
                depth_filenames = list(data[:,1])

                timer = -time.time()
                image_filenames = [dataset_path + filename for filename in image_filenames]
                depth_filenames = [dataset_path + filename for filename in depth_filenames]
                timer += time.time()
                print('time:',timer,'s\n')

            except OSError:
                raise OSError("Could not find the '%s' file." % filename)

            return image_filenames,depth_filenames


        image_filenames,depth_filenames = read_text_file(
            filename='/media/olorin/Documentos/raul/SIDE/kitti_continuous_train (2).txt',
            dataset_path='/media/olorin/Documentos/datasets/kitti/raw_data/')

        image_validation,depth_validation = read_text_file(
            filename='/media/olorin/Documentos/raul/SIDE/kitti_continuous_test (2).txt',
            dataset_path='/media/olorin/Documentos/datasets/kitti/raw_data/')

        image = sorted(image_filenames)
        depth = sorted(depth_filenames)
        image_val = sorted(image_validation)
        depth_val = sorted(depth_validation)

        train_images = image
        train_labels = depth
        test_images = image_val
        test_labels = depth_val

        print(len(image))
        print(len(depth))

        skip_words = ['0000000005.png']

        skip = []
        skip_val = []

        for i in train_labels:
            if any(skip_word in i for skip_word in skip_words):
                skip.append(i)

        for j in test_labels:
            if any(skip_word in j for skip_word in skip_words):
                skip_val.append(j)

        print(len(skip))
        print(len(skip_val))

        timer1 += time.time()

    except OSError:
        raise SystemExit


# print(train_images)
print(len(train_images))

# print(train_labels)
print(len(train_labels))

# print(test_images[0:5])
print(len(test_images))

# print(test_labels[0:5])
print(len(test_labels))

print(timer1)

def fn_get_model_convLSTM_tframe_8(input_size=(None,128,416,3)):

    inputs = tf.keras.layers.Input(input_size)

    convlstm2 = tf.keras.layers.ConvLSTM2D(filters=64,kernel_size=(3,3),padding='same',return_sequences=True,
                                           activation='relu',recurrent_activation='hard_sigmoid',strides=(1,1),
                                           kernel_initializer='he_uniform')(inputs)
    BN2 = tf.keras.layers.BatchNormalization()(convlstm2)
    convlstm3 = tf.keras.layers.ConvLSTM2D(filters=64,kernel_size=(3,3),padding='same',return_sequences=True,
                                           activation='relu',recurrent_activation='hard_sigmoid',strides=(1,1),
                                           kernel_initializer='he_uniform')(BN2)
    BN3 = tf.keras.layers.BatchNormalization()(convlstm3)

    convlstm4 = tf.keras.layers.ConvLSTM2D(filters=128,kernel_size=(3,3),padding='same',return_sequences=True,
                                           activation='relu',recurrent_activation='hard_sigmoid',strides=(2,2),
                                           kernel_initializer='he_uniform')(BN3)
    BN4 = tf.keras.layers.BatchNormalization()(convlstm4)
    convlstm5 = tf.keras.layers.ConvLSTM2D(filters=128,kernel_size=(3,3),padding='same',return_sequences=True,
                                           activation='relu',recurrent_activation='hard_sigmoid',strides=(1,1),
                                           kernel_initializer='he_uniform')(BN4)
    BN5 = tf.keras.layers.BatchNormalization()(convlstm5)

    convlstm6 = tf.keras.layers.ConvLSTM2D(filters=256,kernel_size=(3,3),padding='same',return_sequences=True,
                                           activation='relu',recurrent_activation='hard_sigmoid',strides=(2,2),
                                           kernel_initializer='he_uniform')(BN5)
    BN6 = tf.keras.layers.BatchNormalization()(convlstm6)
    convlstm7 = tf.keras.layers.ConvLSTM2D(filters=256,kernel_size=(3,3),padding='same',return_sequences=True,
                                           activation='relu',recurrent_activation='hard_sigmoid',strides=(1,1),
                                           kernel_initializer='he_uniform')(BN6)
    BN7 = tf.keras.layers.BatchNormalization()(convlstm7)

    convlstm8 = tf.keras.layers.ConvLSTM2D(filters=512,kernel_size=(3,3),padding='same',return_sequences=True,
                                           activation='relu',recurrent_activation='hard_sigmoid',strides=(2,2),
                                           kernel_initializer='he_uniform')(BN7)
    BN8 = tf.keras.layers.BatchNormalization()(convlstm8)
    convlstm9 = tf.keras.layers.ConvLSTM2D(filters=512,kernel_size=(3,3),padding='same',return_sequences=True,
                                           activation='relu',recurrent_activation='hard_sigmoid',strides=(1,1),
                                           kernel_initializer='he_uniform')(BN8)
    BN9 = tf.keras.layers.BatchNormalization()(convlstm9)

    tconv0 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(1024,(3,3),activation='relu',padding='same',strides=(2,2),kernel_initializer='he_uniform'))(BN9)
    tBN0 = tf.keras.layers.BatchNormalization()(tconv0)
    tconv1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(1024,(3,3),activation='relu',padding='same',strides=(1,1),kernel_initializer='he_uniform'))(tBN0)
    tBN = tf.keras.layers.BatchNormalization()(tconv1)









    deconv = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(512,(2,2),activation='relu',strides=(2,2),padding='same',kernel_initializer='he_uniform'))(tBN)
    merge = tf.keras.layers.concatenate([convlstm9,deconv])
    conv = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=(1,1),kernel_initializer='he_uniform'))(merge)
    conv_ = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same',strides=(1,1),kernel_initializer='he_uniform'))(conv)


    deconv0 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(256,(2,2),activation='relu',strides=(2,2),padding='same',kernel_initializer='he_uniform'))(conv_)
    merge0 = tf.keras.layers.concatenate([convlstm7,deconv0])
    conv0 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',strides=(1,1),kernel_initializer='he_uniform'))(merge0)
    conv0_ = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same',strides=(1,1),kernel_initializer='he_uniform'))(conv0)


    deconv1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(128,(2,2),activation='relu',strides=(2,2),padding='same',kernel_initializer='he_uniform'))(conv0_)
    merge1 = tf.keras.layers.concatenate([convlstm5,deconv1])
    conv1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',strides=(1,1),kernel_initializer='he_uniform'))(merge1)
    conv1_ = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same',strides=(1,1),kernel_initializer='he_uniform'))(conv1)


    deconv2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(64,(2,2),activation='relu',strides=(2,2),padding='same',kernel_initializer='he_uniform'))(conv1_)
    merge2 = tf.keras.layers.concatenate([convlstm3,deconv2])
    conv2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',strides=(1,1),kernel_initializer='he_uniform'))(merge2)
    conv2_ = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same',strides=(1,1),kernel_initializer='he_uniform'))(conv2)
    conv2__ = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(2,(3,3),activation='relu',padding='same',strides=(1,1),kernel_initializer='he_uniform'))(conv2_)

    conv6 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=1,kernel_size=(1,1),activation='sigmoid',padding='same'))(conv2__)

    model = tf.keras.Model(inputs=inputs,outputs=conv6)

    return model

model = fn_get_model_convLSTM_tframe_8()

model.summary()

print('Training ...')

# initialize the number of epochs and batch size
EPOCHS = 45
seq_len = 3
lr = 0.0001

#-------------------

# model.load_weights('30_8_90163.44432.h5')

#-------------------

#TODO: Remove tf and insert tf.keras.backend

# ===========
#  Mask     #
# ===========
def tf_mask_out_invalid_pixels(tf_pred, tf_true):
    # Identify Pixels to be masked out.
    tf_idx = tf.where(tf_true > 0.0)  # Tensor 'idx' of Valid Pixel values (batchID, idx)

    # Mask Out Pixels without depth values
    tf_valid_pred = tf.gather_nd(tf_pred, tf_idx)
    tf_valid_true = tf.gather_nd(tf_true, tf_idx)

    return tf_valid_pred, tf_valid_true


# ------- #
#  BerHu  #
# ------- #
def tf_berhu_loss(y_true, y_pred):
    valid_pixels = True

    # C Constant Calculation
    tf_abs_error = tf.abs(y_pred - y_true, name='abs_error')
    tf_c = tf.multiply(tf.constant(0.2), tf.reduce_max(tf_abs_error))  # Consider All Pixels!

    # Mask Out
    if valid_pixels:
        # Overwrites the 'y' and 'y_' tensors!
        y_pred, y_true = tf_mask_out_invalid_pixels(y_pred, y_true)

        # Overwrites the previous tensor, so now considers only the Valid Pixels!
        tf_abs_error = tf.abs(y_pred - y_true, name='abs_error')

    # Loss
    tf_berhu_loss = tf.where(tf_abs_error <= tf_c, tf_abs_error,
                             tf.div((tf.square(tf_abs_error) + tf.square(tf_c)), tf.multiply(tf.constant(2.0), tf_c)))

    tf_loss = tf.reduce_sum(tf_berhu_loss)

    return tf_loss

model.compile(loss=tf_berhu_loss,optimizer=tf.keras.optimizers.Nadam(lr=lr))

class WeightHistory(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        args = vars(parser.parse_args())
        if args['bar'] == '1':
            model.save_weights('weights_convlstm2_berhu_#5_%s.h5' % batch)

whistory = WeightHistory()

# cbk2 = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=2, verbose=0)

cbk3 = keras.callbacks.ModelCheckpoint(filepath='convlstm2_berhu_#5_{epoch:d}_{val_loss:.5f}.h5',save_weights_only=False,
                                monitor='val_loss',save_best_only=True,verbose=0,period=1)


model.fit_generator(imageLoader(train_images,train_labels,skip,seq_len),
                        steps_per_epoch=((len(train_images))-((seq_len-1)*58)), #58 -> 116/2 -> is the number of train files in the dataset
                        epochs=EPOCHS,
                        validation_data=imageLoader(test_images,test_labels,skip_val,seq_len),
                        validation_steps=((len(test_images))-((seq_len-1)*13)), #13 -> 26/2 -> is the number of test files in the dataset
                        callbacks=[whistory,cbk3])#, cbk2])


# ----- Save ----- #
# if saveModel:
model.save_weights('weights_convlstm2_berhu_#5F_.h5')
model.save('model_convlstm2_berhu_#5F_.h5')

print("Done.")

