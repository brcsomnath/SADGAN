
import tensorflow as tf
import random as r
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from scipy.misc import imread
from scipy.misc import imresize

IMAGE_HEIGHT  = 224
IMAGE_WIDTH   = 224
NUM_CHANNELS  = 3
BATCH_SIZE    = 100

def read_label_file(file1, file2):
    f = open(file1, "r")
    filepaths1 = []
    for line in f:
        filepath = line
        filepaths1.append(filepath[:-1])


    f = open(file2, "r")
    filepaths2 = []
    for line in f:
        filepath = line
        filepaths2.append(filepath[:-1])

    x = []
    z = []
    for i in range(len(filepaths1)):
        filepath = filepaths1[i]
        filepath = filepath[:-6] + "_2.png"
        if filepath in filepaths2:
            idx = filepaths2.index(filepath)
            x.append(filepaths1[i])
            z.append(filepaths2[idx])
    return x, z



def mark_label(x):
    if x == 'L':
        return 0
    elif x == 'R':
        return 1
    else:
        return 2

def Dataset():

    
    # reading labels and file path
    train_X, train_Z = read_label_file('train_before.txt','train_after.txt')

    train_path = "train/"


    train_filepathsX = [ train_path + fp for fp in train_X]
    train_filepathsZ = [ train_path + fp for fp in train_Z]
    train_label = [mark_label(fp[0]) for fp in train_X]


    all_imagesX = ops.convert_to_tensor(train_filepathsX, dtype=dtypes.string)
    all_imagesZ = ops.convert_to_tensor(train_filepathsZ, dtype=dtypes.string)
    all_labels = ops.convert_to_tensor(train_label, dtype=dtypes.float32)

    train_input_queue = tf.train.slice_input_producer([all_imagesX, all_imagesZ, all_labels], shuffle=True)

    file_content = tf.read_file(train_input_queue[0])
    image = tf.image.decode_png(file_content, channels=NUM_CHANNELS)
    train_imageX = tf.image.resize_images(image, [224, 224])

    file_content = tf.read_file(train_input_queue[1])
    image = tf.image.decode_png(file_content, channels=NUM_CHANNELS)
    train_imageZ = tf.image.resize_images(image, [224, 224])

    label_train = train_input_queue[2]

    train_imageX.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    train_imageZ.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

    train_image_batchX, train_image_batchZ, train_labels = tf.train.batch([train_imageX, train_imageZ, label_train],
                                                            allow_smaller_final_batch=True,
                                                             batch_size=BATCH_SIZE)
    train_labels =  tf.reshape(train_labels, [BATCH_SIZE,1])
    print train_labels.shape
    print "input pipeline ready"

    return train_image_batchX, train_image_batchZ, train_labels
