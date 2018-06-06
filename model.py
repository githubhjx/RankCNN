
from skimage import io, transform
from scipy.io import loadmat
from scipy.special import comb
import glob
import os
import sys
import shutil
import numpy as np
import tensorflow as tf

sys.setrecursionlimit(1000000)

# decision  parameter
N_LABEL = 2  # Number of classes
N_BATCH = 16  # Number of data points per mini-batch

# decision picture parameter
w = 224
h = 224
c = 3

tr_path = 'E:/CK_4class/CK_new/angry/train/'
te_path = 'E:/CK_4class/CK_new/angry/test/'

log_path = 'E:/CK_4class/log/'
model_path = 'E:/CK_4class/model/'


##################################################
# Load data
##################################################
def read_img1(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    cate.sort()
    imgs0 = []
    imgs1 = []
    labels = []
    for idx, folder in enumerate(cate):

        data = []
        data0_0 = []
        data1_1 = []
        label = []

        L = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
        temp = glob.glob(folder + '/*.png')
        for im in range(0, L, 3):
            print('reading the images:%s' % (temp[im]))
            img = io.imread(temp[im])
            img = transform.resize(img, (w, h))
            data.append(img)

        # rank data
        label0 = [1, 0]  # negative label
        label1 = [0, 1]  # positive label
        for i in range(len(data)):
            dat = data[i]
            for j in range(len(data)):
                if j > i:
                    data0_0.append(dat)  # negative sample
                    data1_1.append(data[j])
                    # data0_0.append(data[j])
                    # data1_1.append(dat)  # positive sample
                    label.append(label1)
                    # label.append(label0)
        # data0 = data0_0.copy()  # negative sample
        # data0.extend(data1_1)
        #
        # data1 = data1_1  # positive sample
        # data1.extend(data0_0)

        # make label
        # N = int(comb(len(data), 2))
        # label0 = [1, 0]  # negative label
        # label1 = [0, 1]  # positive label
        # label.extend([label1] * N)
        # labels_op = [label0] * N
        # label.extend(labels_op)

        # push data to img and label
        imgs0.extend(data0_0)
        imgs1.extend(data1_1)
        labels.extend(label)
        # label[idx] = 1
        # labels.append(label)
    return np.asarray(imgs0, np.float32), np.asarray(imgs1, np.float32), np.asarray(labels, np.float32)


def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    cate.sort()
    imgs0 = []
    imgs1 = []
    labels = []
    for idx, folder in enumerate(cate):

        data = []
        data0_0 = []
        data1_1 = []
        label = []

        L = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
        temp = glob.glob(folder + '/*.png')
        for im in range(0, L, 3):
            print('reading the images:%s' % (temp[im]))
            img = io.imread(temp[im])
            img = transform.resize(img, (w, h))
            data.append(img)

        # rank data
        label0 = [1, 0]  # negative label
        label1 = [0, 1]  # positive label
        for i in range(len(data)):
            dat = data[i]
            for j in range(len(data)):
                if j > i:
                    data0_0.append(dat)  # negative sample
                    data1_1.append(data[j])
                    data0_0.append(data[j])
                    data1_1.append(dat)  # positive sample
                    label.append(label1)
                    label.append(label0)
        # data0 = data0_0.copy()  # negative sample
        # data0.extend(data1_1)
        #
        # data1 = data1_1  # positive sample
        # data1.extend(data0_0)

        # make label
        # N = int(comb(len(data), 2))
        # label0 = [1, 0]  # negative label
        # label1 = [0, 1]  # positive label
        # label.extend([label1] * N)
        # labels_op = [label0] * N
        # label.extend(labels_op)

        # push data to img and label
        imgs0.extend(data0_0)
        imgs1.extend(data1_1)
        labels.extend(label)
        # label[idx] = 1
        # labels.append(label)
    return np.asarray(imgs0, np.float32), np.asarray(imgs1, np.float32), np.asarray(labels, np.float32)


trX0, trX1, trY = read_img(tr_path)
teX0, teX1, teY = read_img1(te_path)

# num_example = teX0.shape[0]
# arr = np.arange(num_example)
# np.random.shuffle(arr)
# teX0 = teX0[arr]
# teX1 = teX1[arr]
# teY = teY[arr]


trX0 = trX0.reshape(-1, 224, 224, 3)
trX1 = trX1.reshape(-1, 224, 224, 3)
teX0 = teX0.reshape(-1, 224, 224, 3)
teX1 = teX1.reshape(-1, 224, 224, 3)


###################################
# Input X, output Y
###################################
X0 = tf.placeholder("float", [N_BATCH, 224, 224, 3], name="input_X0")
X1 = tf.placeholder("float", [N_BATCH, 224, 224, 3], name="input_X1")
# X2 = tf.placeholder("float", [N_BATCH, 224, 224, 3], name="input_X2")
# X3 = tf.placeholder("float", [N_BATCH, 224, 224, 3], name="input_X3")
Y = tf.placeholder("float", [N_BATCH, N_LABEL], name="input_Y")


#######################################
# init weights function
#######################################
def extract_weights(path):
    kernels = []
    data = loadmat(path)
    layers = data['layers']
    for layer in layers[0]:
        layer_type = layer[0]['type'][0][0]
        if layer_type == 'conv':
            kernel, bias = layer[0]['weights'][0][0]
            kernels.append(kernel)
    return kernels


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def init_prob_weights(shape, minval=-5, maxval=5):
    return tf.Variable(tf.random_uniform(shape, minval, maxval))


global kernels
kernels = extract_weights('vgg-face.mat')


def conv_op(input_op, name, kh, kw, n_out, dh, dw, p, wp):
    with tf.name_scope(name) as scope:
        kernel = kernels.copy()
        kernel = tf.Variable(kernel[wp], trainable=False, name="conv_kernel")
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=False, name="conv_b")
        # biases =bias[wp]
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation


def fc_op(input_op, name, n_out, p, wp):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel_1 = kernels.copy()
        kernel = tf.Variable(kernel_1[wp], trainable=True)
        kernel = tf.reshape(kernel, shape=[n_in, n_out], name="fc_kernel")
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), trainable=True, name="fc_b")
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation


def fc_op8(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 trainable=True)
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), trainable=True, name="b")
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation


def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)


def inference_op(input_op, keep_prob):
    p = []
    # assume input_op shape is 224x224x3

    # block 1 -- outputs 112x112x64
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p, wp=0)
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p, wp=1)
    pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)

    # block 2 -- outputs 56x56x128
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p, wp=2)
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p, wp=3)
    pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # # block 3 -- outputs 28x28x256
    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p, wp=4)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p, wp=5)
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p, wp=6)
    pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)

    # block 4 -- outputs 14x14x512
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p, wp=7)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p, wp=8)
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p, wp=9)
    pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

    # block 5 -- outputs 7x7x512
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p, wp=10)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p, wp=11)
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p, wp=12)
    pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)

    # flatten
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    # fully connected
    fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p, wp=13)
    # fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")
    return fc6, p


keep_prob = tf.placeholder("float", name="keep_prob")
fc6_0, p1 = inference_op(X0, keep_prob)
fc6_1, p2 = inference_op(X0, keep_prob)
fc6_0_0 = tf.nn.l2_normalize(fc6_0, dim=0)
fc6_1_1 = tf.nn.l2_normalize(fc6_1, dim=0)
fc6_c = fc6_1_1 - fc6_0_0
# fc6_c_n = tf.norm(fc6_c, ord='euclidean')
fc6 = tf.nn.l2_normalize(fc6_c, dim=0)


def difference_op(input_op, keep_prob, p):
    fc7 = fc_op(input_op, name="fc7", n_out=4096, p=p, wp=14)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    fc8 = fc_op8(fc7_drop, name="fc8", n_out=N_LABEL, p=p)

    softmax = tf.nn.softmax(fc8, name="softmax")

    predictions = tf.argmax(softmax, 1, name="predictions")

    return predictions, softmax, fc8, fc7, p


predictions, softmax, fc8, fc7, p = difference_op(fc6, keep_prob, p2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=fc8))

train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
predict = predictions

tf.add_to_collection("fc7", fc7)
tf.add_to_collection("fc8", fc8)
tf.add_to_collection("train_step", train_step)
tf.add_to_collection("predict", predict)
tf.add_to_collection("X0", X0)
tf.add_to_collection("X1", X1)
# tf.add_to_collection("X2", X2)
# tf.add_to_collection("X3", X3)
tf.add_to_collection("Y", Y)
tf.add_to_collection("keep_prob", keep_prob)

###################################################
# Train and Test
###################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

maxAcy = 0
for step in range(10000):
    # One epoch
    costs = []
    for start, end in zip(range(0, len(trX0), N_BATCH), range(N_BATCH, len(trX0), N_BATCH)):
        _, c = sess.run([train_step, cost], feed_dict={X0: trX0[start:end], X0: trX1[start:end], Y: trY[start:end], keep_prob: 1.0})
    costs.append(c)

    # Result on the test set
    results = []
    for start, end in zip(range(0, len(teX0), N_BATCH), range(N_BATCH, len(teX0), N_BATCH)):
        results.extend(np.argmax(teY[start:end], axis=1) ==
                       sess.run(predict, feed_dict={X0: teX0[start:end], X0: teX1[start:end], keep_prob: 1.0}))

        print(sess.run(predict, feed_dict={X0: teX0[start:end], X1: teX1[start:end], keep_prob: 1.0}))
    print('Epoch: %d, Test Accuracy: %f, cost: %f' % (step + 1, np.mean(results), np.mean(costs)))

    # now accuracy
    nowAcy = np.mean(results)

    if nowAcy > maxAcy:
        maxAcy = nowAcy

        # check path exists
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            os.mkdir(model_path)
        else:
            os.mkdir(model_path)

        saver.save(sess, model_path + 'model.ckpt')

        writer = tf.summary.FileWriter(log_path, tf.get_default_graph())
        writer.close()

# saver.save(sess, model_path + 'model.ckpt')
# writer = tf.summary.FileWriter(log_path, tf.get_default_graph())
# writer.close()

sess.close()
