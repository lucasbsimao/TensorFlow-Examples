import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random
import shutil
import math
from DC_Utils import *
from Classifier import *

mb_size = 64
total_img = 50000
X_dim = 784
z_dim = 10
h_dim = 128

df_dim = 64
gf_dim = 64
epsilon = 1e-3

def batch_norm_wrapper(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

def convert_to_mat(X_mb):
    select_batch = np.zeros([len(X_mb),X_dim])
    for i in range(len(X_mb)):
        select_batch[i,:] = np.asmatrix(X_mb[i])

    return select_batch

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

X_mb, Y_mb = mnist.train.next_batch(total_img)
'''
Bd_matlab = []
for i in range(total_img):
    tempy = Y_mb[i,:]
    if np.argmax(tempy) == 1:
        tempx = X_mb[i,:]
        Bd_matlab.append(tempx)

X_feed = convert_to_mat(Bd_matlab)

Bd_matlab = []
for i in range(total_img):
    tempy = Y_mb[i,:]
    if np.argmax(tempy) == 4:
        tempx = X_mb[i,:]
        Bd_matlab.append(tempx)

X_sh = convert_to_mat(Bd_matlab)

Bd_matlab = []
for i in range(total_img):
    tempy = Y_mb[i,:]

    if np.argmax(tempy) == 8:
        tempx = X_mb[i,:]
        Bd_matlab.append(tempx)

X_rp = convert_to_mat(Bd_matlab)

mnist = np.concatenate((X_feed, X_sh, X_rp))

it_feed = 0
it_sh = 0
it_rp = 0

for i in range(len(mnist)):
    bd = np.random.randint(0,3)

    passed = False
    if bd == 0:
        if len(X_feed) != (it_feed):
            mnist[i] = X_feed[it_feed]
            it_feed += 1
        else:
            passed = True

    if bd == 1:
        if len(X_sh) != (it_sh):
            mnist[i] = X_sh[it_sh]
            it_sh += 1
        else:
            passed = True

    if bd == 2:
        if len(X_rp) != (it_rp):
            mnist[i] = X_rp[it_rp]
            it_rp += 1
        else:
            passed = True

    if passed:
        i -= 1

#np.random.shuffle(mnist)
'''
def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        
    #plt.show()
    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])



D_W1 = tf.Variable(xavier_init([5,5,1, df_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[df_dim]))

D_W2 = tf.Variable(xavier_init([5,5,df_dim, df_dim*2]))
D_b2 = tf.Variable(tf.zeros(shape=[df_dim*2]))

D_W3 = tf.Variable(xavier_init([5,5,df_dim*2, df_dim*4]))
D_b3 = tf.Variable(tf.zeros(shape=[df_dim*4]))

D_W4 = tf.Variable(xavier_init([5,5,df_dim*4, df_dim*8]))
D_b4 = tf.Variable(tf.zeros(shape=[df_dim*8]))

D_W5 = tf.Variable(xavier_init([4096, 1]))
#D_W5 = tf.Variable(xavier_init([2048, 1]))
D_b5 = tf.Variable(tf.zeros(shape=[1]))

#theta_D = [D_W1, D_W2, D_W3, D_W4, D_W5, D_b1, D_b2, D_b3, D_b4, D_b5]
theta_D = [D_W1, D_W2, D_W3, D_W5, D_b1, D_b2, D_b3, D_b5]

s_h, s_w = 28, 28
s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, 4*gf_dim*s_h8*s_w8]))
G_b1 = tf.Variable(tf.zeros(shape=[4*gf_dim*s_h8*s_w8]))

#G_W1 = tf.Variable(xavier_init([z_dim, 8*gf_dim*s_h16*s_w16]))
#G_b1 = tf.Variable(tf.zeros(shape=[8*gf_dim*s_h16*s_w16]))

#G_W2 = tf.Variable(xavier_init([5,5, 4*gf_dim , 8*gf_dim]))
#G_b2 = tf.Variable(tf.zeros(shape=[4*gf_dim]))

G_W3 = tf.Variable(xavier_init([5,5, 2*gf_dim , 4*gf_dim]))
G_b3 = tf.Variable(tf.zeros(shape=[2*gf_dim]))

G_W4 = tf.Variable(xavier_init([5,5, gf_dim , 2*gf_dim]))
G_b4 = tf.Variable(tf.zeros(shape=[gf_dim]))

G_W5 = tf.Variable(xavier_init([5, 5, 1, gf_dim]))
G_b5 = tf.Variable(tf.zeros(shape=[1]))

#theta_G = [G_W1, G_W2, G_W3, G_W4, G_W5, G_b1, G_b2, G_b3, G_b4, G_b5]
theta_G = [G_W1, G_W3, G_W4, G_W5, G_b1, G_b3, G_b4, G_b5]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def sample_z_unit():
    return np.identity(10)

def generator(z, is_training):
    G_h1 = (tf.matmul(z, G_W1) + G_b1)
    G_h1 = tf.reshape(G_h1, [tf.shape(z)[0], s_h8, s_w8, gf_dim*4])
    G_h1 = tf.nn.relu(tf.layers.batch_normalization(G_h1,training=is_training))

    #G_h1 = (tf.matmul(z, G_W1) + G_b1)
    #G_h1 = tf.reshape(G_h1, [tf.shape(z)[0], s_h4, s_w4, gf_dim*2])
    #G_h1 = tf.nn.relu(tf.layers.batch_normalization(G_h1,training=is_training))

    '''G_h2 = tf.nn.conv2d_transpose(G_h1, G_W2, output_shape=[tf.shape(G_h1)[0], s_h8, s_w8, gf_dim*4],strides=[1, 2, 2, 1])
    G_h2 = tf.nn.bias_add(G_h2, G_b2)
    G_h2 = tf.nn.relu(batch_norm_wrapper(G_h2,is_training))'''

    G_h3 = tf.nn.conv2d_transpose(G_h1, G_W3, output_shape=[tf.shape(G_h1)[0], s_h4, s_w4, gf_dim*2],strides=[1, 2, 2, 1])
    G_h3 = tf.nn.bias_add(G_h3, G_b3)
    G_h3 = tf.nn.relu(tf.layers.batch_normalization(G_h3,training=is_training))

    G_h4 = tf.nn.conv2d_transpose(G_h3, G_W4, output_shape=[tf.shape(G_h3)[0], s_h2, s_w2, gf_dim],strides=[1, 2, 2, 1])
    G_h4 = tf.nn.bias_add(G_h4, G_b4)
    G_h4 = tf.nn.relu(tf.layers.batch_normalization(G_h4,training=is_training))

    G_h5 = tf.nn.conv2d_transpose(G_h4, G_W5, output_shape=[tf.shape(G_h4)[0], s_h, s_w, 1],strides=[1, 2, 2, 1])
    G_h5 = tf.nn.bias_add(G_h5, G_b5)

    G_h5 = tf.nn.sigmoid(G_h5)
    return G_h5

def discriminator(x, is_training):
    D_h1 = tf.nn.conv2d(x, D_W1,strides=[1,2,2,1],padding="SAME")
    D_h1 = lrelu((tf.nn.bias_add(D_h1, D_b1 )))

    D_h2 = tf.nn.conv2d(D_h1, D_W2,strides=[1,2,2,1],padding="SAME")
    D_h2 = lrelu(tf.layers.batch_normalization(tf.nn.bias_add(D_h2, D_b2 ), training=is_training))

    D_h3 = tf.nn.conv2d(D_h2, D_W3,strides=[1,2,2,1],padding="SAME")
    D_h3 = lrelu(tf.layers.batch_normalization(tf.nn.bias_add(D_h3, D_b3 ), training=is_training))
    '''
    D_h4 = tf.nn.conv2d(D_h3, D_W4,strides=[1,2,2,1],padding="SAME")
    D_h4 = lrelu(tf.layers.batch_normalization(tf.nn.bias_add(D_h4, D_b4 ), training=is_training))'''

    #D_h5 = tf.reshape(D_h2,[tf.shape(D_h2)[0],-1])
    D_h5 = tf.reshape(D_h3,[tf.shape(D_h3)[0],-1])
    D_h5 = tf.matmul(D_h5, D_W5) + D_b5
    out = tf.sigmoid(D_h5)
    return out, D_h5
                        
G_sample = generator(z, True)
D_real, D_logits_r = discriminator(X, True)
D_fake, D_logits_f = discriminator(G_sample, True)

D_loss = tf.reduce_mean(D_logits_r) - tf.reduce_mean(D_logits_f)
G_loss = -tf.reduce_mean(D_logits_f)

D_solver = (tf.train.RMSPropOptimizer(learning_rate=5e-5)
            .minimize(-D_loss, var_list=theta_D))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=5e-5)
            .minimize(G_loss, var_list=theta_G))

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver1 = tf.train.Saver([w, w2, w3, w4,w_o])

saver1.restore(sess, "./Classification_Model/Model-D.ckpt")

if not os.path.exists('WGAN_Test/out_samp'):
    os.makedirs('WGAN_Test/out_samp/')
else:
    shutil.rmtree('WGAN_Test/out_samp')
    os.makedirs('WGAN_Test/out_samp/')

if not os.path.exists('WGAN_Test/out_orig/'):
    os.makedirs('WGAN_Test/out_orig/')
else:
    shutil.rmtree('WGAN_Test/out_orig')
    os.makedirs('WGAN_Test/out_orig/')


def next_batch(mb_size, X_mb):
    select_batch = np.zeros([mb_size,784])
    for i in range(mb_size):
        select_batch[i] = random.choice(X_mb[:])

    return select_batch.reshape(mb_size,28,28,1)
i=0
k = 1
num_samples = 1000

probabilities = np.zeros([(num_samples/500),10])

cont_num = np.zeros(10)
for it in range(1000000):
    X_feed = np.zeros([mb_size,28,28,1])
    for _ in range(k):
        X_feed = mnist.train.next_batch(mb_size)
        X_feed = np.reshape(X_feed,[mb_size,28,28,1])

        _, D_loss_curr = sess.run(
            [D_solver, D_loss],
            feed_dict={X: X_feed, z: sample_z(mb_size, z_dim)}
        )

    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={z: sample_z(mb_size, z_dim)}
    )

    if it % 100 == 0:
        print('Iter BD: {}; D loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss_curr, G_loss_curr))

    if it % 500 == 0:
        samples = sess.run(G_sample, feed_dict={z: sample_z(64, z_dim)})

        for i in range(num_samples):
            samples_t = sess.run(G_sample, feed_dict={z: sample_z(1, z_dim)})

            index = sess.run(predict_op, feed_dict={X: samples_t, p_keep_hidden: 1.0})
            print(i, index[0])

            cont_num[index[0]] += 1
        
        probabilities[it/500,:] = float(cont_num)/float(num_samples)

        fig = plot(samples)
        plt.savefig('GAN_Test/out_samp/{}.png'
                    .format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

        fig = plot(X_feed[0:mb_size])
        plt.savefig('GAN_Test/out_orig/{}.png'
                    .format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
        savePath = saver.save(sess, "WGAN_Model/Model.ckpt")
        print("Model saved in file: %s" % savePath)

        print(probabilities[:])

    '''if it % 2000 == 0:
        plt.plot(np.linspace(0,(it/500)),probabilities[])
        plt.show()'''