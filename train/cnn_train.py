import tensorflow as tf

from .layer import *
from .dataset import DataSet

import os
import datetime
import time


class Lenet(object):
    def __init__(self, params):
        self.batch_size = params['batch_size']
        self.image_size = params['image_size']
        self.prob = params['prob']
        self.learning_rate = params['lr']
        self.max_steps = params['max_steps']
        self.log_dir = params['log_dir']
        print("bs: {}, img_size: {}, prob: {}, lr: {}, max_steps: {}".format(
            self.batch_size, self.image_size, self.prob, self.learning_rate, self.max_steps
        ))

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.num_classes = 65

        self.x = None
        self.y = None
        self.keep_prob = None

        self.pred = None
        self.loss = None
        self.total_loss = None

        self.global_step = tf.Variable(0)
        self.saver = tf.train.Saver(max_to_keep=5)

        self.sess = None

    def compile(self):
        self.x = tf.placeholder(tf.float32, (self.batch_size, self.image_size, self.image_size, 1))
        self.y = tf.placeholder(tf.int64, shape=(self.batch_size,))
        self.keep_prob = tf.placeholder(tf.float32)

        conv1_weights = weight_variable([5, 5, 1, 32])  # 5x5 filter, depth 32.
        conv1_biases = bias_variable([32])
        conv2_weights = weight_variable([5, 5, 32, 64])
        conv2_biases = bias_variable([64])

        fc1_weights = weight_variable([25 * 64, 512])
        fc1_biases = bias_variable([512])
        fc2_weights = weight_variable([512, self.num_classes])
        fc2_biases = bias_variable([self.num_classes])

        def model(data):
            conv1 = conv2d(data, conv1_weights)
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
            pool1 = max_pool(relu1)

            conv2 = conv2d(pool1, conv2_weights)
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
            pool2 = max_pool(relu2)

            pool_shape = pool2.get_shape().as_list()
            reshape = tf.reshape(pool2, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

            hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

            hidden = tf.nn.dropout(hidden, self.keep_prob)

            return tf.matmul(hidden, fc2_weights) + fc2_biases

        logits = model(self.x)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=self.y))

        regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                        tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))

        self.total_loss = self.loss + 5e-4 * regularizers
        self.pred = tf.argmax(logits, 1)
        correct_pred = tf.equal(self.pred, self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.__add_optimal()

    def __add_optimal(self):
        optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        train_op = optimizer.minimize(self.total_loss, global_step=self.global_step)

        self.train_op = train_op

    def train(self, train_dataset, val_dataset):
        if self.sess is None:
            self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(self.log_dir + "train/summary/", self.sess.graph)
        model_dir = self.log_dir + 'models/'

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model restored...", ckpt.model_checkpoint_path)

        for step in range(int(self.max_steps)):

            train_x, train_y = train_dataset.batch()
            feed_dict = {self.x: train_x, self.y: train_y, self.keep_prob: 0.5}
            self.sess.run(self.train_op, feed_dict=feed_dict)

            if step % 10 == 0:
                train_loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
                print("Step: %d, Train_loss: %g, acc: %g" % (step, train_loss, acc))

            if step % 100 == 0:
                val_x, val_y = val_dataset.batch()
                feed_dict = {self.x: val_x, self.y: val_y, self.keep_prob: 1}
                valid_loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
                print("%s ---> Validation_loss: %g, acc: %g" % (datetime.datetime.now(), valid_loss, acc))

            if step % 5000 == 0:
                print('Saving checkpoint: ', step)
                self.saver.save(self.sess, model_dir + "model.ckpt", step)

        train_writer.close()

    def predict(self, test_images):

        assert(test_images.ndim == 3 or test_images.ndim == 4)
        if test_images.ndim == 3:
            test_images = test_images[None]

        assert(test_images.ndim == 4)

        if self.sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            model_dir = self.log_dir + 'models/'
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Model restored...", ckpt.model_checkpoint_path)

        starttime = time.time()
        pred = self.sess.run(self.pred, feed_dict={self.x: test_images,
                                                              self.keep_prob: 1.0})
        endtime = time.time()
        print('Predict done: time {:.4f} sec'.format(endtime - starttime))
        return pred, endtime - starttime

    def close(self):
        self.sess.close()

def main():
    batch_size = 32
    dataset_params = {
        'batch_size': batch_size,
        'path': 'resources/train_data/chars',
        'labels_path': 'resources/train_data/chars_list_train.pickle',
        'thread_num': 3
    }
    train_dataset_reader = DataSet(dataset_params)
    dataset_params['labels_path'] = 'resources/train_data/chars_list_val.pickle'
    #dataset_params['batch_size'] = -1
    val_dataset_reader = DataSet(dataset_params)

    params = {
        'image_size': 20,
        'batch_size': batch_size,
        'prob': 0.5,
        'lr': 0.01,
        'max_steps': train_dataset_reader.num_batch_per_epoch * 20,
        'log_dir': 'train/model/chars/'
    }

    model = Lenet(params)
    model.compile()
    model.train(train_dataset_reader, val_dataset_reader)

