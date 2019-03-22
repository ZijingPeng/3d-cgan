from src.layers import *
import pickle as pkl
from src.preprocess import *
import numpy as np
import tensorflow as tf
import codecs
import json
import glob


class Stage1:
    def __init__(self, view_shape=(128, 128, 1), model_shape=(32, 32, 32, 1), z_size=100, channel=1, learning_rate=0.0002, alpha=0.2, beta1=0.5):

        tf.reset_default_graph()

        self.input_views, self.input_models, self.input_z = self.model_inputs(view_shape, model_shape, z_size)

        self.d_loss, self.g_loss, self.g_models = self.model_loss(self.input_views, self.input_models, self.input_z, channel, alpha=alpha)

        self.d_opt, self.g_opt = self.model_opt(self.d_loss, self.g_loss, learning_rate, beta1)



    # train cgan
    # save_path to save checkpoints
    # use sess-path if restore checkpoints
    def train(self, dataset, epochs, save_path="../ckp/stage1/", sess_path=None, z_size=100, batch_size=32, print_every=10, show_every=200):
        saver = tf.train.Saver()

        samples, losses = [], []
        steps = 0

        with tf.Session() as sess:
            # if it is restore
            if sess_path is None:
                sess.run(tf.global_variables_initializer())
            else:
                saver = tf.train.import_meta_graph("{}.meta".format(sess_path))
                saver.restore(sess, sess_path)

            for e in range(epochs):
                for x, y in dataset.batches(batch_size):
                    steps += 1
                    batch_z = np.random.normal(-0.1, 0.1, size=(x.shape[0], z_size))
                    _ = sess.run(self.d_opt, feed_dict={self.input_views: x, self.input_models: y, self.input_z: batch_z})
                    _ = sess.run(self.g_opt, feed_dict={self.input_views: x, self.input_models: y, self.input_z: batch_z})
                    if steps % print_every == 0:
                        train_loss_d = self.d_loss.eval({self.input_views: x, self.input_models: y, self.input_z: batch_z})
                        train_loss_g = self.g_loss.eval({self.input_views: x, self.input_models: y, self.input_z: batch_z})

                        print("Epoch {}/{}...".format(e + 1, epochs),
                              "Discriminator Loss: {}...".format(train_loss_d),
                              "Generator Loss: {}".format(train_loss_g))

                    # display when train if you want
                    '''
                    if steps % show_every == 0:
                        gen_samples = sess.run(
                            generate_shape(self.input_views, channel, reuse=True, training=False),
                            feed_dict={self.origin_pic: x}
                        )
                        samples.append(gen_samples)
                        _ = display_view(gen_samples[0])
                    '''

            saver.save(sess, '{}dm0.ckpt'.format(save_path))

        sess.close()

        return losses, samples


    # restore the network and show the results of input pics
    # pics should be a list of pics
    # num is the numbers of pics to show
    def restore(self, pics, num, sess_path, z_size=100, channel=1):
        saver = tf.train.Saver()
        sample_z = np.random.normal(-0.01, 0.01, size=(1, z_size))

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("{}.meta".format(sess_path))
            saver.restore(sess, sess_path)
            gen_samples = sess.run(
                generate_shape(self.input_z, self.input_views, channel, reuse=True, training=False),
                feed_dict={self.input_z: sample_z, self.input_views: pics}
            )

            return gen_samples


    def model_inputs(self, view_shape, model_shape, z_size):
        input_views = tf.placeholder(tf.float32, (None, *view_shape), name='input_views_stage1')
        input_models = tf.placeholder(tf.float32, (None, *model_shape), name='input_models_stage1')
        input_z = tf.placeholder(tf.float32, (None, z_size), name='input_z_stage1')

        return input_views, input_models, input_z


    def model_loss(self, input_views, input_models, input_z, channel, alpha=0.2):
        g_models = generate_shape(input_z, input_views, channel, alpha=alpha)

        d_model_real, d_logits_real = discriminate_shape(input_views, input_models, alpha=alpha)
        d_model_fake, d_logits_fake = discriminate_shape(input_views, g_models, reuse=True, alpha=alpha)

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))
        g_loss_similar = tf.reduce_mean(tf.square(g_models - input_models))

        d_loss = d_loss_real + d_loss_fake
        g_loss = g_loss * 0.3 + g_loss_similar

        return d_loss, g_loss, g_models


    def model_opt(self, d_loss, g_loss, learning_rate, beta1):
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminate_shape')]
        g_vars = [var for var in t_vars if var.name.startswith('generate_shape')]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
            g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

        return d_train_opt, g_train_opt


class Stage2:
    def __init__(self, view_shape=(32, 32, 3), model_shape=(32, 32, 32, 3), z_size=100, channel=3, learning_rate=0.0002, alpha=0.2, beta1=0.5):

        tf.reset_default_graph()

        self.input_views, self.input_models, self.input_z = self.model_inputs(view_shape, model_shape, z_size)

        self.d_loss, self.g_loss = self.model_loss(self.input_views, self.input_models, self.input_z, channel, alpha=alpha)

        self.d_opt, self.g_opt = self.model_opt(self.d_loss, self.g_loss, learning_rate, beta1)



    # train cgan
    # save_path to save checkpoints
    # use sess-path if restore checkpoints
    def train(self, dataset, epochs, save_path="../ckp/stage2/", sess_path=None, batch_size=32, z_size=100, print_every=10, show_every=200):

        saver = tf.train.Saver()

        samples, losses = [], []
        steps = 0

        with tf.Session() as sess:

            # if it is restore
            if sess_path is None:
                sess.run(tf.global_variables_initializer())
            else:
                saver = tf.train.import_meta_graph("{}.meta".format(sess_path))
                saver.restore(sess, sess_path)

            for e in range(epochs):
                for x, y in dataset.batches(batch_size):
                    steps += 1
                    batch_z = np.random.normal(-0.1, 0.1, size=(x.shape[0], z_size))

                    _ = sess.run(self.d_opt, feed_dict={self.input_views: x, self.input_models: y, self.input_z: batch_z})
                    _ = sess.run(self.g_opt, feed_dict={self.input_views: x, self.input_models: y, self.input_z: batch_z})
                    if steps % print_every == 0:
                        train_loss_d = self.d_loss.eval({self.input_views: x, self.input_models: y, self.input_z: batch_z})
                        train_loss_g = self.g_loss.eval({self.input_views: x, self.input_models: y, self.input_z: batch_z})

                        print("Epoch {}/{}...".format(e + 1, epochs),
                              "Discriminator Loss: {}...".format(train_loss_d),
                              "Generator Loss: {}".format(train_loss_g))

                    # display when train if you want
                    '''
                    if steps % show_every == 0:
                        
                        gen_pics = sess.run(
                            generate_pixel(self.stage1.origin_pic, 1, reuse=True, training=False),
                            feed_dict={self.stage1.origin_pic: x}
                        )
                    '''

            saver.save(sess, '{}dm0.ckpt'.format(save_path))
            sess.close()

        return losses, samples


    # restore the network and show the results of input pics
    # pics should be a list of pics
    # num is the numbers of pics to show
    def restore(self, color_pics, num, sess_path, z_size=100, channel=3):
        saver = tf.train.Saver()
        sample_z = np.random.normal(-0.1, 0.1, size=(1, z_size))

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("{}.meta".format(sess_path))
            saver.restore(sess, sess_path)

            gen_samples = sess.run(
                generate_color(self.input_z, self.input_views, channel, reuse=True, training=False),
                feed_dict={self.input_views: color_pics, self.input_z: sample_z}
            )

            return gen_samples


    def model_inputs(self, view_shape, model_shape, z_size):
        input_views = tf.placeholder(tf.float32, (None, *view_shape), name='input_views_stage2')
        input_models = tf.placeholder(tf.float32, (None, *model_shape), name='input_models_stage2')
        input_z = tf.placeholder(tf.float32, (None, z_size), name='input_z_stage2')

        return input_views, input_models, input_z


    def model_loss(self, input_views, input_models, input_z, channel, alpha=0.2):
        g_model = generate_color(input_z, input_views, channel, alpha=alpha)

        d_model_real, d_logits_real = discriminate_color(input_views, input_models, alpha=alpha)
        d_model_fake, d_logits_fake = discriminate_color(input_views, g_model, reuse=True, alpha=alpha)

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

        num_pos = tf.shape(tf.where(input_models > -0.5))[0]  # int
        num_pos = tf.to_float(num_pos)
        g_loss_similar = tf.abs(g_model - input_models)
        g_loss_similar = tf.where(input_models > -0.5, g_loss_similar,
                                          tf.zeros_like(g_loss_similar))
        g_loss_similar = tf.reduce_sum(g_loss_similar) / num_pos

        d_loss = d_loss_real + d_loss_fake
        g_loss = g_loss * 0.1 + g_loss_similar

        return d_loss, g_loss


    def model_opt(self, d_loss, g_loss, learning_rate, beta1):
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminate_color')]
        g_vars = [var for var in t_vars if var.name.startswith('generate_color')]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
            g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

        return d_train_opt, g_train_opt
