from src.layers import generate_grid, discriminate_grid
import pickle as pkl
from src.preprocess import *
import numpy as np
import tensorflow as tf
import codecs
import json


class Stage:
    def __init__(self, inputs_pic=(128, 128, 1), inputs_vox=(32, 32, 32, 1), z_size=100, channel=1, learning_rate=0.0002, alpha=0.2, beta1=0.5):
        tf.reset_default_graph()

        self.inputs_pic, self.inputs_vox, self.input_z = self.model_inputs(inputs_pic, inputs_vox, z_size)

        self.d_loss, self.g_loss = self.model_loss(self.inputs_pic, self.inputs_vox, self.input_z, channel, alpha=alpha)

        self.d_opt, self.g_opt = self.model_opt(self.d_loss, self.g_loss, learning_rate, beta1)


    # train cgan
    # save_path to save checkpoints
    # use sess-path if restore checkpoints
    def train(self, dataset, epochs, save_path="../ckp/dm0.ckpt", sess_path=None, batch_size=32, z_size=100, print_every=10, show_every=200):

        saver = tf.train.Saver()
        sample_z = np.random.normal(-1, 1, size=(32, z_size))

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
                    batch_z = np.random.normal(-1, 1, size=(x.shape[0], z_size))
                    _ = sess.run(self.d_opt, feed_dict={self.inputs_pic: x, self.inputs_vox: y, self.input_z: batch_z})
                    _ = sess.run(self.g_opt, feed_dict={self.inputs_pic: x, self.inputs_vox: y, self.input_z: batch_z})

                    if steps % print_every == 0:
                        train_loss_d = self.d_loss.eval({self.inputs_pic: x, self.inputs_vox: y, self.input_z: batch_z})
                        train_loss_g = self.g_loss.eval({self.inputs_pic: x, self.inputs_vox: y, self.input_z: batch_z})

                        print("Epoch {}/{}...".format(e + 1, epochs),
                              "Discriminator Loss: {}...".format(train_loss_d),
                              "Generator Loss: {}".format(train_loss_g))

                    if steps % show_every == 0:
                        gen_samples = sess.run(
                            generate_grid(self.input_z, self.inputs_pic, 1, reuse=True, training=False),
                            feed_dict={self.input_z: sample_z, self.inputs_pic: x}
                        )
                        samples.append(gen_samples)
                        #_ = display_binvox(gen_samples[0])

            if save_path is not None:
                saver.save(sess, save_path)

        with open('samples.pkl', 'wb') as f:
            pkl.dump(samples, f)

        return losses, samples


    # restore the network and show the results of input pics
    # pics should be a list of pics
    # num is the numbers of pics to show
    def restore(self, pics, num, sess_path, save_path="../out/file.json", z_size=100, channel=1):
        saver = tf.train.Saver()
        sample_z = np.random.normal(-0.1, 0.1, size=(1, z_size))

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("{}.meta".format(sess_path))
            saver.restore(sess, sess_path)

            gen_samples = sess.run(
                generate_grid(self.input_z, self.inputs_pic, 1, reuse=True, training=False),
                feed_dict={self.input_z: sample_z, self.inputs_pic: pics}
            )

            json_file = save_path
            f = codecs.open(json_file, 'w', encoding='utf-8')
            data = {}

            for i in range(num):
                _ = display_binvox(gen_samples[i])
                x0 = np.squeeze(gen_samples[i]) > 0
                x0 = np.reshape(x0, (32, 32, 32))
                x0 = x0.tolist()
                data['out{}'.format(i)] = x0
                print("finish")

            json.dump(data, f, sort_keys=True)
            f.write("\n")

            return 0


    def model_inputs(self, pic_dim, vox_dim, z_dim):
        inputs_pic = tf.placeholder(tf.float32, (None, *pic_dim), name='inputs_pic')
        inputs_vox = tf.placeholder(tf.float32, (None, *vox_dim), name='inputs_vox')
        inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')

        return inputs_pic, inputs_vox, inputs_z


    def model_loss(self, inputs_pic, inputs_vox, inputs_z, channel, alpha=0.2):
        g_model = generate_grid(inputs_z, inputs_pic, channel, alpha=alpha)

        d_model_real, d_logits_real = discriminate_grid(inputs_vox, inputs_pic, alpha=alpha)
        d_model_fake, d_logits_fake = discriminate_grid(g_model, inputs_pic, reuse=True, alpha=alpha)

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))
        g_loss_similar = tf.reduce_mean(tf.square(g_model - inputs_vox))

        d_loss = d_loss_real + d_loss_fake
        g_loss = g_loss + g_loss_similar

        return d_loss, g_loss


    def model_opt(self, d_loss, g_loss, learning_rate, beta1):
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
            g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

        return d_train_opt, g_train_opt
