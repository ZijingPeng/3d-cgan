import pickle as pkl

from src.dataset import *
from src.preprocess import *
from src.gan import *
from src.train_params import *


def model_inputs(pic_dim, vox_dim, z_dim):
    inputs_pic = tf.placeholder(tf.float32, (None, *pic_dim), name='inputs_pic')
    inputs_vox = tf.placeholder(tf.float32, (None, *vox_dim), name='inputs_vox')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')

    return inputs_pic, inputs_vox, inputs_z


def model_loss(inputs_pic, inputs_vox, inputs_z, channel, batch_size, alpha=0.2):
    g_model = generator(inputs_z, inputs_pic, channel, alpha=alpha)

    d_model_real, d_logits_real = discriminator(inputs_vox, inputs_pic, batch_size, alpha=alpha)
    d_model_fake, d_logits_fake = discriminator(g_model, inputs_pic, batch_size, reuse=True, alpha=alpha)

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


def model_opt(d_loss, g_loss, learning_rate, beta1):
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt


class GAN:
    def __init__(self, inputs_pic, inputs_vox, inputs_z, channel, learning_rate, alpha=0.2, beta1=0.5):
        tf.reset_default_graph()

        self.inputs_pic, self.inputs_vox, self.input_z = model_inputs(inputs_pic, inputs_vox, inputs_z)

        self.d_loss, self.g_loss = model_loss(self.inputs_pic, self.inputs_vox, self.input_z, channel, batch_size,
                                              alpha=alpha)

        self.d_opt, self.g_opt = model_opt(self.d_loss, self.g_loss, learning_rate, beta1)


# first
def train(net, dataset, epochs, batch_size, print_every=10, show_every=200):
    saver = tf.train.Saver()
    sample_z = np.random.normal(-1, 1, size=(32, z_size))

    samples, losses = [], []
    steps = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for x, y in dataset.batches(batch_size):
                steps += 1
                batch_z = np.random.normal(-1, 1, size=(x.shape[0], z_size))
                _ = sess.run(net.d_opt, feed_dict={net.inputs_pic: x, net.inputs_vox: y, net.input_z: batch_z})
                _ = sess.run(net.g_opt, feed_dict={net.inputs_pic: x, net.inputs_vox: y, net.input_z: batch_z})

                if steps % print_every == 0:
                    train_loss_d = net.d_loss.eval({net.inputs_pic: x, net.inputs_vox: y, net.input_z: batch_z})
                    train_loss_g = net.g_loss.eval({net.inputs_pic: x, net.inputs_vox: y, net.input_z: batch_z})

                    print("Epoch {}/{}...".format(e + 1, epochs),
                          "Discriminator Loss: {}...".format(train_loss_d),
                          "Generator Loss: {}".format(train_loss_g))

                if steps % show_every == 0:
                    gen_samples = sess.run(
                        generator(net.input_z, net.inputs_pic, 1, reuse=True, training=False),
                        feed_dict={net.input_z: sample_z, net.inputs_pic: x}
                    )
                    samples.append(gen_samples)
                    _ = display_binvox(gen_samples[0])


            if (e + 1) % 20 == 0:
                saver.save(sess, '../checkpoints/dm{}.ckpt'.format(e // 20))

    with open('samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    return losses, samples

if __name__ == '__main__':
    net = GAN(real_pic, real_vox, z_size, 1, learning_rate, alpha=alpha, beta1=beta1)

    trainset = Dataset()

    losses, samples = train(net, trainset, epochs, batch_size)
