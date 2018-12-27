import tensorflow as tf


def generator(z, x, output_dim, reuse=False, alpha=0.2, training=True):
    with tf.variable_scope('generator', reuse=reuse):

        x1 = tf.layers.conv2d(x, 64, 4, strides=4, padding='same')
        relu1 = tf.maximum(alpha * x1, x1)
        # 64x64x64

        x2 = tf.layers.conv2d(relu1, 128, 4, strides=2, padding='same')
        bn2 = tf.layers.batch_normalization(x2, training=True)
        relu2 = tf.maximum(alpha * bn2, bn2)
        # 32x32x128

        x3 = tf.layers.conv2d(relu2, 256, 4, strides=2, padding='same')
        bn3 = tf.layers.batch_normalization(x3, training=True)
        relu3 = tf.maximum(alpha * bn3, bn3)
        # 16X16X256

        x4 = tf.layers.conv2d(relu3, 512, 4, strides=2, padding='same')
        bn4 = tf.layers.batch_normalization(x4, training=True)
        relu4 = tf.maximum(alpha * bn4, bn4)
        # 8x8x512

        x5 = tf.layers.conv2d(relu4, 512, 4, strides=2, padding='same')
        bn5 = tf.layers.batch_normalization(x5, training=True)
        relu5 = tf.maximum(alpha * bn5, bn5)
        # 4x4x512

        x6 = tf.layers.conv2d(relu5, 512, 4, strides=2, padding='same')
        bn6 = tf.layers.batch_normalization(x6, training=True)
        relu6 = tf.maximum(alpha * bn6, bn6)
        # 2x2x512

        x7 = tf.layers.conv2d(relu6, 512, 4, strides=2, padding='same')
        bn7 = tf.layers.batch_normalization(x7, training=True)
        relu7 = tf.maximum(alpha * bn7, bn7)
        # 1x1x512

        relu7 = tf.reshape(relu7, (-1, 1, 1, 1, 512))

        nz = tf.layers.dense(z, 512)

        # Reshape it to start the convolutional stack
        xz = tf.reshape(nz, (-1, 1, 1, 1, 512))
        relu7 += xz

        x8 = tf.layers.conv3d_transpose(relu7, 256, 4, strides=2, padding='same')
        bn8 = tf.layers.batch_normalization(x8, training=training)
        relu8 = tf.maximum(alpha * bn8, bn8)
        # 2x2x2x512

        x9 = tf.layers.conv3d_transpose(relu8, 128, 4, strides=2, padding='same')
        bn9 = tf.layers.batch_normalization(x9, training=training)
        relu9 = tf.maximum(alpha * bn9, bn9)
        # 4x4x4x128

        x10 = tf.layers.conv3d_transpose(relu9, 64, 4, strides=2, padding='same')
        bn10 = tf.layers.batch_normalization(x10, training=training)
        relu10 = tf.maximum(alpha * bn10, bn10)
        # 8x8x8x64

        x11 = tf.layers.conv3d_transpose(relu10, 32, 4, strides=2, padding='same')
        bn11 = tf.layers.batch_normalization(x11, training=training)
        relu11 = tf.maximum(alpha * bn11, bn11)
        # 16x16x16x32

        x12 = tf.layers.conv3d_transpose(relu11, 16, 4, strides=2, padding='same')
        bn12 = tf.layers.batch_normalization(x12, training=training)
        relu12 = tf.maximum(alpha * bn12, bn12)
        # 32x32x32x16

        logits = tf.layers.conv3d_transpose(relu12, output_dim, 4, strides=1, padding='same')

        out = tf.tanh(logits)
        return out


def discriminator(xv, xp, reuse=False, alpha=0.2):
    with tf.variable_scope('discriminator', reuse=reuse):
        xp = tf.layers.conv2d(xp, 2, 4, strides=1, padding='same')
        xp = tf.layers.batch_normalization(xp, training=True)
        xp = tf.maximum(alpha * xp, xp)
        # 128x128x2
        xp = tf.reshape(xp, (-1, 32, 32, 32, 1))

        x = tf.concat([xv, xp], 4)
        # xv=32x32x32x2

        x1 = tf.layers.conv3d(x, 64, 4, strides=2, padding='same')
        relu1 = tf.maximum(alpha * x1, x1)
        # 16x16x16x64

        x2 = tf.layers.conv3d(relu1, 128, 4, strides=2, padding='same')
        bn2 = tf.layers.batch_normalization(x2, training=True)
        relu2 = tf.maximum(alpha * bn2, bn2)
        # 8x8x8x128

        x3 = tf.layers.conv3d(relu2, 256, 4, strides=2, padding='same')
        bn3 = tf.layers.batch_normalization(x3, training=True)
        relu3 = tf.maximum(alpha * bn3, bn3)
        # 4x4x4x256

        x4 = tf.layers.conv3d(relu3, 512, 4, strides=1, padding='same')
        bn4 = tf.layers.batch_normalization(x4, training=True)
        relu4 = tf.maximum(alpha * bn4, bn4)
        # 4x4x4x512

        x5 = tf.layers.conv3d(relu4, 1, 4, strides=1, padding='same')
        bn5 = tf.layers.batch_normalization(x5, training=True)
        relu5 = tf.maximum(alpha * bn5, bn5)
        # 4x4x4x1

        logits = tf.layers.dense(relu5, 1)
        out = tf.sigmoid(logits)

        return out, logits
