import tensorflow as tf


def generate_shape(z, x, output_dim=1, reuse=False, alpha=0.2, training=True):
    with tf.variable_scope('generate_shape', reuse=reuse):

        x1 = tf.layers.conv2d(x, 64, 4, strides=2, padding='same')
        bn1 = tf.layers.batch_normalization(x1, training=True)
        relu1 = tf.maximum(alpha * bn1, bn1)
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
        # 2x2x2x256

        relu8 = tf.concat([relu8, tf.reshape(relu6, (-1, 2, 2, 2, 256))], 4)

        x9 = tf.layers.conv3d_transpose(relu8, 128, 4, strides=2, padding='same')
        bn9 = tf.layers.batch_normalization(x9, training=training)
        relu9 = tf.maximum(alpha * bn9, bn9)
        # 4x4x4x128

        relu9 = tf.concat([relu9, tf.reshape(relu5, (-1, 4, 4, 4, 128))], 4)

        x10 = tf.layers.conv3d_transpose(relu9, 64, 4, strides=2, padding='same')
        bn10 = tf.layers.batch_normalization(x10, training=training)
        relu10 = tf.maximum(alpha * bn10, bn10)
        # 8x8x8x64

        relu10 = tf.concat([relu10, tf.reshape(relu4, (-1, 8, 8, 8, 64))], 4)

        x11 = tf.layers.conv3d_transpose(relu10, 32, 4, strides=2, padding='same')
        bn11 = tf.layers.batch_normalization(x11, training=training)
        relu11 = tf.maximum(alpha * bn11, bn11)
        # 16x16x16x32

        relu11 = tf.concat([relu11, tf.reshape(relu3, (-1, 16, 16, 16, 16))], 4)

        x12 = tf.layers.conv3d_transpose(relu11, 16, 4, strides=2, padding='same')
        bn12 = tf.layers.batch_normalization(x12, training=training)
        relu12 = tf.maximum(alpha * bn12, bn12)
        # 32x32x32x16

        logits = tf.layers.conv3d_transpose(relu12, output_dim, 4, strides=1, padding='same')

        out = tf.tanh(logits)
        return out


def discriminate_shape(xv, xm, reuse=False, alpha=0.2):
    with tf.variable_scope('discriminate_shape', reuse=reuse):

        xv = tf.layers.conv2d(xv, 2, 4, strides=1, padding='same')
        xv = tf.layers.batch_normalization(xv, training=True)
        xv = tf.maximum(alpha * xv, xv)
        # 128x128x2
        xv = tf.reshape(xv, (-1, 32, 32, 32, 1))

        x = tf.concat([xm, xv], 4)
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

        x4 = tf.layers.conv3d(relu3, 128, 4, strides=2, padding='same')
        bn4 = tf.layers.batch_normalization(x4, training=True)
        relu4 = tf.maximum(alpha * bn4, bn4)
        # 4x4x4x128

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


def generate_color(z, x, output_dim=3, reuse=False, alpha=0.2, training=True):
    with tf.variable_scope('generate_color', reuse=reuse):

        x = tf.reshape(x, (-1, 1, 32, 32, 3))
        x = tf.tile(x, (1, 32, 1, 1, 1))
        # 32x32x32x3

        x1 = tf.layers.conv3d(x, 32, 6, strides=2, padding='same')
        x1 = tf.layers.batch_normalization(x1, training=True)
        x1 = tf.maximum(alpha * x1, x1)
        # 16x16x16x32

        x2 = tf.layers.conv3d(x1, 64, 5, strides=2, padding='same')
        x2 = tf.layers.batch_normalization(x2, training=True)
        x2 = tf.maximum(alpha * x2, x2)
        # 8x8x8x64

        x3 = tf.layers.conv3d(x2, 128, 5, strides=2, padding='same')
        x3 = tf.layers.batch_normalization(x3, training=True)
        x3 = tf.maximum(alpha * x3, x3)
        # 4x4x4X128

        x4 = tf.layers.conv3d(x3, 256, 4, strides=2, padding='same')
        x4 = tf.layers.batch_normalization(x4, training=True)
        x4 = tf.maximum(alpha * x4, x4)
        # 2x2x2x256

        x5 = tf.layers.conv3d(x4, 512, 4, strides=2, padding='same')
        x5 = tf.layers.batch_normalization(x5, training=True)
        x5 = tf.maximum(alpha * x5, x5)
        # 1x1x1x512

        x6 = tf.layers.conv3d_transpose(x5, 256, 4, strides=2, padding='same')
        x6 = tf.layers.batch_normalization(x6, training=training)
        x6 = tf.maximum(alpha * x6, x6)
        # 2x2x2x256

        x6 = tf.concat([x6, x4], 4)
        # 2x2x2x512

        x7 = tf.layers.conv3d_transpose(x6, 128, 4, strides=2, padding='same')
        x7 = tf.layers.batch_normalization(x7, training=training)
        x7 = tf.maximum(alpha * x7, x7)
        # 4x4x4x128

        x7 = tf.concat([x7, x3], 4)
        # 4x4x4x256

        x8 = tf.layers.conv3d_transpose(x7, 64, 4, strides=2, padding='same')
        x8 = tf.layers.batch_normalization(x8, training=training)
        x8 = tf.maximum(alpha * x8, x8)
        # 8x8x8x64

        x8 = tf.concat([x8, x2], 4)
        # 8x8x8x128

        x9 = tf.layers.conv3d_transpose(x8, 32, 4, strides=2, padding='same')
        x9 = tf.layers.batch_normalization(x9, training=training)
        x9 = tf.maximum(alpha * x9, x9)
        # 16x16x16x32

        x9 = tf.concat([x9, x1], 4)
        # 16x16x16x64

        x10 = tf.layers.conv3d_transpose(x9, 16, 4, strides=2, padding='same')
        x10 = tf.layers.batch_normalization(x10, training=training)
        x10 = tf.maximum(alpha * x10, x10)
        # 32x32x32x16

        logits = tf.layers.conv3d_transpose(x10, output_dim, 4, strides=1, padding='same')

        out = tf.tanh(logits)
        return out


def discriminate_color(xv, xm, reuse=False, alpha=0.2):
    with tf.variable_scope('discriminate_color', reuse=reuse):

        xv = tf.reshape(xv, (-1, 1, 32, 32, 3))
        xv = tf.tile(xv, (1, 32, 1, 1, 1))
        # 32x32x32x3

        x = tf.concat([xm, xv], 4)
        # xv=32x32x32x6

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

        x4 = tf.layers.conv3d(relu3, 128, 4, strides=2, padding='same')
        bn4 = tf.layers.batch_normalization(x4, training=True)
        relu4 = tf.maximum(alpha * bn4, bn4)
        # 4x4x4x128

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
