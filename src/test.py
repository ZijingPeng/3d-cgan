from src.trainer import *
import codecs
import json

def test(inputs_pic, inputs_z, channel, alpha=0.2):
    sample_z = np.random.normal(-0.1, 0.1, size=(1, z_size))
    tf.reset_default_graph()

    with tf.Session() as sess:
        input_pic = tf.placeholder(tf.float32, (None, *inputs_pic), name='inputs_pic')
        input_z = tf.placeholder(tf.float32, (None, inputs_z), name='input_z')

        g_model = generator(input_z, input_pic, 1, alpha=alpha)

        dataset = Dataset()
        #t_vars = tf.trainable_variables()
        #g_vars = [var for var in t_vars if var.name.startswith('generator')]

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        #saver = tf.train.import_meta_graph('../checkpoints/dm3.ckpt.meta')
        saver.restore(sess, '../ckp_new/dm4.ckpt')

        json_file = "../file.json"
        f = codecs.open(json_file, 'w', encoding='utf-8')
        data = {}

        for x, y in dataset.batches(batch_size):
            gen_samples = sess.run(
                generator(input_z, input_pic, channel, reuse=True, training=False),
                feed_dict={input_z: sample_z, input_pic: x}
            )
            for i in range(32):
                #_ = display_binvox(gen_samples[i])
                x0 = np.squeeze(gen_samples[i]) > 0
                x0 = np.reshape(x0, (32, 32, 32))
                x0 = x0.tolist()
                data['fake{}'.format(i)] = x0
                print("finish fake{}".format(i))
                #_ = display_binvox(y[i])
                y0 = np.squeeze(y[i]) > 0
                y0 = np.reshape(y0, (32, 32, 32))
                y0 = y0.tolist()
                data['real{}'.format(i)] = y0
                print("finish real{}".format(i))
        json.dump(data, f, sort_keys=True)
        f.write("\n")
        return 0

if __name__ == '__main__':
    dataset = Dataset()
    _ = test(real_pic, real_vox, z_size, 1, alpha)
