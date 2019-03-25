import tensorflow as tf

def input_fn(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    iterator = dataset.make_one_shot_iterator()
    sample = iterator.get_next()
    return sample

if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.DEBUG)

    sample = input_fn('../data/dev.tsv')
    with tf.Session() as sess:
        print(sess.run(sample))