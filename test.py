#encoding: utf-8

from datetime import datetime
from tensorflow.python.platform import gfile   #use this library to read files via tensorflow
import numpy as np
import tensorflow as tf
#writer write followings
from dataset import DataSet
from dataset import output_predict
from dataset import output_predict_all
import model
import train_operation as op

TEST_FILE = "testforcal.csv"
IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 55
TARGET_WIDTH = 74
REFINE_TRAIN = True
FINE_TUNE = True
COARSE_DIR = "coarse"
REFINE_DIR = "refine"
batch_size =36

def csv_inputs(batch_size,csv_file_path):
    filename_queue = tf.train.string_input_producer([csv_file_path], shuffle=False)
    reader = tf.TextLineReader()
    _, serialized_example = reader.read(filename_queue)
    filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])
    print(filename, depth_filename)
    # input
    jpg = tf.read_file(filename)
    image = tf.image.decode_jpeg(jpg, channels=3)
    image = tf.cast(image, tf.float32)
    # target
    depth_png = tf.read_file(depth_filename)
    depth = tf.image.decode_png(depth_png, channels=1)
    depth = tf.cast(depth, tf.float32)
    depth = tf.div(depth, [255.0])
    # depth = tf.cast(depth, tf.int64)
    # resize
    image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    depth = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH))
    invalid_depth = tf.sign(depth)
    images, depths, invalid_depths = tf.train.batch(
        [image, depth, invalid_depth],
        batch_size=batch_size,
        num_threads=1,
        capacity=50 + 3 * batch_size,
    )
    return images, depths, invalid_depths

def test():
    with tf.Graph().as_default():
        images,depths,invalid_depths = csv_inputs(batch_size,TEST_FILE)
        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)
        print("testing")
        if REFINE_TRAIN:
            print("refine train.")
            coarse = model.inference(images, keep_conv, reuse= False,trainable=False)  #搭建coarse网络
            logits = model.inference_refine(images, coarse, keep_conv, keep_hidden)
            #logits是网络的计算结果
        else:
            print("coarse train.")
            logits = model.inference(images, keep_conv, keep_hidden)
        loss = model.loss(logits, depths, invalid_depths)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            # parameters
            coarse_params = {}
            refine_params = {}
            if REFINE_TRAIN:
                for variable in tf.all_variables():
                    variable_name = variable.name
                    # print("parameter: %s" % (variable_name))
                    if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                        continue
                    if variable_name.find('coarse') >= 0:
                        coarse_params[variable_name] = variable
                    # print("parameter: %s" %(variable_name))
                    if variable_name.find('fine') >= 0:
                        refine_params[variable_name] = variable
            else:
                for variable in tf.trainable_variables():
                    variable_name = variable.name
                    # print("parameter: %s" %(variable_name))
                    if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                        continue
                    if variable_name.find('coarse') >= 0:
                        coarse_params[variable_name] = variable
                    if variable_name.find('fine') >= 0:
                        refine_params[variable_name] = variable
            # define saver
            # print(coarse_params)
            saver_coarse = tf.train.Saver(coarse_params)
            if REFINE_TRAIN:
                saver_refine = tf.train.Saver(refine_params)
            # fine tune
            if FINE_TUNE:
                coarse_ckpt = tf.train.get_checkpoint_state(COARSE_DIR)
                if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
                    print("Pretrained coarse Model Loading.")
                    saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
                    print("Pretrained coarse Model Restored.")
                else:
                    print("No Pretrained coarse Model.")
                if REFINE_TRAIN:
                    refine_ckpt = tf.train.get_checkpoint_state(REFINE_DIR)
                    if refine_ckpt and refine_ckpt.model_checkpoint_path:
                        print("Pretrained refine Model Loading.")
                        saver_refine.restore(sess, refine_ckpt.model_checkpoint_path)
                        print("Pretrained refine Model Restored.")
                    else:
                        print("No Pretrained refine Model.")

            coord = tf.train.Coordinator()
            threads=tf.train.start_queue_runners(coord=coord)
            if REFINE_TRAIN:
                loss_value, coarse_val, logits_val, depth_val, images_val = sess.run([loss, coarse,logits, depths, images],
                                                                         feed_dict={keep_conv: 1, keep_hidden: 1})
            else:
                loss_value, logits_val, depth_val, images_val = sess.run([loss, logits, depths, images],
                                                                        feed_dict={keep_conv: 1, keep_hidden: 1})
            if REFINE_TRAIN:
                output_predict_all(coarse_val,logits_val, depth_val, images_val, "images/test_result")
            else:
                output_predict(logits_val,depth_val,images_val, "images/test_result")
            print(loss_value)
            coord.request_stop()
            coord.join(threads)

def main(argv=None):
    test()


if __name__ == '__main__':
    tf.app.run()