# encoding: utf-8

# tensorflow
import tensorflow as tf
import math
from model_part import conv2d
from model_part import fc

def inference(images,keep_conv,reuse=False,trainable=True):
    print(trainable)
    coarse1_conv = conv2d('coarse1', images, [12, 12, 3, 96], [96], [1, 4, 4, 1], padding='VALID', reuse=reuse, trainable=trainable)
    print(images) #(10,228,304,3)
    print(coarse1_conv) #(10,55,74,96)
    coarse1 = tf.nn.max_pool(coarse1_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    print(coarse1)#(10,27,36,96)
    coarse2_conv = conv2d('coarse2', coarse1, [5, 5, 96, 256], [256], [1, 1, 1, 1], padding='SAME', reuse=reuse, trainable=trainable)
    print(coarse2_conv)#(10,23,32,256)
    coarse2 = tf.nn.max_pool(coarse2_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    print(coarse2)#(10,12,16,256)
    coarse3 = conv2d('coarse3', coarse2, [3, 3, 256, 384], [384], [1, 1, 1, 1], padding='SAME', reuse=reuse, trainable=trainable)
    print(coarse3)#(10,10,14,384)
    coarse4 = conv2d('coarse4', coarse3, [3, 3, 384, 384], [384], [1, 1, 1, 1], padding='SAME', reuse=reuse, trainable=trainable)
    print(coarse4)#(10,8,12,384)
    coarse5 = conv2d('coarse5', coarse4, [3, 3, 384, 256], [256], [1, 2, 2, 1], padding='VALID', reuse=reuse, trainable=trainable)
    print(coarse5)#(10,6,10,256)
    coarse6 = fc('coarse6', coarse5, [6*8*256, 4096], [4096], reuse=reuse, trainable=trainable)
    coarse6_dropout = tf.nn.dropout(coarse6, keep_conv)
    print(coarse6)#(10,4096)
    coarse7 = fc('coarse7', coarse6_dropout, [4096, 4070], [4070], reuse=reuse, trainable=trainable)
    print(coarse7)#(10,4070)
    coarse7_output = tf.reshape(coarse7, [-1, 55, 74, 1])
    print(coarse7_output)#(10,55,74,1)
    return coarse7_output


def inference_refine(images, coarse7_output, keep_hidden, reuse=False, trainable=True):
    fine1_conv = conv2d('fine1', images, [9, 9, 3, 63], [63], [1, 2, 2, 1], padding='VALID', reuse=reuse, trainable=trainable)
    print(images) #(10,228,304,3)
    print(fine1_conv) #(10,110,148,63)
    fine1 = tf.nn.max_pool(fine1_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='fine_pool1')
    print(fine1) #(10,55,74,63)
    fine1_dropout = tf.nn.dropout(fine1, keep_hidden)
    print(fine1_dropout) #(10,55,74,63)
    fine2 = tf.concat([fine1_dropout, coarse7_output],3)
    print(fine2)#（10，55，74，64）
    print(coarse7_output) #（10，55，74，1）
    fine3 = conv2d('fine3', fine2, [5, 5, 64, 64], [64], [1, 1, 1, 1], padding='SAME', reuse=reuse, trainable=trainable)
    print(fine3)#（10，55，74，64）
    fine3_dropout = tf.nn.dropout(fine3, keep_hidden)
    print(fine3_dropout)#（10，55，74，64）
    fine4 = conv2d('fine4', fine3_dropout, [5, 5, 64, 1], [1], [1, 1, 1, 1], padding='SAME', reuse=reuse, trainable=trainable)
    print(fine4)#（10，55，74，1）
    return fine4


def loss(logits, depths, invalid_depths):
    print("losting")
    logits_flat = tf.reshape(logits, [-1, 55*74])  #-1表示，不用指定这一维的大小，自动算
    depths_flat = tf.reshape(depths, [-1, 55*74])
    invalid_depths_flat = tf.reshape(invalid_depths, [-1, 55*74])

    predict = tf.multiply(logits_flat, invalid_depths_flat)
    target = tf.multiply(depths_flat, invalid_depths_flat) #矩阵中数字相乘
    d = tf.subtract(predict, target)  #减法
    square_d = tf.square(d)  #平方
    sum_square_d = tf.reduce_sum(square_d, 1) #降维求和
    sum_d = tf.reduce_sum(d, 1)
    sqare_sum_d = tf.square(sum_d)
    cost = tf.reduce_mean(sum_square_d / 55.0*74.0 - 0.5*sqare_sum_d / math.pow(55*74, 2))
    tf.add_to_collection('losses', cost)  #放入集合变成列表
    print("done")
    return tf.add_n(tf.get_collection('losses'), name='total_loss') #从集合中取出并相加


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op
