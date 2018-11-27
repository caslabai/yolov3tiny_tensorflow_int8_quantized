# -*- coding: utf-8 -*-
import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import time
from tensorflow.python.framework import graph_util

# import argparse

import yolo_v3
import yolo_v3_tiny

from utils import load_coco_names, draw_boxes, convert_to_original_size, \
    load_weights, detections_boxes, non_max_suppression

FLAGS = tf.app.flags.FLAGS


# parser = argparse.ArgumentParser()
# parser.add_argument('--tiny', dest='flag', action='store_true')
# args = parser.parse_args()


tf.app.flags.DEFINE_string('input_img', '', 'Input image')
tf.app.flags.DEFINE_string('output_img', '', 'Output image')

tf.app.flags.DEFINE_string(
    'class_names', 'data/coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NHWC', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'ckpt_file', './saved_model/model.ckpt', 'Checkpoint file')
tf.app.flags.DEFINE_bool(
    'tiny', False, 'Use tiny version of YOLOv3')

tf.app.flags.DEFINE_integer('thread', 1, 'thread number')

tf.app.flags.DEFINE_integer('size', 416, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')


def yolo_full():
    '''
    if FLAGS.tiny:
            model = yolo_v3_tiny.yolo_v3_tiny
            ckpt_file = './saved_model/yolov3-tiny.ckpt'
    else:
        model = yolo_v3.yolo_v3
        ckpt_file = './saved_model/yolov3.ckpt'
    '''
    model = yolo_v3_tiny.yolo_v3_tiny
    ckpt_file = './save_model/tiny/yolov3-tiny.ckpt'

    img = Image.open(FLAGS.input_img)
    img_resized = img.resize(size=(FLAGS.size, FLAGS.size))

    classes = load_coco_names(FLAGS.class_names)

    # placeholder for detector inputs
    inputs = tf.placeholder(tf.float32, [1, FLAGS.size, FLAGS.size, 3])

    with tf.variable_scope('detector'):
        detections = model(inputs, len(classes),
                           data_format=FLAGS.data_format)

    saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))

    boxes = detections_boxes(detections)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=FLAGS.thread, inter_op_parallelism_threads=FLAGS.thread, \
                        allow_soft_placement=True, device_count = {'GPU': 1})
    with tf.Session(config=session_conf) as sess:
    
        saver.restore(sess, ckpt_file)
        tf.summary.FileWriter("TensorBoard/", graph = sess.graph)
        print( ">>>>>>>>>>>>>>>>> %d"  % len(tf.get_default_graph().as_graph_def().node))
        print('Model restored.')
        start = time.time()
        detected_boxes = sess.run(
            boxes, feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})
        end = time.time()
        print("%2.2f secs"%(end - start))
        '''
        opts = tf.profiler.ProfileOptionBuilder.float_operation()    
        flops = tf.profiler.profile(tf.get_default_graph() , run_meta=tf.RunMetadata(), cmd='op', options=opts)
        if flops is not None:
            #print('Flops should be ~',2*25*16*9)
            #print('25 x 25 x 9 would be',2*25*25*9) # ignores internal dim, repeats first
            print('TF stats gives',flops.total_float_ops)
        '''
        #output_node_names = "detector/yolo-v3-tiny/detections"
        output_node_names = "concat_1"
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names.split( "," ))
    
        print( ">>>>>>>>>>> %d ops in the final graph."  % len( output_graph_def.node))
        with tf.gfile.GFile( "save_model/tiny/pb/frozen_model_yolov3-tiny.pb", "wb" ) as f:
            f.write( output_graph_def.SerializeToString( ))
        #builder = tf.saved_model.builder.SavedModelBuilder('./savemodel')
        #builder.add_meta_graph_and_variables(sess, ['cpu_server_1'])




    #builder.save()
    print (detected_boxes.shape) 
    #print (detected_boxes[0,1,1]) 
    #print (np.array(img_resized, dtype=np.float32)[111,111]) 
    #print (inputs.shape) 
    filtered_boxes = non_max_suppression(detected_boxes,
                                         confidence_threshold=FLAGS.conf_threshold,
                                         iou_threshold=FLAGS.iou_threshold)

    draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size))

    img.save(FLAGS.output_img)
    




def main(argv=None):
    
    time_yolo = time.process_time() 
    yolo_full()
    time_yolo = time.process_time() - time_yolo
    print(time_yolo)
    


if __name__ == '__main__':
    tf.app.run()
