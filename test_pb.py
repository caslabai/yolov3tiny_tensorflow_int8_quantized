#-*- coding: utf-8 -*-
#https://blog.csdn.net/guyuealian/article/details/82218092
import tensorflow as tf
from tensorflow.python.framework import graph_util
from utils import load_coco_names, draw_boxes, convert_to_original_size, \
    load_weights, detections_boxes, non_max_suppression
from create_tf_record import *

import yolo_v3
import yolo_v3_tiny

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('size', 416, 'Image size')
#tf.app.flags.DEFINE_string('output_img', './pb_img_result', 'Output image')
tf.app.flags.DEFINE_string('output_img', '', 'Output image')
tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')
tf.app.flags.DEFINE_string('class_names', 'coco.names', 'File with class names')


def freeze_graph( input_checkpoint,output_graph):
    '''
    :param input_checkpoint:
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state( model_folder) 
    # input_checkpoint = checkpoint.model_checkpoint_path
 
    saver = tf.train.import_meta_graph( input_checkpoint + '.meta', clear_devices=True )
    #saver2 = tf.train.Saver(var_list=tf.global_variables(scope='detector'))
 
    with tf.Session( ) as sess:
        saver.restore( sess, input_checkpoint)
        print( "[ckpt] op count >>>>>>>>>>>>>>>>> %d"  % len(tf.get_default_graph().as_graph_def().node))
        '''
        from tensorflow.tools.graph_transforms import TransformGraph
        transforms = ['add_default_attributes',
                      'remove_nodes(op=Identity, op=CheckNumerics)',
                      'fold_batch_norms', 'fold_old_batch_norms',
                      'strip_unused_nodes', 'sort_by_execution_order']
        transformed_graph_def = TransformGraph(tf.get_default_graph().as_graph_def(),'Placeholder', output_node_names.split(","), transforms)
        '''

        #[  print( n.name) for n in tf.get_default_graph( ).as_graph_def( ).node]
        tf.summary.FileWriter("TensorBoard/", graph = sess.graph)
        #graph=tf.Graph().as_default()
        output_node_names = "detector/yolo-v3-tiny/detections" 
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            #input_graph_def= graph.as_graph_def() , 
            input_graph_def=sess.graph_def, 
            #input_graph_def=transformed_graph_def, 
            output_node_names=output_node_names.split( "," ))
 
        print( "[pb] op count >>>>>>>>>>>>>>>>> %d ops in the final graph."  % len( output_graph_def.node))
        with tf.gfile.GFile( output_graph, "wb" ) as f: 
            f.write( output_graph_def.SerializeToString( ))
        

    #tf.train.ExponentialMovingAverage(0.999)
    #variables_to_restore = variable_averages.variables_to_restore()
    #saver = tf.train.Saver(variables_to_restore) ,



def freeze_graph_test(pb_path, image_path):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

        #tf.global_variables_initializer()
        with tf.Session() as sess:
            #saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))
            #input_checkpoint='./saved_model/yolov3-tiny.ckpt'
            #saver.restore( sess, input_checkpoint)
            
            sess.run(tf.global_variables_initializer())

            #sess.run(tf.initialize_all_variables())


 
            # 定义输入的张量名称,对应网络结构的输入张量
            # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
            #input_image_tensor = sess.graph.get_tensor_by_name("detector/truediv:0")
            input_image_tensor = sess.graph.get_tensor_by_name("Placeholder:0")
            #input_image_tensor = input_image_tensor / 255
            #input_image_tensor = sess.graph.get_tensor_by_name("inputs:0")
            #input_keep_prob_tensor = sess.graph.get_tensor_by_name("keep_prob:0")
            #input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")
 
            # 定义输出的张量名称
            #output_tensor_name = sess.graph.get_tensor_by_name("detector/yolo-v3-tiny/detections:0")
            output_tensor_name = sess.graph.get_tensor_by_name("concat_1:0")
 
            # 读取测试图片

            img = Image.open(image_path )
            img_resized = img.resize( size=(416,416) )
            
            #im=read_image(image_path,416,416,normalization=True)
            #im=im[np.newaxis,:]
            # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
            # out=sess.run("InceptionV3/Logits/SpatialSqueeze:0", feed_dict={'input:0': im,'keep_prob:0':1.0,'is_training:0':False})
            
            #model = yolo_v3_tiny.yolo_v3_tiny
            #detections = model(input_image_tensor , 80 ,data_format='NHWC')
            #boxes = detections_boxes(detections)
            #print (np.array(img_resized, dtype=np.float32))
            
            #detected_boxes = sess.run( sess.graph_def , 
            #        feed_dict={ input_image_tensor: [np.array(img_resized, dtype=np.float32)] })
           
          
            detected_boxes = sess.run(output_tensor_name , 
                    feed_dict={ input_image_tensor: [np.array(img_resized, dtype=np.float32)] })
           
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.profiler.profile(tf.get_default_graph() , run_meta=tf.RunMetadata(), cmd='op'    , options=opts)
            if flops is not None:
                #print('Flops should be ~',2*25*16*9)
                #print('25 x 25 x 9 would be',2*25*25*9) # ignores internal dim, repeats first
                print('TF stats gives',flops.total_float_ops) 
        
    print ('[pb] output tensor shape: ', detected_boxes.shape) 
    #print (detected_boxes[0,1,1]) 
    #print (np.array(img_resized, dtype=np.float32)[111,111]) 
    #print (input_image_tensor.shape) 
    filtered_boxes = non_max_suppression(detected_boxes,
                                         confidence_threshold=FLAGS.conf_threshold,
                                         iou_threshold=FLAGS.iou_threshold)
    classes = load_coco_names(FLAGS.class_names)
    draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size))
    
    writer = tf.summary.FileWriter("TensorBoard/", graph = sess.graph)
    #img.save(FLAGS.output_img)
    img.save("pboutimage.jpg")


image_path = './dog.jpg'
#image_path = './horses.jpg'

input_checkpoint='./saved_model/yolov3-tiny.ckpt'
out_pb_path="frozen_model_yolov3-tiny.pb" 
#out_pb_path="./saved_model/tiny/frozen_yolov3-tiny.pb" 
#out_pb_path="frozen_flowers_model_yolov3-tiny.pb" 

#freeze_graph( input_checkpoint, out_pb_path)
freeze_graph_test(out_pb_path, image_path)


