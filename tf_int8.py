import sys
import os

if sys.version_info.major >= 3:
    import pathlib
else:
    import pathlib2 as pathlib



import tensorflow as tf
lite=tf.contrib.lite

saved_models_root="../saved_model/tiny/"
tf.enable_eager_execution()
#help(lite.TFLiteConverter.from_saved_model)
#saved_model_dir = str(sorted(pathlib.Path(saved_models_root).glob("*"))[-1])
converter = tf.lite.TocoConverter.from_saved_model(saved_models_root )
#https://www.tensorflow.org/guide/saved_model
#converter = lite


#converter = lite.TFLiteConverter.from_saved_model(saved_models_root)
#tflite_model = converter.convert()


'''
tflite_model = converter.convert()
tflite_models_dir = pathlib.Path(saved_model_dir )
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"yolov3-tiny.tflite"
tflite_model_file.write_bytes(tflite_model)

tf.logging.set_verbosity(tf.logging.INFO)
converter.post_training_quantize = True
tflite_quant_model = converter.convert()
tflite_model_quant_file = tflite_models_dir/"yolov3-tiny_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model)
'''




