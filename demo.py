import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sys.path.append('tools')
sys.path.append('network')
from hp import HP
from model_graph import model
tf.reset_default_graph()
gr=model('null','demo')
sentenses=["معلوم بود واقعا به دنبال جوابى براى سوالش نیست"]
gr.predict(sentenses)
import IPython
IPython.display.Audio("generated_samples/1.wav")
