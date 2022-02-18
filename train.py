import sys
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow.compat.v1 as tf
import tensorflow as tf
sys.path.append('tools')
from model_graph_test import model
#use training_text2sp for training text to mel
#use training_superresolution for training super resolution network
gr=model('...DATA PATH...','training_text2sp')
#The supervisor takes care of session initialization and all that jazz     
# Use tqdm for progress bar
#https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/tf.train.Supervisor.md
logdir = 'logs/text-to-spec'
#config = tf.ConfigProto(allow_soft_placement = True)
sv = tf.train.Supervisor(logdir=logdir, save_model_secs=0, global_step=gr.global_step)
with sv.managed_session() as sess:
    while True:
            for _ in tqdm(range(gr.num_batch), total=gr.num_batch, ncols=70, leave=False, unit='b'):
                global_s,_=sess.run([gr.global_step,gr.train_operation])
                #print(global_s)
                if global_s % 1000 == 0:
                    sv.saver.save(sess, logdir + '/model_gs_{}'.format(str(global_s // 1000).zfill(3) + "k"))