#https://stackoverflow.com/questions/22947905/flask-example-with-post
	
from io import StringIO 
import json

from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin


'''
Persian tts Project
'''
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

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods = ['GET', 'POST'])
@cross_origin()

def hello():
    out = ''
    res = dict()
    val = ''
    is_form = False
    if request.method == 'POST':
        data = ''
        if(request.get_json() and 'text' in request.get_json()):
            data = request.get_json()['text']
        elif request.form:
            data = request.form.get('text')
            is_form = True
        if data:
            try:
                sentenses=[data]
                gr.predict(sentenses)
                #import IPython
                #IPython.display.Audio("generated_samples/1.wav")
                res['result'] = 'Audio saved successfully as generated_samples/1.wav'
                out = json.dumps(res)
            except Exception as e:
                res['error'] = 'An unexpected error occurred!'
                out = json.dumps(res)
            val = data
        else:
            out = 'please fill text field!'
    form = '<form action="" method="post"><textarea name="text" rows="15" cols="80" placeholder="متن خود را برای تبدیل به صورت وارد کنید.">'+val+'</textarea><br /><input type="submit"></form>'
    if request.method == 'GET':
        out = form
    if is_form:
        out += form
    return out

if __name__ == '__main__':
    app.run()