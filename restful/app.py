#coding=utf-8
from flask import Flask
from model import Model
from flask import request,Response
from scipy import misc

import json
import urllib,urllib2


import cv2
import os
#import core.infer_simple_test as infer_test
#from infer_simple_test import Model
from model import Model
app = Flask(__name__)



cfg_path = './model/retinanet.yaml'
weights_path = './model/model.pkl'
mm = Model(cfg_path,weights_path)

    

@app.route('/user', methods=['POST'])
def info():

    #imagepath = request.form.getlist('data')
    #imagepath = request.form.get("data",type=str,default=None)
    js = request.get_json()
    imagepath = js.get('data',None)
    if (imagepath.startswith("https://") or imagepath.startswith("http://") or imagepath.startswith("file://")):
        imagefile = urllib.urlopen(imagepath)
        status=imagefile.code
        # url
        if(status==200): 
            image_data = imagefile.read()
            image_name = os.path.basename(imagepath)
            #new_imagepath = filepath+"/"+image_name
            new_imagepath = image_name
            with open(new_imagepath, 'wb') as code:
                code.write(image_data)
            img_np = misc.imread(new_imagepath)
        else:
            out_json = {"data":[]}
            return json.dumps(out_json)
    # path 
    else:
        if not os.path.exists(imagepath):
            out_json = {"data":[]}
            return json.dumps(out_json)
        else:
            img_np = misc.imread(imagepath)
    return mm.predict(img_np)
    
    
    
if __name__ == '__main__':
    
    app.run(host="0.0.0.0",port=8080,debug=False)
    
    
#,threaded=True
    