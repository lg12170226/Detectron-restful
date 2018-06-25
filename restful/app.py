#coding=utf-8
import logging
from logging.handlers import RotatingFileHandler

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
    #app.logger.warning('A warning occurred (%d apples)', 42)
    #app.logger.error('An error occurred')                     #private log
    #logging.info('Info')                                      #root log
    
    
    #imagepath = request.form.getlist('data')
    #imagepath = request.form.get("data",type=str,default=None)
    js = request.get_json()
    
    #return json init
    out_json = {"data":[]}
    
    #js is not dict return []
    if isinstance(js,dict) and js.has_key('data') :
        imagepath = js.get('data',None)
        info_str = 'images path:' + imagepath
        logger.info(info_str)
    else:
        logger.warning('post has not data or data-key!!!')
        return json.dumps(out_json)
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
            #img_np = misc.imread(new_imagepath)
            img_np = cv2.imread(new_imagepath)  #read image by cv2 ,the same as /tool/test_net.py
        else:
            logger.warning('the image is not download on internet!!!')
            return json.dumps(out_json)
    # path 
    else:
        if not os.path.exists(imagepath):
            logger.warning('the image is not exists!!!')
            return json.dumps(out_json)
        else:
            #img_np = misc.imread(imagepath)
            img_np = cv2.imread(imagepath)  #read image by cv2 ,the same as /tool/test_net.py
            
   
    predict_datalist = mm.predict(img_np)
    if len(predict_datalist) > 0:
        logger.info('the images predict completed!!!')
        res_log = []
        for i in range(len(predict_datalist)):
            single_data = {}
            single_data = predict_datalist[i]
            res_log.append(single_data['cls'])
        logger.info(res_log)
        out_json["data"] = predict_datalist
    else:
        logger.warning('the images has not right bbox!!!')
    return json.dumps(out_json)
    
    
if __name__ == '__main__':

    if not os.path.exists('./log'):
        os.makedirs('./log')
    
    logger = logging.getLogger('')    #set root level , default is WRAINING
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        "[%(asctime)s] {%(pathname)s - %(module)s - %(funcName)s:%(lineno)d} - %(message)s")
    handler = RotatingFileHandler('./log/oilsteal.log', maxBytes=10000000, backupCount=10)
    handler.setFormatter(formatter)
    logger.addHandler(handler)     #ok  start root log
    #app.logger.addHandler(handler)  #ok  start private log
    app.run(host="0.0.0.0",port=8080,debug=False)   #threaded=True

    