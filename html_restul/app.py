
#encoding:utf-8
#!/usr/bin/env python
from werkzeug.utils import secure_filename
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort
import time
import os
#from strutil import Pic_str
import uuid



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
from model import Model

from bson.objectid import ObjectId


app = Flask(__name__)
cfg_path = './model/retinanet.yaml'
weights_path = './model/model.pkl'
mm = Model(cfg_path,weights_path)

 
app = Flask(__name__)
UPLOAD_FOLDER = 'upload'
PREDICT_FOLDER = 'predict'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICT_FOLDER'] = PREDICT_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])

PREDICTED_IMAGE = ''
#ALLOWED_EXTENSIONS = set(['jpg'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
 
 
@app.route('/predict')
def upload_test():
    return render_template('index.html')
 
 
# 上传文件
@app.route('/predict', methods=['POST'], strict_slashes=False)
def api_upload():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    predict_file_dir = os.path.join(basedir, app.config['PREDICT_FOLDER'])
    if not os.path.exists(predict_file_dir):
       os.makedirs(predict_file_dir)
        
    f = request.files['photo']
    if f and allowed_file(f.filename):
        fname = secure_filename(f.filename)
        print fname
        ext = fname.rsplit('.', 1)[1]
        #UUID
        uid = str(uuid.uuid4())
        suid = ''.join(uid.split('-'))
        
        new_filename = suid + '_' + fname.rsplit('.', 1)[0] +'.' + ext
        #new_filename = fname.rsplit('.', 1)[0] + '_predict' + '.' + ext
        f.save(os.path.join(file_dir, new_filename))

        #predict
        out_json = {"data":[]}
        imagespath = os.path.join(file_dir, new_filename)
        out_json = predict(os.path.join(file_dir, new_filename))
        #plot
        out_imagespath = os.path.join(predict_file_dir, new_filename)
        plot(imagespath,out_json,out_imagespath)
        #show
        #return jsonify({"success": 0, "msg": "上传成功"})
        ############
        #image_data = open(os.path.join(predict_file_dir, '%s' % new_filename), "rb").read()
        #response = make_response(image_data)
        #response.headers['Content-Type'] = 'image/png'
        #return response
        ################
        global PREDICTED_IMAGE
        PREDICTED_IMAGE = new_filename
        return render_template('index.html')
        ################
    else:
        return jsonify({"error": 1001, "msg": "上传失败"})
 
@app.route('/download/<string:filename>', methods=['GET'])
def download(filename):
    if request.method == "GET":
        if os.path.isfile(os.path.join('upload', filename)):
            return send_from_directory('upload', filename, as_attachment=True)
        pass
    
    
# show photo
@app.route('/show/<string:filename>', methods=['GET'])
def show_photo(filename):
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if request.method == 'GET':
        if filename is None:
            pass
        else:
            image_data = open(os.path.join(file_dir, '%s' % filename), "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        pass
        
# show photo
@app.route('/predict/', methods=['GET'])
def show_predict_photo():
    #<string:filename>  af961ebd197c45d68ff414b9cbaccafa_180627030112.jpg 
    filename = PREDICTED_IMAGE
    print filename
    image_data = open(os.path.join('predict', filename), "rb").read()
    response = make_response(image_data)
    response.headers['Content-Type'] = 'image/png'
    return response

        
 
def predict(imagepath):
    out_json = {"data":[]}
    if not os.path.exists(imagepath):
        logger.warning('the image is not exists!!!')
        return out_json
    else:
            #img_np = misc.imread(imagepath)
        img_np = cv2.imread(imagepath)  #read image by cv2 ,the same as /tool/test_net.py
    predict_datalist = mm.predict(img_np)
    info_str = ''
    if len(predict_datalist) > 0:
        logger.info('the images predict completed!!!')
        res_log = []
        res_log.append(info_str)
        for i in range(len(predict_datalist)):
            single_data = {}
            single_data = predict_datalist[i]
            res_log.append(single_data['cls'])
        logger.info(res_log)
        out_json["data"] = predict_datalist
    else:
        logger.warning('the images has not right bbox!!!')
    return out_json

def plot(imagepath,out_json,out_imagespath):
    img = cv2.imread(os.path.join(imagepath))
    data= {}
    data = out_json
    datalist = []
    datalist = data['data']
    print(len(datalist))
    for j in xrange(len(datalist)):
        singledata = {}
        boxdict = {}
        singledata = datalist[j]
        boxdict = singledata['bbox']
        xmin = boxdict['xmin']
        ymin = boxdict['ymin']
        xmax = boxdict['xmax']
        ymax = boxdict['ymax']
        cv2.rectangle(img, (xmin,ymin), (xmax,ymax),(0,255,0))
        
        font= cv2.FONT_HERSHEY_SIMPLEX
        strname = singledata['cls']
        strscore = singledata['score']
        #print (type(strscore))
        print (strscore)
        cv2.putText(img, strname + str(strscore) + '(' + str(xmax - xmin) + ',' + str(ymax - ymin) + ')', (xmin,ymin-10), font, 1,(0,0,255),2)
    print(out_imagespath)
    cv2.imwrite(out_imagespath, img)
    
    
    
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
    app.run(host="0.0.0.0",port=8080,debug=True)