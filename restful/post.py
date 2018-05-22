#coding=utf-8
import requests
import time
import json
import base64
import cv2
from scipy import misc

num = 1
mtcnn_elapsed = 0
facenet_elapsed = 0
emotion_elapsed = 0
eye_elapsed = 0
angle_elapsed = 0
alltime = 0


start = time.time()
for i in xrange(num):
    start = time.clock()
    s = requests

    imagepath = '/opt/ligang/detectron/Detectron-master/restful/vis/806_180507070134.jpg'
    data={"data":imagepath}
    my_json_data = json.dumps(data)
    headers = {'Content-Type': 'application/json'}
    r = s.post('http://192.168.200.213:9527/user', headers=headers,data = my_json_data,)
    #print type(r)
    #print (r)
    #print type(r.json())
    print (r.json())
    print (i)
end = time.time() - start
print end

#plot
imagepath = '/data/ligang/detectron/Detectron-master/restful/vis/806_180507070134.jpg'
img = cv2.imread(imagepath)
cv2.rectangle(img, (136,63), (765,474),3)
cv2.rectangle(img, (130,50), (537,239),3)
cv2.imwrite('./001_new.jpg', img)
'''
################################################################
############################# curl #############################
curl -X POST 'http://192.168.200.213:9527/user' -d '{"data":"/opt/ligang/detectron/Detectron-master/restful/vis/785_180507070102.jpg"}' -H 'Content-Type: application/json'


curl -X POST 'http://192.168.200.213:9527/user' -d '{"data":"https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1526895699811&di=5ce6acbcfe8f1d93fe65d3ae8eb3287d&imgtype=0&src=http%3A%2F%2Fimg1.fblife.com%2Fattachments1%2Fday_130616%2F20130616_e4c0b7ad123ca263d1fcCnkYLFk97ynn.jpg.thumb.jpg"}' -H 'Content-Type: application/json'
'''