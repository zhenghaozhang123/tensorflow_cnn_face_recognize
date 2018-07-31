# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 16:15:02 2018

@author: MR.ZHENG
"""

output = cnnLayer()  
predict = tf.argmax(output, 1)  

saver = tf.train.Saver()  
sess = tf.Session()  
saver.restore(sess, tf.train.latest_checkpoint('.'))  

def is_my_face(image):  
    res = sess.run(predict, feed_dict={x: [image/255.0], keep_prob_5:1.0, keep_prob_75: 1.0})  
    if res[0] == 1:  
        return True  
    else:  
        return False  

#使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()

cam = cv2.VideoCapture(0)  

while True:  
    _, img = cam.read()  
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    if not len(dets):
        #print('Can`t get face.')
        cv2.imshow('img', img)
        key = cv2.waitKey(30) & 0xff  
        if key == 27:
            sys.exit(0)

    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img[x1:y1,x2:y2]
        # 调整图片的尺寸
        face = cv2.resize(face, (size,size))
        print('Is this my face? %s' % is_my_face(face))

        cv2.rectangle(img, (x2,x1),(y2,y1), (255,0,0),3)
        cv2.imshow('image',img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)

sess.close() 