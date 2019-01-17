import numpy as np
import cv2
import glob

all_train=glob.glob('./data/train_label/*')

for the_train in all_train:
    print(the_train)
    image=cv2.imread(the_train, 0)

    (y,x)=image.shape

    new=np.zeros((1536,1536))
    ystart=int((1536-y)/2)
    yend=int((1536-y)/2+y)
    xstart=int((1536-x)/2)
    xend=int((1536-x)/2+x)

    new[ystart:yend,xstart:xend]=image
    t=the_train.split('/')
    name=t[-1]
    new_name='./data/train_label_pad/'+name
    cv2.imwrite(new_name,new)

   

