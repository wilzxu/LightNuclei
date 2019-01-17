import numpy as np
import cv2
import glob

all_train=glob.glob('./data/train_image/*')

for the_train in all_train:
    print(the_train)
    image=cv2.imread(the_train,-1)

    (y,x,z)=image.shape

    new=np.zeros((1536,1536,z))
    ystart=int((1536-y)/2)
    yend=int((1536-y)/2+y)
    xstart=int((1536-x)/2)
    xend=int((1536-x)/2+x)

    new[ystart:yend,xstart:xend,:]=image
    t=the_train.split('/')
    name=t[-1]
    new_name='./data/train_image_pad/'+name
    cv2.imwrite(new_name,new)

   

