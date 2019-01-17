# flip according to the relative intensity of the cells to the background
# Take the .png files from the whole dataset
# put original .png into train_image
# put PCA corrected .png into train_images_flip
import cv2
import glob
import numpy as np

import os

all_input=glob.glob('./data/stage1_train/*/images/*')
os.system('rm -rf ./data/train_image_flip')
os.system('mkdir ./data/train_image_flip')

os.system('rm -rf ./data/train_image')
os.system('mkdir ./data/train_image')


for the_image in all_input:
    t=the_image.split('/')
    t=t[-1].split('.png')
    if (len(t)>2):
        id=t[0]+'.png'
    else:
        id=t[0]


    a=cv2.imread(the_image)
    image_avg=np.mean(a)

### get all blobs
    thecount=0
    thesum=0
    all_masks=glob.glob(('./data/stage1_train/'+id+'/masks/*'))
    for the_mask in all_masks:
        mmm=cv2.imread(the_mask)[:,:,0]
        b=np.zeros((a.shape[0:2]))
        b[mmm>127]=1
        c1=np.sum(a[:,:,0][mmm>127])/float(np.sum(b))
        c2=np.sum(a[:,:,1][mmm>127])/float(np.sum(b))
        c3=np.sum(a[:,:,2][mmm>127])/float(np.sum(b))

        c=(c1+c2+c3)/3.0
        if ((c1>c2) and (c1>c3)):
            maxchannel=0
        if ((c2 >c3) and (c2>c1)):
            maxchannel=1
        if ((c3 >c1) and (c3>c2)):
            maxchannel=2

        thesum=thesum+c
        thecount=thecount+1
    cf=thesum/(thecount+0.1)
    ## determine if bw
#    maxdiff=np.max(a[:,:,0]-a[:,:,1])
#    print(the_image,image_avg,cf)
    if (image_avg>cf):
        anew=255-a
        anew=anew-anew.min()
    else:
        anew=a
    #a= cv2.cvtColor(a, cv2.COLOR_RGB2GRAY )

## convert with max channel
    
    cv2.imwrite(('./data/train_image/'+id+'.png'), a)
    cv2.imwrite(('./data/train_image_flip/'+id+'.png'),anew)

