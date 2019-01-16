
# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
# 
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster. 

# In[1]:


import rle
import os
import sys
import random 
import math
import re
import time
import numpy as np
import cv2
import skimage.color
import skimage.io
import glob 

dir2=glob.glob('vis_*')

#dir1=glob.glob('vsssis_*')

final_dir=dir2

print(len(final_dir))


os.system('rm -rf vis')
os.system('mkdir vis')
FILE=open('list_test_final','r')
RESULT=open('submission.csv','w')
for line in FILE:

    line=line.rstrip()
# Load a single image and its associated masks
    table=line.split('\t')
    img=cv2.imread(table[0])
    (height,width,z)=img.shape
    table=table[0].split('/')
    table=table[-1].split('.png')
    id = table[0]
    masks=np.zeros((0,height,width))
    dir_id=0
    for the_dir in final_dir:
        the_dir_name=the_dir+'/{}/*.png'
        masks_1 = the_dir_name.format(id)
        #print(masks_1)
        try:
            masks_1 = skimage.io.imread_collection(masks_1).concatenate()
        except:
            masks_1=np.zeros((0,height,width))

        all_masks=masks_1
        if (len(all_masks)>0):
            occlusion = np.logical_not(all_masks[-1, :, :]).astype(np.uint8)
        count=all_masks.shape[0]
        for i in range(count-2, -1, -1):
            all_masks[i, :, :] = all_masks[i, :, :] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(all_masks[i, :, :]))

        masks_1=all_masks
    
        num_masks = masks.shape[0]
        print(dir_id,masks.shape, img.shape)
        labels = np.zeros((height, width), np.uint16)
        for index in range(0, num_masks):
            labels[masks[index] > 0.5] = index + 1

        labels_1 = np.zeros((height, width), np.uint16)
        num_masks_1 = masks_1.shape[0]
        #print(masks_1.shape, img.shape)

        for index in range(0, num_masks_1):
            labels_1[masks_1[index] > 0] = index + 1

#       masks_objects = len(np.unique(labels))
#       masks_1_objects = len(np.unique(labels_1))
        #this is the line needs to be tested

        intersection = np.histogram2d(labels.flatten(), labels_1.flatten(), bins=((num_masks+1), (num_masks_1+1)))[0]
        #print(num_masks, num_masks_1, intersection.shape)
        #print(np.argmax(intersection,axis=0).shape)
        #print(np.argmax(intersection,axis=0))
    
        #print(masks_1.max())
        try:
            print(masks.max())
        except:
            pass
        i=1
        while (i<=num_masks_1):
            if np.argmax(intersection,axis=0)[i]==0:
                if (masks_1[i-1,:,:].max()>0):
                    masks=np.append(masks,masks_1[i-1,:,:].reshape([1,height,width]),axis=0)
            else:
                masks[np.argmax(intersection,axis=0)[i]-1,:,:]=masks[np.argmax(intersection,axis=0)[i]-1,:,:]+masks_1[i-1,:,:]
                pass
            i=i+1
            #print(masks.max())
            pass
        dir_id=dir_id+1
        
    print(masks.max())
    all_masks=masks/float(dir_id-0.5)
    all_masks[all_masks>=0.5]=1
    all_masks[all_masks<0.5]=0
    (z,y,x)=all_masks.shape
    new_all_masks=np.zeros((y,x,z))
    i=0
    while (i<z):
        new_all_masks[:,:,i]=all_masks[i,:,:]
        i=i+1

    if (z>0):
        all_masks=new_all_masks
        occlusion = np.logical_not(all_masks[:, :, -1]).astype(np.uint8)
        count=all_masks.shape[2]
        for i in range(count-2, -1, -1):
            all_masks[:, :, i] = all_masks[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(all_masks[:, :, i]))
            all_sum=np.sum(all_masks,axis=2)
    else:
        all_masks=np.zeros((0,0,0))
    


    if (len(all_masks)>0):
        (ytmp,xtmp,ztmp)=all_masks.shape
        b=np.zeros((ytmp,xtmp,3))
        iiii=0
        while (iiii<ztmp):
                x=all_masks[:,:,iiii]
                b[:,:,0][x>0.5]=255*random.random()
                b[:,:,1][x>0.5]=255*random.random()
                b[:,:,2][x>0.5]=255*random.random()
                if (np.sum(x)>3):
                    the_t=table[0].split('/')
                    the_t=the_t[-1].split('.png')
                    the_name=the_t[0]
                    RESULT.write('%s,' % the_name)

    #               x=np.rot90(x, k=3)
    #               x=cv2.flip(x,1)
                    a=rle.rle_encoding(x)
       #             print(a)
                    iii=0
                    RESULT.write('%d' % a[iii])
                    iii=iii+1
                    while (iii<len(a)):
                         RESULT.write(' %d' % a[iii])
                         iii=iii+1
                    RESULT.write('\n')
                iiii=iiii+1
        the_map=np.concatenate((img,b),axis=1)

        cv2.imwrite('vis/'+id+'.png',the_map)
RESULT.close()

    
    

