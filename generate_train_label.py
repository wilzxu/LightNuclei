import cv2
import numpy as np
import glob
all_file=glob.glob('./data/split_label/*')

for the_file in all_file:
    FILE=open(the_file,'r')
    t=the_file.split('/')
    name=t[-1]
    image_path='./data/stage1_train/'+name+'/images/'+name+'.png'
    print(image_path)

    image=cv2.imread(image_path, -1)
    y,x,z=image.shape
    print(y,x)

    total=y*x
    label_img=np.zeros(total)
    for line in FILE:
        line=line.rstrip()
        table=line.split(',')
        label=table[1].split(' ')
        j=0
        while (j<len(label)):
            position=int(label[j])
            length=int(label[j+1])
            k=position-1
            print(k,length)
            while (k<(position-1+length)):
                label_img[k]=255

                k=k+1

            j=j+2
    label_img=label_img.reshape((x,y))
    label_img=np.rot90(label_img, k=3)
    label_img=cv2.flip(label_img,1)
    output_name='./data/train_label/'+name+'.png'
    cv2.imwrite(output_name,label_img)



