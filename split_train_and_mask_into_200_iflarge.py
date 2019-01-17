# crop large images and corresponding masks into max(200, x) X max(200, y) patches
# output stored in ./data/stage1_train_200_iflarge/
import cv2
import numpy as np
import glob
import os
import copy

os.system('rm -rf ./data/stage1_train_200_iflarge')

os.system('mkdir ./data/stage1_train_200_iflarge')

all_set=glob.glob('./data/stage1_train/*')


for the_set in all_set:
	table=the_set.split('/')
	name=table[-1]
	the_image=the_set+'/images/'+name+'.png'

	all_label=glob.glob(the_set+'/masks/*')
	if (len(all_label)<101):

		the_newset=the_set.replace('stage1_train','stage1_train_200_iflarge')
		os.system(('cp -rf '+the_set+' '+the_newset))
	else:
		print(len(all_label))
		image_img=cv2.imread(the_image)
		(y,x,z)=image_img.shape
		print(image_img.shape)
		x_section=int(x/200)+1
		y_section=int(y/200)+1
		x_i=0;
		while (x_i<x_section):
			y_i=0
			while (y_i<y_section):
				x_start=int(x_i*200)
				x_end=int((x_i+1)*200)
				if (x_end>x):
					x_end=int(x)
					x_start=int(x-200)
				y_start=int(y_i*200)
				y_end=int((y_i+1)*200)
				if (y_end>y):
					y_end=int(y)
					y_start=int(y-200)

				exist=0
				for the_label in all_label:
					label_img=cv2.imread(the_label,-1)
					patch_label=label_img[y_start:y_end,x_start:x_end]
					if (np.max(np.sum(patch_label,axis=0))>1000 and np.max(np.sum(patch_label,axis=1))>1000):
						command='mkdir -p ./data/stage1_train_200_iflarge/'+str(y_i)+'_'+str(x_i)+'_'+name+'/masks'
						os.system(command)
						command='mkdir -p ./data/stage1_train_200_iflarge/'+str(y_i)+'_'+str(x_i)+'_'+name+'/images'
						os.system(command)
						label_output='./data/stage1_train_200_iflarge/'+str(y_i)+'_'+str(x_i)+'_'+name+'/masks/'+str(exist)+'.png'
						cv2.imwrite(label_output,patch_label)

						exist=exist+1

				if (exist >0):
					image_output='./data/stage1_train_200_iflarge/'+str(y_i)+'_'+str(x_i)+'_'+name+'/images/'+str(y_i)+'_'+str(x_i)+'_'+name+'.png'
					print(image_output)

					patch_image=image_img[y_start:y_end,x_start:x_end,:]
					cv2.imwrite(image_output,patch_image)

				y_i=y_i+1
			x_i=x_i+1
			
			#	break

