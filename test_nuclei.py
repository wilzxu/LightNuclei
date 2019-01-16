
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

from config import Config
import utils
import model as modellib
#import visualize
from model import log

#get_ipython().run_line_magic('matplotlib', 'inline')

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "../../mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# ## Configurations

# In[2]:


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 125
    IMAGE_MAX_DIM = 1280

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64,128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 10
    #STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    DETECTION_MAX_INSTANCES = 400
    
config = ShapesConfig()
config.display()


# ## Notebook Preferences

# In[3]:


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# ## Dataset
# 
# Create a synthetic dataset
# 
# Extend the Dataset class and add a method to load the shapes dataset, `load_shapes()`, and override the following methods:
# 
# * load_image()
# * load_mask()
# * image_reference()

# In[4]:


class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_shapes(self, the_file, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "nuclei")
        #self.add_class("shapes", 2, "circle")
        #self.add_class("shapes", 3, "triangle")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        FILE=open(the_file,'r')
        i=0
        for line in FILE:
            line=line.rstrip()
            table=line.split('\t')
            self.add_image(
                "shapes", image_id=i,
                path=os.path.join(table[0]),
                           width=width, height=height)

            i=i+1
        #for i in range(count):
        #   bg_color, shapes = self.random_image(height, width)
        #   self.add_image("shapes", image_id=i, path=None,
        #                  width=width, height=height,
        #                  bg_color=bg_color, shapes=shapes)

    #def load_image(self, image_id):
    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in emage_info.
        """
        #image = skimage.io.imread(self.image_info[image_id]['path'])
        image = cv2.imread(self.image_info[image_id]['path'])
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def image_name(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
#       info = self.image_info[image_id]

        the_path=self.image_info[image_id]['path']
        the_t=the_path.split('/')
        the_t=the_t[-1].split('.png')
        the_name=the_t[0]
        return(the_name)


    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
#       info = self.image_info[image_id]

        the_path=self.image_info[image_id]['path']
        the_t=the_path.split('/')
        the_t=the_t[-1].split('.png')
        the_name=the_t[0]
        #print(the_name)
        split_mask=glob.glob(('../../../data/stage1_train_pad_1024/'+the_name+'/masks/*'))

       # shapes = info['shapes']
        mask=np.zeros((1024,1024,len(split_mask)))
        iii=0
        for the_mask in split_mask:
            mmm=cv2.imread(the_mask,-1)
            mask[:,:,iii]=mmm
            iii=iii+1
        count=len(split_mask)

        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        #class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        class_ids=np.ones(len(split_mask))
        return mask, class_ids.astype(np.int32)



# In[5]:


# Training dataset


# In[6]:


# ## Ceate Model

# In[ ]:


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# In[7]:


# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)


# ## Detection

# In[11]:


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
#model_path = model.find_last()[1]
#model_path='logs/shapes20180121T1836/mask_rcnn_shapes_0002.h5'

model_path=sys.argv[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)



# In[13]:


#image_ids = np.random.choice(dataset_val.image_ids, 10)
#os.system('rm -rf vis_0')
os.system('mkdir vis_0')
#os.system('rm -rf vis_1')
os.system('mkdir vis_1')
#os.system('rm -rf vis_2')
os.system('mkdir vis_2')
#os.system('rm -rf vis_3')
os.system('mkdir vis_3')

FILE=open('list_test','r')
RESULT=open('submission.csv','w')
for line in FILE:
    line=line.strip()
    table=line.split('\t')
    image=cv2.imread(table[0])
    t=table[0].split('/')
    t=t[-1].split('.png')
    id=t[0]

    # Load image and ground truth data
    # Run object detection
    # assemble 4 results
    results = model.detect([image], verbose=1)
    r = results[0]
    all_masks=np.asarray(results[0]["masks"])

    occlusion = np.logical_not(all_masks[:, :, -1]).astype(np.uint8)
    count=all_masks.shape[2]
    for i in range(count-2, -1, -1):
        all_masks[:, :, i] = all_masks[:, :, i] * occlusion
        occlusion = np.logical_and(occlusion, np.logical_not(all_masks[:, :, i]))
    i=0
    while (i < count):
        command='mkdir -p vis_0/'+id
        os.system(command)
        output_name='vis_0/'+id+'/'+str(i)+'.png'
        try:
             if (all_masks[:,:,i].max()>0):
                 cv2.imwrite(output_name,all_masks[:,:,i])
        except:
             #print(i)
             #print(all_masks[:,:,i])
            pass
        i=i+1

    imagea=np.fliplr(image)
    resultsa = model.detect([imagea], verbose=1)
    all_masks_a=np.asarray(resultsa[0]["masks"])
    all_masks_a=np.fliplr(all_masks_a)

    occlusion = np.logical_not(all_masks_a[:, :, -1]).astype(np.uint8)
    count=all_masks_a.shape[2]
    for i in range(count-2, -1, -1):
        all_masks_a[:, :, i] = all_masks_a[:, :, i] * occlusion
        occlusion = np.logical_and(occlusion, np.logical_not(all_masks_a[:, :, i]))
    i=0
    while (i < count):
        command='mkdir -p vis_0a/'+id
        os.system(command)
        output_name='vis_0a/'+id+'/'+str(i)+'.png'
        try:
             if (all_masks_a[:,:,i].max()>0):
                 cv2.imwrite(output_name,all_masks_a[:,:,i])
        except:
             #print(i)
             #print(all_masks[:,:,i])
            pass
        i=i+1





    image1=np.rot90(image,k=1,axes=(0,1))
    results1 = model.detect([image1], verbose=1)
    all_masks_1=np.asarray(results1[0]["masks"])
    all_masks_1=np.rot90(all_masks_1,k=3,axes=(0,1))


    occlusion = np.logical_not(all_masks_1[:, :, -1]).astype(np.uint8)
    count=all_masks_1.shape[2]
    for i in range(count-2, -1, -1):
        all_masks_1[:, :, i] = all_masks_1[:, :, i] * occlusion
        occlusion = np.logical_and(occlusion, np.logical_not(all_masks_1[:, :, i]))
    i=0
    while (i < count):
        command='mkdir -p vis_1/'+id
        os.system(command)
        output_name='vis_1/'+id+'/'+str(i)+'.png'
        try:
             if (all_masks_1[:,:,i].max()>0):
                 cv2.imwrite(output_name,all_masks_1[:,:,i])
        except:
             #print(i)
             #print(all_masks_1[:,:,i])
            pass
        i=i+1

    image1a=np.fliplr(image1)
    results1a = model.detect([image1a], verbose=1)
    all_masks_1a=np.asarray(results1a[0]["masks"])
    all_masks_1a=np.fliplr(all_masks_1a)
    all_masks_1a=np.rot90(all_masks_1a,k=3,axes=(0,1))

    occlusion = np.logical_not(all_masks_1a[:, :, -1]).astype(np.uint8)
    count=all_masks_1a.shape[2]
    for i in range(count-2, -1, -1):
        all_masks_1a[:, :, i] = all_masks_1a[:, :, i] * occlusion
        occlusion = np.logical_and(occlusion, np.logical_not(all_masks_1a[:, :, i]))
    i=0
    while (i < count):
        command='mkdir -p vis_1a/'+id
        os.system(command)
        output_name='vis_1a/'+id+'/'+str(i)+'.png'
        try:
             if (all_masks_1a[:,:,i].max()>0):
                 cv2.imwrite(output_name,all_masks_1a[:,:,i])
        except:
             #print(i)
             #print(all_masks[:,:,i])
            pass
        i=i+1


    
    image2=np.rot90(image,k=2,axes=(0,1))
    results2 = model.detect([image2], verbose=1)
    all_masks_2=np.asarray(results2[0]["masks"])
    all_masks_2=np.rot90(all_masks_2,k=2,axes=(0,1))


    occlusion = np.logical_not(all_masks_2[:, :, -1]).astype(np.uint8)
    count=all_masks_2.shape[2]
    for i in range(count-2, -1, -1):
        all_masks_2[:, :, i] = all_masks_2[:, :, i] * occlusion
        occlusion = np.logical_and(occlusion, np.logical_not(all_masks_2[:, :, i]))
    i=0
    while (i < count):
        command='mkdir -p vis_2/'+id
        os.system(command)
        output_name='vis_2/'+id+'/'+str(i)+'.png'
        try:
             if (all_masks_2[:,:,i].max()>0):
                 cv2.imwrite(output_name,all_masks_2[:,:,i])
        except:
            pass
        i=i+1

    
    image2a=np.fliplr(image2)
    results2a = model.detect([image2a], verbose=1)
    all_masks_2a=np.asarray(results2a[0]["masks"])
    all_masks_2a=np.fliplr(all_masks_2a)
    all_masks_2a=np.rot90(all_masks_2a,k=2,axes=(0,1))

    occlusion = np.logical_not(all_masks_2a[:, :, -1]).astype(np.uint8)
    count=all_masks_2a.shape[2]
    for i in range(count-2, -1, -1):
        all_masks_2a[:, :, i] = all_masks_2a[:, :, i] * occlusion
        occlusion = np.logical_and(occlusion, np.logical_not(all_masks_2a[:, :, i]))
    i=0
    while (i < count):
        command='mkdir -p vis_2a/'+id
        os.system(command)
        output_name='vis_2a/'+id+'/'+str(i)+'.png'
        try:
             if (all_masks_2a[:,:,i].max()>0):
                 cv2.imwrite(output_name,all_masks_2a[:,:,i])
        except:
             #print(i)
             #print(all_masks[:,:,i])
            pass
        i=i+1

    image3=np.rot90(image,k=3,axes=(0,1))
    results3 = model.detect([image3], verbose=1)
    all_masks_3=np.asarray(results3[0]["masks"])
    all_masks_3=np.rot90(all_masks_3,k=1,axes=(0,1))


    occlusion = np.logical_not(all_masks_3[:, :, -1]).astype(np.uint8)
    count=all_masks_3.shape[2]
    for i in range(count-2, -1, -1):
        all_masks_3[:, :, i] = all_masks_3[:, :, i] * occlusion
        occlusion = np.logical_and(occlusion, np.logical_not(all_masks_3[:, :, i]))
    i=0
    while (i < count):
        command='mkdir -p vis_3/'+id
        os.system(command)
        output_name='vis_3/'+id+'/'+str(i)+'.png'
        try:
             if (all_masks_3[:,:,i].max()>0):
                 cv2.imwrite(output_name,all_masks_3[:,:,i])
        except:
            pass
        i=i+1
    image3a=np.fliplr(image3)
    results3a = model.detect([image3a], verbose=1)
    all_masks_3a=np.asarray(results3a[0]["masks"])
    all_masks_3a=np.fliplr(all_masks_3a)
    all_masks_3a=np.rot90(all_masks_3a,k=1,axes=(0,1))

    occlusion = np.logical_not(all_masks_3a[:, :, -1]).astype(np.uint8)
    count=all_masks_3a.shape[2]
    for i in range(count-2, -1, -1):
        all_masks_3a[:, :, i] = all_masks_3a[:, :, i] * occlusion
        occlusion = np.logical_and(occlusion, np.logical_not(all_masks_3a[:, :, i]))
    i=0
    while (i < count):
        command='mkdir -p vis_3a/'+id
        os.system(command)
        output_name='vis_3a/'+id+'/'+str(i)+'.png'
        try:
             if (all_masks_3a[:,:,i].max()>0):
                 cv2.imwrite(output_name,all_masks_3a[:,:,i])
        except:
             #print(i)
             #print(all_masks[:,:,i])
            pass
        i=i+1

    
