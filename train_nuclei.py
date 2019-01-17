
# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
# 
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster. 

# In[1]:



import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage.color
import skimage.io
import glob 

from config import Config
import utils
import model as modellib
from model import log

#get_ipython().run_line_magic('matplotlib', 'inline')

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "../mask_rcnn_coco.h5")
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
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 1280

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64,128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 800

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 3000
    #STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 500
    
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
        #    bg_color, shapes = self.random_image(height, width)
        #    self.add_image("shapes", image_id=i, path=None,
        #                   width=width, height=height,
        #                   bg_color=bg_color, shapes=shapes)

    #def load_image(self, image_id):
    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in emage_info.
        """
        #image = skimage.io.imread(self.image_info[image_id]['path'])
        image = cv2.imread(self.image_info[image_id]['path'])
        image=(image-image.min())/(image.max()-image.min())*255
        return image
#########################
#why need to rewrite this
#########################
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
#        info = self.image_info[image_id]

        the_path=self.image_info[image_id]['path']
        the_t=the_path.split('/')
        the_t=the_t[-1].split('.png')
        if (len(the_t)>2):
            the_name=the_t[0]+'.png'
        else:
            the_name=the_t[0]
        print(the_name)
        split_mask=glob.glob(('./data/train_image_flip_iflarge_min200/'+the_name+'/masks/*'))
##################################################
# how many files in split_mask, number of classes?
##################################################
       # shapes = info['shapes']
        mmm=cv2.imread(split_mask[0],-1)
        (y,x)=mmm.shape
        mask=np.zeros((y,x,len(split_mask)))
        iii=0
        for the_mask in split_mask:
            mmm=cv2.imread(the_mask,-1)  ## flag: <0 Return the loaded image as is (with alpha channel).
            mask[:,:,iii]=mmm
            iii=iii+1
        count=len(split_mask)

        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
            ## all the previous masked class should be "occluded" for future masks

        # Map class names to class IDs.
        #class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        class_ids=np.ones(len(split_mask))
        return mask, class_ids.astype(np.int32)



# In[5]:


# Training dataset
dataset_train_1 = ShapesDataset()
dataset_train_1.load_shapes('list_train_1.1', config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train_1.prepare()

# Validation dataset
dataset_val_1 = ShapesDataset()
dataset_val_1.load_shapes('list_train_1.2', config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val_1.prepare()

# Training dataset
dataset_train_2 = ShapesDataset()
dataset_train_2.load_shapes('list_train_2.1', config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train_2.prepare()

# Validation dataset
dataset_val_2 = ShapesDataset()
dataset_val_2.load_shapes('list_train_2.2', config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val_2.prepare()


# Training dataset
dataset_train_3 = ShapesDataset()
dataset_train_3.load_shapes('list_train_3.1', config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train_3.prepare()

# Validation dataset
dataset_val_3 = ShapesDataset()
dataset_val_3.load_shapes('list_train_3.2', config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val_3.prepare()

# Training dataset
dataset_train_4 = ShapesDataset()
dataset_train_4.load_shapes('list_train_4.1', config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train_4.prepare()

# Validation dataset
dataset_val_4 = ShapesDataset()
dataset_val_4.load_shapes('list_train_4.2', config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val_4.prepare()


# Training dataset
dataset_train_5 = ShapesDataset()
dataset_train_5.load_shapes('list_train_5.1', config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train_5.prepare()

# Validation dataset
dataset_val_5 = ShapesDataset()
dataset_val_5.load_shapes('list_train_5.2', config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val_5.prepare()




model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# In[7]:

model.load_weights(COCO_MODEL_PATH, by_name=True,
           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
            "mrcnn_bbox", "mrcnn_mask"])
###############################
## which layers were taken out?
###############################


# Which weights to start with?
#init_with = "coco"  # imagenet, coco, or last

#model.load_weights(model.find_last()[1], by_name=True)
#if init_with == "imagenet":
#    model.load_weights(model.get_imagenet_weights(), by_name=True)
#elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
#elif init_with == "last":
    # Load the last model you trained and continue training

# Passing layers="all" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train_1, dataset_val_1, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='all')


model.train(dataset_train_2, dataset_val_2, 
            learning_rate=config.LEARNING_RATE, 
            epochs=2, 
            layers='all')


model.train(dataset_train_3, dataset_val_3, 
            learning_rate=config.LEARNING_RATE, 
            epochs=3, 
            layers='all')

model.train(dataset_train_4, dataset_val_4, 
            learning_rate=config.LEARNING_RATE, 
            epochs=4, 
            layers='all')

model.train(dataset_train_5, dataset_val_5, 
            learning_rate=config.LEARNING_RATE, 
            epochs=5, 
            layers='all')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train_1, dataset_val_1, 
            learning_rate=config.LEARNING_RATE *0.1,
            epochs=6, 
            layers="all")

model.train(dataset_train_2, dataset_val_2, 
            learning_rate=config.LEARNING_RATE *0.1,
            epochs=7, 
            layers="all")

model.train(dataset_train_3, dataset_val_3, 
            learning_rate=config.LEARNING_RATE *0.1,
            epochs=8, 
            layers="all")

model.train(dataset_train_4, dataset_val_4, 
            learning_rate=config.LEARNING_RATE *0.1,
            epochs=9, 
            layers="all")

model.train(dataset_train_5, dataset_val_5, 
            learning_rate=config.LEARNING_RATE *0.1,
            epochs=10, 
            layers="all")




model.train(dataset_train_1, dataset_val_1, 
            learning_rate=config.LEARNING_RATE*0.5, 
            epochs=11, 
            layers='all')


model.train(dataset_train_2, dataset_val_2, 
            learning_rate=config.LEARNING_RATE*0.4, 
            epochs=12, 
            layers='all')


model.train(dataset_train_3, dataset_val_3, 
            learning_rate=config.LEARNING_RATE*0.4, 
            epochs=13, 
            layers='all')

model.train(dataset_train_4, dataset_val_4, 
            learning_rate=config.LEARNING_RATE*0.3, 
            epochs=14, 
            layers='all')

model.train(dataset_train_5, dataset_val_5, 
            learning_rate=config.LEARNING_RATE*0.3, 
            epochs=15, 
            layers='all')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train_1, dataset_val_1, 
            learning_rate=config.LEARNING_RATE*0.2,
            epochs=16, 
            layers="all")

model.train(dataset_train_2, dataset_val_2, 
            learning_rate=config.LEARNING_RATE *0.2,
            epochs=17, 
            layers="all")

model.train(dataset_train_3, dataset_val_3, 
            learning_rate=config.LEARNING_RATE *0.1,
            epochs=18, 
            layers="all")

model.train(dataset_train_4, dataset_val_4, 
            learning_rate=config.LEARNING_RATE *0.1,
            epochs=19, 
            layers="all")

model.train(dataset_train_5, dataset_val_5, 
            learning_rate=config.LEARNING_RATE *0.09,
            epochs=20, 
            layers="all")



model.train(dataset_train_1, dataset_val_1, 
            learning_rate=config.LEARNING_RATE*0.09, 
            epochs=21, 
            layers='all')


model.train(dataset_train_2, dataset_val_2, 
            learning_rate=config.LEARNING_RATE*0.09, 
            epochs=22, 
            layers='all')


model.train(dataset_train_3, dataset_val_3, 
            learning_rate=config.LEARNING_RATE*0.08, 
            epochs=23, 
            layers='all')

model.train(dataset_train_4, dataset_val_4, 
            learning_rate=config.LEARNING_RATE*0.08, 
            epochs=24, 
            layers='all')

model.train(dataset_train_5, dataset_val_5, 
            learning_rate=config.LEARNING_RATE*0.07, 
            epochs=25, 
            layers='all')

