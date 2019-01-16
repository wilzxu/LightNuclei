import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import skimage.segmentation
import cv2

FILE=open('list_test_short','r')

for line in FILE:
    line=line.rstrip()
# Load a single image and its associated masks
    table=line.split('/')
    table=table[-1].split('.png')
    id = table[0]
    file = "../../../data/stage1_train/{}/images/{}.png".format(id,id)
    masks = "../../../data/stage1_train/{}/masks/*.png".format(id)

    image = skimage.io.imread(file)
    masks = skimage.io.imread_collection(masks).concatenate()
    height, width, _ = image.shape
    num_masks = masks.shape[0]

    # Make a ground truth label image (pixel value is index of object label)
    labels = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labels[masks[index] > 0] = index + 1


    print(id, height,width)
    preds = np.zeros((height*width), np.uint16)
    PRED=open('submission.csv','r')
    pred_index=0
    for line in PRED:
        line=line.rstrip()
        table=line.split(',')
        if (table[0] == id):
            tmp_label=table[1].split(' ')
            print(len(tmp_label))
            j=0
            while (j<len(tmp_label)):
                position=int(tmp_label[j])
                length=int(tmp_label[j+1])
                k=position-1
               # print(k,length)
                while (k<(position-1+length)):
                    preds[k]=(pred_index+1)

                    k=k+1

                j=j+2
            pred_index=pred_index+1

    preds=preds.reshape((width,height))

    preds=np.rot90(preds, k=3)
    preds=cv2.flip(preds,1)

    output_name='tmp/'+id+'.png'
    cv2.imwrite(output_name,preds)



    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(preds))
    print("Number of true objects:", true_objects)
    print("Number of predicted objects:", pred_objects)

    intersection = np.histogram2d(labels.flatten(), preds.flatten(), bins=(true_objects, pred_objects))[0]

    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(preds, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    union = area_true + area_pred - intersection

# Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

# Compute the intersection over union
    iou = intersection / union

# Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

# Loop over IoU thresholds
    prec = []
    print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp / float(tp + fp + fn)
        print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
