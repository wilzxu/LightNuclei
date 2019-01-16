## LightNuclei: Simultaneous Nuclei Counting and Segmentation by a Light-weighted Deep Learning Model

This is a software for automated nuclei detection that is based on Yuanfang's gold medal solution in the [2018 International Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018).

Please contact (gyuanfan@umich.edu) if you have any questions or suggestions.

<p align="left">
<img src="https://github.com/wilzxu/LightNuclei/blob/master/fig1a.png" width="700">
</p>


---

## Installation
Git clone LightNuclei:
```
git clone https://github.com/GuanLab/LightNuclei.git
```

## Dependency
* [python](https://www.python.org) (3.6.5)
* [tensorflow (or tensorflow-gpu)](https://www.tensorflow.org) (1.9.0) 
* [keras](https://keras.io/) (2.6.1)

## Examples
## 1. Use pre-trained model for prediction:
* Step 1. Test set augmentation and prediction:

```
python test_nuclei.py list_test ./weight/pretrained.h5
```

* list_test: a list of files that contains the images you need to make prediction on.
* pretrained.h5: weight of a pretrained model. The model used here is trained as describe in [link_to_paper]().

This step will generate a set of folders named as `vis_0`, `vis_0a`, `vis_1`, `vis_1a`, `vis_2`, `vis_2a`, `vis_3`, `vis_3a`. Each folder contains a set of images that is a rotation/flip variant of the original test image set. The prediction is visulized in these images as binary mask. 

* Step 2. Assemble the rotation/flip variants to make final prediction

```
python assemble.py
```

This step will generate a file named 'prediction.csv' in which each entry corresponds to an instance mask that is run-length encoded. Formatting details can be found [here](https://www.kaggle.com/c/data-science-bowl-2018)


## 2. Cross-validation setting:
The complete dataset for training can be downloaded from [2018 International Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018).

* Step 1. Split the data set:
```
perl step1_split.pl seed
```
* seed: random seed for data split.

This step will generate a set of list files that split 80% of the images for training and 20% for testing (`list_test`). Furthermore, The training set is randomly split into nested training (`list_train_x.1`) and validation set(`list_train_x.2`) for five times.

* Step 2. Training stage:
```
python train_nuclei.py
```

* Step 3. Prediction on test set:
```
python test_nuclei.py /path/to/weight
```

* Step 4. Assemble the rotation/flip variants to make final prediction
```
python assemble.py
```

* Step 5. Evaluation
```
python eval.py
```
This step outputs the mAP for each image in 'eval.csv'.
