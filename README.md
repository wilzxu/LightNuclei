## LightNuclei: Simultaneous Nuclei Counting and Segmentation by a Light-weighted Deep Learning Model

This repository contains codes for an automated nuclei detection pipeline that is based on Yuanfang's gold medal solution in the [2018 International Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018).

Please contact (gyuanfan@umich.edu) if you have any questions or suggestions.

<p align="left">
<img src="https://github.com/wilzxu/LightNuclei/blob/master/figures/fig1a.png" width="700">
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


The complete dataset for training can be downloaded from [2018 International Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018).

Please download stage1_train.zip into `data` directory, and decompress there.

* Step 1. Preprocessing and split the data set:
```
python LightNuclei.py
```

This step will generate a set of list files that split 80% of the images for training and 20% for testing (`list_test`). Furthermore, The training set is randomly split into 80% nested training (`list_train_x.1`) and 20% validation set(`list_train_x.2`) for five times.

* Step 2. Training stage:
```
python train_nuclei.py
```

* Step 3. Prediction on test set:
```
python test_nuclei.py /path/to/weight
```

`/path/to/weight` : .h5 weight file is passed to the program. Example is `./weights/pretrained.h5`, or is produced and stored in `./logs/` after training the model in step 2.

This step will generate a set of folders named as `vis_0`, `vis_0a`, `vis_1`, `vis_1a`, `vis_2`, `vis_2a`, `vis_3`, `vis_3a`. Each folder contains a set of images that is a rotation/flip variant of the original test image set. The prediction is visualized in these images as binary mask. 


* Step 4. Assemble the rotation/flip variants to make final prediction
```
python assemble.py
```
This step will generate a file named 'prediction.csv' in which each entry corresponds to an instance mask that is run-length encoded. Formatting details can be found [here](https://www.kaggle.com/c/data-science-bowl-2018)



* Step 5. Evaluation
```
python eval.py
```
This step outputs the mAP for each image in 'eval.csv'.

Example prediction is shown below.
<p>
<imcg src="https://github.com/wilzxu/LightNuclei/blob/master/figures/fig2.png">
</p>
