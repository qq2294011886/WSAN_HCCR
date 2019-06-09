# Writing Style Adversarial Network forHandwritten Chinese Character Recognition
![](https://img.shields.io/badge/python-3.5%7C3.6%7C3.7-blue.svg) ![](https://img.shields.io/badge/tensorflow-1.10.0-orange.svg)![](https://img.shields.io/badge/keras-2.2.4-yellow.svg) ![](https://img.shields.io/badge/license-LGPL%203.0-green.svg)
## Requisites
We assume you have already installed and configured Keras with any of its backends. Install the other required dependencies with:
```
$ pip3 install -r requirements.txt
```
The code was last tested on Keras 2.2.4 using TensorFlow 1.10.0 as backend, h5py 2.7.1, numpy 1.14.2, Pillow 5.1.0, scikit-image 0.14.0 and scipy 1.0.0. 

## Usage

###  (Lite version, Only CASIA-HWDB1.1)

0. Download the CASIA-HWDB1.1 data set from the official locations ([HWDB1.1trn_gnt.zip](http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip) (1873 MB) and [HWDB1.1tst_gnt.zip](http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip) (471 MB)) and unzip it (it is required to decompress an archive in ALZ format). All training gnt files and testing gnt files should be placed in the same directory, respectively (for example, *HWDB1.1trn_gnt/* and *HWDB1.1tst_gnt/*). 
1. Convert the dataset into the HDF5 binary data format:
```
$ python3 1-gnt_to_dataset-lite.py HWDB1.1trn_gnt/ HWDB1.1tst_gnt/
```
2. Preprocessing the HDF5 dataset:
```
$ python3 2-process_dataset-lite.py HWDB1.1.hdf5
```
3. Train/Evaluation the model on the subset:
```
$ python3 WSAN.py train HWDB1.1_processed.hdf5
...
```
```
$ python3 WSAN.py eval HWDB1.1_processed.hdf5
...
```


###  (Full version, Training on CASIA-HWDB1.0+1.1, Evaluation on ICDAR-2013)
0. Download the CASIA-HWDB1.0+1.1 and ICDAR2013 dataset from the official locations ([HWDB1.0train_gnt.rar](http://www.nlpr.ia.ac.cn/databases/Download/feature_data/1.0train-gb1.rar) (2741 MB), [HWDB1.0test_gnt.rar](http://www.nlpr.ia.ac.cn/databases/Download/feature_data/1.0test-gb1.rar) (681 MB), [HWDB1.1trn_gnt.zip](http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip) (1873 MB) , [HWDB1.1tst_gnt.zip](http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip) (471 MB) and [ICDAR2013](http://www.nlpr.ia.ac.cn/databases/Download/competition/competition-gnt.zip) (448 MB)) and unzip it (it is required to decompress an archive in ALZ format). All training gnt files and testing gnt files should be placed in the same directory, respectively (for example, *full_trn_gnt/* and *full_tst_gnt/*). 

1. Convert the dataset into the HDF5 binary data format:
```
$ python3 1-gnt_to_dataset-full.py full_trn_gnt/ full_tst_gnt/
```
2. Preprocessing the HDF5 dataset:
```
$ python3 2-process_dataset-full.py CASIA_ICDAR.hdf5
```
3. Train/Evaluation the model on the subset:
```
$ python3 WSAN.py train CASIA_ICDAR_processed.hdf5
...
```
```
$ python3 WSAN.py eval CASIA_ICDAR_processed.hdf5

loading weight from:  model_weight/model.9727.hdf5 ...
224419/224419 [==============================] - 111s 495us/step
Evaluate on testing set   -CR_acc: 0.9727   -CR_top5_acc: 0.9968
```


## Hint
- The highest accuracy is trained by the following methods: First, training model with the Adaldeta optimizer with lr decay. When the model converges, then use SGD with ReduceLROnPlateau for fine tuning.
- We are training model with 2 gpus, so the prediction also requires two gpus (legacy issues)
- This model consumes memory because of the use of data augment. If you need a data-free version (read hdf5 data directly from disk, no memory, but slightly lower accuracy), please leave your email.