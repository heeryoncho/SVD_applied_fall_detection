# Applying Singular Value Decomposition on Accelerometer Data for 1D Convolutional Neural Network Based Fall Detection
This is the code for the 2019 Electronics Letters paper [Applying Singular Value Decomposition on Accelerometer Data for 1D Convolutional Neural Network Based Fall Detection](https://www.growkudos.com/publications/10.1049%25252Fel.2018.6117/reader) by Heeryon Cho and Sang Min Yoon.

![](https://github.com/heeryoncho/SVD_applied_fall_detection/blob/master/fig/LOSO_Accuracy.png)

## Requirements
This code runs with:
* Ubuntu 16.04
* NVIDIA GeForce GTX 960
* TensorFlow version 1.5.0
* Python 2.7.12

## Downloading Fall Recognition Datasets
Please download the following three Human Activity Recognition benchmark datasets (which include fall activities) from the respective sites and place them inside the 'raw_dataset' folder.
* [SisFall] (http://sistemic.udea.edu.co/en/research/projects/english-falls/)
* [UMAFall] (https://figshare.com/articles/UMA_ADL_FALL_Dataset_zip/4214283)
* [UniMiB] (http://www.sal.disco.unimib.it/technologies/unimib-shar/)

## Remarks
1. Place the unzipped benchmark dataset into the 'raw_dataset' folder. Refer to the directory structure given in the 'raw_dataset' folder's readme.txt. (Note: For UniMiB data, you first need to convert the .mat files to .csv files using the 'convert_mat2csv.py' code located inside the UniMiB folder.)
2. Generate processed (raw or dimension reduced) data by executing codes marked 'gen_XXX.py'. A sample data, generated using the code 'umafall/gen_umafall_dataset_kpca.py', is given inside 'umafall/data/' folder (Two pickle files: X_umafall_kpca.p & y_umafall_kpca.p).
3. Build and evaluate 1D CNN models. A sample model, generated using the code 'umafall/umafall_kpca_conv1d_10.py', is given inside 'umafall/model/umafall_kpca_conv1d_10/' folder.

## Citation
If you find this useful, please cite our work as follows:
```
@article{ChoYoon_2019ElectronicsLetters,
  author    = {Heeryon Cho and Sang Min Yoon},
  title     = {Applying Singular Value Decomposition on Accelerometer Data for 
               1D Convolutional Neural Network Based Fall Detection},
  journal   = {Electronics Letters},
  volume    = {55},
  number    = {6},
  pages     = {320--322},
  doi       = {10.1049/el.2018.6117},
  year      = {2019},
}
```
