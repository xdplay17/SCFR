# Supervised Contrastive Learning Based Deep Hashing with Fusion of Global and Local Features for Remote Sensing Image Retrieval
This is the code implementation for our paper "Supervised Contrastive Learning Based Deep Hashing with Fusion of Global and Local Features for Remote Sensing Image Retrieval".

# Usage
### 1. Install dependencies:

Requirements:
```
python
numpy
pytorch
torchvision
tqdm
PIL
```
### 2. Data:
You should download three data sets including [UC Merced](http://weegee.vision.ucmerced.edu/datasets/landuse.html), [AID](https://captain-whu.github.io/AID/), and NWPU-RESISC45 and put data set in the corresponding directory under `dataset`. If you want to construct your own training set and testing set, you should modify the path of images of training set, testing set and database respectively in the `txtfile/.../train.txt` , `txtfile/.../test.txt` and `txtfile/.../database.txt` for corresponding data set.

### 3. Training:
```python
python UCMD_main.py
```
