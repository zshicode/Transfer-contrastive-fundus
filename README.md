# Transfer contrastive learning for ocular disease diagnosis using medical images

This repo proposes a transfer learning framework for ocular disease diagnosis using medical images. Two datasets, ODIR and GOALS, are retrieved from fundus images and OCT respectively. We adopt the foundation model in the field of computer vision, i.e., DINOv2 and DINOv3, for representation learning of medical images.

For ODIR dataset as classification task, each image is compressed to an embedding vector using pretrained DINO model. Two embeddings of fundus images from different eyes of one patient are fused using contrastive learning, and focal loss is also adopted to tackle the imbalanced classification. We also implement a baseline model using LightGBM by simply concatenating left-eye embedding and right-eye embedding as feature vector. 

For GOALS dataset as classification task, one image is randomly masked twice to obtain two masked images for two branches of contrastive framework respectively. 

For GOALS dataset as segmentation task, since DINOv3 not only outputs embedding vector, but also outputs last hidden state, which generates embedding vector for each patch, we implement a patch-representation-patch UNet. The representations learned by UNet are trained to align the representations learned by DINOv3 under a teacher-student framework, and the alignment loss is formed as contrastive learning. 

This repo evaluates representation learning's capability of pretrained foundation models, and develops a deployable, lightweight transfer learning solution from pretrained foundation models to deliver fast inference and great potential for real‑world computer vision applications.

## Requirements

This repo has been tested running on Python 3.12.12, with the following libraries and their dependencies installed.

```
numpy==2.2.6
pandas==2.3.3
pytorch==2.15.0
sklearn==1.8.0
transformers==5.22.1
lightgbm==4.6.0
PIL==12.1.0
torchvision==0.29.2
```

## Usage

```
python odir.py --model dinov3
python goals.py --model dinov3
```

Model can be either DINOv2 or DINOv3 for classification, but only DINOv3 for segmentation. `goals.py` runs both of classification and segmentation.

## Datasets

### ODIR dataset

The details and download links of ODIR dataset are in https://github.com/nkicsl/OIA-ODIR and https://odir2019.grand-challenge.org/dataset. To run the code, please storage the images in `./ODIRImages`, and storage the labels in `label.csv` of train/val/test set respectively. The directory structure is

```
ODIRImages
|---train
    |---1449_left.jpg
    |---1449_right.jpg
    |---...
    |---labels.csv
|---val
    |---...
|---test
    |---...
```

The format of `label.csv` is

```
ID,label
1449,O
1500,G
...,...
```

Note that the original dataset includes some multi-label samples. To simplify the task, we only select some of single-label samples for experiments.

### GOALS dataset

The details and download links of GOALS dataset are in http://hdmilab.cn/ichallenge and https://aistudio.baidu.com/competition/detail/783/0/introduction. To run the code, please storage the images in `./GOALSImages`, and storage the labels in `label.csv` of train/val/test set respectively. The directory structure is

```
GOALSImages
|---train
    |---12.png
    |---12_mask.png
    |---...
    |---labels.csv
|---val
    |---...
|---test
    |---...
```

The format of `label.csv` is

```
ID,label
12,1
13,0
...,...
```

GOALS dataset includes both classification and segmentation tasks. The mask images are labels for segmentation.