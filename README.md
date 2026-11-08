# Transfer contrastive learning for ocular disease diagnosis using fundus images

This repo proposes a transfer learning framework for ocular disease diagnosis using fundus images. We adopt the foundation model in the field of computer vision, i.e., DINOv2 and DINOv3, for representation learning of fundus images. Each image is compressed to an embedding vector using pretrained DINO model. Then, two embeddings of fundus images from different eyes of one patient are fused using contrastive learning, and focal loss is also adopted to tackle the imbalanced dataset. We also implement a baseline model using LightGBM by simply concatenating left-eye embedding and right-eye embedding as feature vector. This repo evaluates representation learning's capability of pretrained foundation models, and develops a deployable, lightweight transfer learning solution from pretrained foundation models to deliver fast inference and great potential for real‑world computer vision applications.

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
```

## Usage

```
python main.py --model dinov3
```

`args.model` can be either DINOv2 or DINOv3.

## Datasets

The details and download links of dataset are in https://github.com/nkicsl/OIA-ODIR. To run the code, please storage the images in `./ODIRImages`, and storage the labels in `label.csv`. The format of `label.csv` is

```
ID,label
1449,O
1500,G
...,...
```

Note that the original dataset includes some multi-label samples. To simplify the task, we only select some of single-label samples for experiments.

## Options

We adopt an argument parser by package  `argparse` in Python, and the options for running code are defined as follow:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--c', type=int, default=8,
                    help='Num of classes')
parser.add_argument('--d', type=int, default=384,
                    help='Num of embedding dimension')
parser.add_argument('--model', type=str, default='dinov2',
                    help='Model')                   

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
```
